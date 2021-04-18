import numpy as np
import nltk
import torch

from .attack_operations import similairty_calculation
from utils.hyper_parameters import pos_tag_filter, binary_words


def get_attack_sequences(text_ls, generator, target_model, sim_predictor,
                         orig_probs, orig_label, stop_words_set,
                         punct_re, words_re, text2=None, thres=None,
                         attack_loc=None):
    num_query = 0
    orig_prob = orig_probs.max()
    text_prime = text_ls.copy()
    len_text = len(text_prime)
    token_tags = nltk.pos_tag(text_prime, tagset='universal')
    # whether to use pos tag to find possible attack locations
    if attack_loc == 'pos_tag_filter':
        replace_indices, insert_indices, merge_indices = index_from_pos_tag(token_tags)
    elif attack_loc == 'silence_score':
        replace_indices, insert_indices, merge_indices = index_from_slience_score(
            token_tags, text_ls, orig_label, orig_probs, target_model)
    elif attack_loc == 'brutal_force':
        replace_indices = range(len(token_tags))
        insert_indices = range(1, len(token_tags))
        merge_indices = find_merge_index(token_tags)

    # if text2 is not None, maybe need to keep the exact same words in text2
    if text2 is not None:
        text2_tags = nltk.pos_tag(text2.lower().split(), tagset='universal')
        text2_tokens = set()
        for text2_tag in text2_tags:
            if text2_tag[1] in ['NOUN', 'VERB', 'ADV', 'NUM', 'ADJ']:
                text2_tokens.add(text2_tag[0])
    
    mask_inputs, mask_tokens, attack_types, pivot_indices = [], [], [], []
    # check replacement choices
    for replace_idx in replace_indices:
        if punct_re.search(text_prime[replace_idx]) is not None and \
            words_re.search(text_prime[replace_idx]) is None:
            continue
        if text_prime[replace_idx].lower() in stop_words_set:
            continue
        if thres.get('filter_adj', False) and token_tags[replace_idx][1] in ['ADJ', 'ADV']:
            continue
        if thres.get('keep_sim', False):
            if text_prime[replace_idx].lower() in binary_words:
                continue
            if text2 is not None and text_prime[replace_idx].lower() in text2_tokens:
                continue
        mask_input = text_prime.copy()
        mask_input[replace_idx] = '<mask>'
        mask_inputs.append(" ".join(mask_input))
        orig_token = text_prime[replace_idx]
        mask_tokens.append(orig_token)
        attack_types.append('replace')
        pivot_indices.append(replace_idx)

    # check insertion choices
    for insert_idx in insert_indices:
        if thres.get('keep_sim', False):
            if text_prime[insert_idx-1].lower() in binary_words:
                continue
        mask_input = text_prime.copy()
        mask_input.insert(insert_idx, '<mask>')
        mask_inputs.append(" ".join(mask_input))
        mask_tokens.append("")
        attack_types.append('insert')
        pivot_indices.append(insert_idx)

    # check merge choices
    for merge_idx in merge_indices:
        if thres.get('keep_sim', False):
            if (text_ls[merge_idx].lower() in binary_words or \
                text_ls[merge_idx+1].lower() in binary_words):
                continue
            if text2 is not None and \
                (text_prime[merge_idx].lower() in text2_tokens or \
                 text_prime[merge_idx+1].lower() in text2_tokens):
                continue
        mask_input = text_prime.copy()
        mask_input[merge_idx] = '<mask>'
        del mask_input[merge_idx+1]
        mask_inputs.append(" ".join(mask_input))
        orig_token = " ".join([text_prime[merge_idx], text_prime[merge_idx+1]])
        mask_tokens.append(orig_token)
        attack_types.append('merge')
        pivot_indices.append(merge_idx)
    if len(mask_inputs) == 0:
        return [], 0
    synonyms, syn_probs = generator(mask_inputs, mask_tokens, mask_info=False)
    if len(synonyms) == 0:
        return [], 0
        
    # filter the candidate by syn_probs and synonyms and then query the target models
    synonyms_, syn_probs_, pivot_indices_, attack_types_, new_texts = [], [], [], [], []
    for i in range(len(synonyms)):
        attack_type = attack_types[i]
        idx = pivot_indices[i]
        if attack_type == 'replace':
            for j in range(len(synonyms[i])):
                if syn_probs[i][j] > thres['replace_prob']:
                    synonym = synonyms[i][j]
                    synonyms_.append(synonym)
                    syn_probs_.append(syn_probs[i][j])
                    pivot_indices_.append(idx)
                    attack_types_.append('replace')
                    new_texts.append(
                        text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):])
        if attack_type == 'insert':
            for j in range(len(synonyms[i])):
                if syn_probs[i][j] > thres['insert_prob']:
                    synonym = synonyms[i][j]
                    # don't insert punctuation
                    if punct_re.search(synonym) is not None and \
                        words_re.search(synonym) is None:
                        continue
                    synonyms_.append(synonym)
                    syn_probs_.append(syn_probs[i][j])
                    pivot_indices_.append(idx)
                    attack_types_.append('insert')
                    new_texts.append(
                        text_prime[:idx] + [synonym] + text_prime[min(idx, len_text):])
        if attack_type == 'merge':
            for j in range(len(synonyms[i])):
                if syn_probs[i][j] > thres['merge_prob']:
                    synonym = synonyms[i][j]
                    synonyms_.append(synonym)
                    syn_probs_.append(syn_probs[i][j])
                    pivot_indices_.append(idx)
                    attack_types_.append('merge')
                    new_texts.append(
                        text_prime[:idx] + [synonym] + text_prime[min(idx + 2, len_text):])
                    
    syn_probs = np.array(syn_probs_)
    synonyms, pivot_indices, attack_types = \
        synonyms_, pivot_indices_, attack_types_

    try:
        semantic_sims = similairty_calculation(
                pivot_indices, [text_prime] * len(new_texts), new_texts,
                sim_predictor, attack_types=attack_types, thres=thres)
    except:
        torch.cuda.empty_cache()
        semantic_sims = similairty_calculation(
                pivot_indices, [text_prime] * len(new_texts), new_texts,
                sim_predictor, attack_types=attack_types, thres=thres)

    synonyms_, syn_probs_, pivot_indices_, attack_types_, \
        new_texts_, semantic_sims_, orig_tokens = \
        [], [], [], [], [], [], []
    # filter by semantic_sims
    for i in range(len(semantic_sims)):
        if attack_types[i] == 'replace':
            orig_token = text_prime[pivot_indices[i]]
            if semantic_sims[i] < thres['replace_sim']:
                continue
        
        if attack_types[i] == 'insert':
            orig_token = text_prime[pivot_indices[i]-1]
            if semantic_sims[i] < thres['insert_sim']:
                continue

        if attack_types[i] == 'merge':
            orig_token = " ".join(text_prime[pivot_indices[i]:pivot_indices[i]+2])
            if semantic_sims[i] < thres['merge_sim']:
                continue
        synonyms_.append(synonyms[i])
        syn_probs_.append(syn_probs[i])
        pivot_indices_.append(pivot_indices[i])
        attack_types_.append(attack_types[i])
        new_texts_.append(new_texts[i])
        semantic_sims_.append(semantic_sims[i])
        orig_tokens.append(orig_token)  
    synonyms, syn_probs, pivot_indices, attack_types, new_texts, semantic_sims = \
        synonyms_, syn_probs_, pivot_indices_, attack_types_, new_texts_, semantic_sims_
    
    if len(new_texts) == 0:
        return [], 0
    # prediction by querying target model, and filter by probs_diffs
    new_probs = target_model(new_texts, text2)
    prob_diffs = (new_probs[:, orig_label] - orig_prob).cpu().numpy()
    collections = []
    num_query = len(new_texts)
    for i in range(len(prob_diffs)):
        if prob_diffs[i] < thres['prob_diff']:
            collections.append(
                [pivot_indices[i], attack_types[i], orig_tokens[i],
                 synonyms[i], semantic_sims[i], syn_probs[i], prob_diffs[i]])

    if len(collections) == 0:
        return [], 0
    # for each choice, find the best attack choices
    attack_sequences = []
    best_prob_diff = collections[0][-1]
    best_sequence = collections[0]
    for sequence in collections:
        # if new choice appear
        if best_sequence[:2] != sequence[:2]:
            attack_sequences.append(best_sequence)
            best_sequence = sequence
            best_prob_diff = sequence[-1]
            continue
        if sequence[-1] < best_prob_diff:
            best_prob_diff = sequence[-1]
            best_sequence = sequence
    attack_sequences.append(best_sequence)
    attack_sequences.sort(key=lambda x : x[-1])
    return attack_sequences, num_query


def find_merge_index(token_tags, indices=None):
    merge_indices = []
    if indices == None:
        indices = range(len(token_tags) - 1)
    for i in indices:
        cur_tag = token_tags[i][1]
        next_tag = token_tags[i+1][1]
        if cur_tag == 'NOUN' and next_tag =='NOUN':
            merge_indices.append(i)
        elif cur_tag == 'ADJ' and next_tag in ['NOUN', 'NUM', 'ADJ', 'ADV']:
            merge_indices.append(i)
        elif cur_tag == 'ADV' and next_tag in ['ADJ', 'VERB']:
            merge_indices.append(i)
        elif cur_tag == 'VERB' and next_tag in ['ADV', 'VERB', 'NOUN', 'ADJ']:
            merge_indices.append(i)
        elif cur_tag == 'DET' and next_tag in ['NOUN', 'ADJ']:
            merge_indices.append(i)
        elif cur_tag == 'PRON' and next_tag in ['NOUN', 'ADJ']:
            merge_indices.append(i)
        elif cur_tag == 'NUM' and next_tag in ['NUM', 'NOUN']:
            merge_indices.append(i)
    return merge_indices


def index_from_pos_tag(token_tags):
    replace_loc, insert_loc, merge_loc = [], [], []
    for idx in range(len(token_tags)):
        if token_tags[idx][1] in pos_tag_filter['replace']:
            replace_loc.append(idx)
        if idx > 0 and "%s/%s" % (token_tags[idx-1][1], token_tags[idx][1]) \
            in pos_tag_filter['insert']:
            insert_loc.append(idx)
        if idx < len(token_tags) - 1 and \
            "%s/%s" % (token_tags[idx][1], token_tags[idx+1][1]) in pos_tag_filter['merge']:
            merge_loc.append(idx)
    return replace_loc, insert_loc, merge_loc


def index_from_slience_score(token_tags, text_ls, orig_label, orig_probs, target_model):
    orig_prob = orig_probs.max()
    len_text = len(text_ls)
    leave_1_texts = [text_ls[:ii] + ['[UNK]'] + text_ls[min(ii + 1, len_text):] for ii in range(len_text)]
    leave_1_probs = target_model(leave_1_texts)
    leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
    import_scores = (orig_prob - leave_1_probs[:, orig_label] + (leave_1_probs_argmax != orig_label).float() * (
                leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0,
                                                                leave_1_probs_argmax))).data.cpu().numpy()

    replace_indices = []
    for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True):
        if score > -1.0:
            replace_indices.append(idx)
    replace_indices = replace_indices[:min(int(len(text_ls) * 0.2), len(replace_indices))]
    
    insert_indices = set()
    for i in replace_indices:
        if i > 0:
            insert_indices.add(i)
        insert_indices.add(i+1)
    insert_indices = list(insert_indices)
    
    merge_indices = set()
    for i in replace_indices:
        if i > 0:
            merge_indices.add(i-1)
        if i < len(text_ls) - 1:
            merge_indices.add(i)
    merge_indices = list(merge_indices)
    merge_indices = find_merge_index(token_tags, indices=merge_indices)
    return replace_indices, insert_indices, merge_indices