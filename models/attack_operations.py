import numpy as np
import nltk


def similairty_calculation(indices, orig_texts, new_texts,
                           sim_predictor, attack_types=None, thres=None):
    # compute semantic similarity
    half_sim_window = (thres['sim_window'] - 1) // 2
    orig_locals = []
    new_locals = []
    for i in range(len(indices)):
        idx = indices[i]
        len_text = len(orig_texts[i])
        if idx >= half_sim_window and len_text - idx - 1 >= half_sim_window:
            text_range_min = idx - half_sim_window
            text_range_max = idx + half_sim_window + 1
        elif idx < half_sim_window and len_text - idx - 1 >= half_sim_window:
            text_range_min = 0
            text_range_max = thres['sim_window']
        elif idx >= half_sim_window and len_text - idx - 1 < half_sim_window:
            text_range_min = len_text - thres['sim_window']
            text_range_max = len_text
        else:
            text_range_min = 0
            text_range_max = len_text
        orig_locals.append(" ".join(orig_texts[i][text_range_min:text_range_max]))
        if attack_types[i] == 'merge':
            text_range_max -= 1
        if attack_types[i] == 'insert':
            text_range_min -= 1
        new_locals.append(" ".join(new_texts[i][text_range_min:text_range_max]))

    return sim_predictor.semantic_sim(orig_locals, new_locals)[0]

def word_replacement(replace_idx, text_prime, generator, target_model,
                     orig_prob, orig_label, sim_predictor, text2=None, thres=None):
    len_text = len(text_prime)
    orig_token = text_prime[replace_idx]
    # according to the context to find synonyms
    mask_input = text_prime.copy()
    mask_input[replace_idx] = '<mask>'
    mask_token = [orig_token]
    synonyms, syn_probs = generator([" ".join(mask_input)], mask_token, mask_info=False)
    synonyms, syn_probs = synonyms[0], syn_probs[0]

    new_texts = [text_prime[:replace_idx] + [synonym] + text_prime[min(replace_idx + 1, len_text):] for synonym in synonyms]
    new_probs = target_model(new_texts, text2)
    prob_diffs = (new_probs[:, orig_label] - orig_prob).cpu().numpy()
    
    # compute semantic similarity
    semantic_sims = similairty_calculation(
        [replace_idx] * len(new_texts), [text_prime] * len(new_texts), new_texts,
        sim_predictor, attack_types=['replace'] * len(new_texts), thres=thres)
    
    # create filter mask
    attack_mask = prob_diffs < thres['prob_diff']
    prob_mask = syn_probs > thres['replace_prob']
    semantic_mask = semantic_sims > thres['replace_sim']
    
    prob_diffs *= (attack_mask * semantic_mask * prob_mask)
    best_idx = np.argmin(prob_diffs)
    
    # for debug purpose
    collections = []
    for i in range(len(synonyms)):
        if attack_mask[i] and semantic_mask[i] and prob_mask[i]:
            collections.append([prob_diffs[i], syn_probs[i], semantic_sims[i], orig_token, synonyms[i]])
    collections.sort(key=lambda x : x[0])
    
    return synonyms[best_idx], syn_probs[best_idx], prob_diffs[best_idx], semantic_sims[best_idx], \
            new_probs[best_idx].cpu().numpy(), collections


def word_insertion(insert_idx, text_prime, generator, target_model,
                   orig_prob, orig_label, punct_re, words_re, sim_predictor,
                   text2=None, thres=None):
    len_text = len(text_prime)
    mask_input = text_prime.copy()
    mask_input.insert(insert_idx, '<mask>')
    synonyms, syn_probs = generator([" ".join(mask_input)])
    synonyms, syn_probs = synonyms[0], syn_probs[0]

    new_texts = [text_prime[:insert_idx] + [synonym] + text_prime[min(insert_idx, len_text):] for synonym in synonyms]
    new_probs = target_model(new_texts, text2)
    prob_diffs = (new_probs[:, orig_label] - orig_prob).cpu().numpy()
    
    semantic_sims = similairty_calculation(
        [insert_idx] * len(new_texts), [text_prime] * len(new_texts), new_texts,
        sim_predictor, attack_types=['insert'] * len(new_texts), thres=thres)
    
    # create filter mask
    attack_mask = prob_diffs < thres['prob_diff']
    prob_mask = syn_probs > thres['insert_prob']
    semantic_mask = semantic_sims > thres['insert_sim']
    punc_mask = np.ones(attack_mask.shape)
    for i in range(len(punc_mask)):
        # don't insert punctuation
        if punct_re.search(synonyms[i]) is not None and words_re.search(synonyms[i]) is None:
            punc_mask[i] = 0
    
    prob_diffs *= (attack_mask * punc_mask * prob_mask * semantic_mask)
    best_idx = np.argmin(prob_diffs)
    
    # for debug purpose
    collections = []
    for i in range(len(synonyms)):
        if attack_mask[i] and punc_mask[i] and prob_mask[i]:
            collections.append([prob_diffs[i], syn_probs[i], semantic_sims[i], text_prime[insert_idx-1], synonyms[i]])
    collections.sort(key=lambda x : x[0])
    
    return synonyms[best_idx], syn_probs[best_idx], prob_diffs[best_idx], \
            semantic_sims[best_idx], new_probs[best_idx].cpu().numpy(), collections
            

def word_merge(merge_idx, text_prime, generator, target_model,
               orig_prob, orig_label, sim_predictor, text2=None, thres=None):
    len_text = len(text_prime)
    orig_token = " ".join([text_prime[merge_idx], text_prime[merge_idx+1]])
    # according to the context to find synonyms
    mask_input = text_prime.copy()
    mask_input[merge_idx] = '<mask>'
    del mask_input[merge_idx+1]
    mask_token = [orig_token]

    synonyms, syn_probs = generator([" ".join(mask_input)], mask_token, mask_info=False)
    synonyms, syn_probs = synonyms[0], syn_probs[0]

    new_texts = [text_prime[:merge_idx] + [synonym] + text_prime[min(merge_idx + 2, len_text):] for synonym in synonyms]
    new_probs = target_model(new_texts, text2)
    prob_diffs = (new_probs[:, orig_label] - orig_prob).cpu().numpy()
    
    semantic_sims = similairty_calculation(
        [merge_idx] * len(new_texts), [text_prime] * len(new_texts), new_texts,
        sim_predictor, attack_types=['merge'] * len(new_texts), thres=thres)

    # create filter mask
    attack_mask = prob_diffs < thres['prob_diff']
    prob_mask = syn_probs > thres['merge_prob']
    semantic_mask = semantic_sims > thres['merge_sim']
    
    prob_diffs *= (attack_mask * semantic_mask * prob_mask)
    best_idx = np.argmin(prob_diffs)
    
    # for debug purpose
    collections = []
    for i in range(len(synonyms)):
        if attack_mask[i] and semantic_mask[i] and prob_mask[i]:
            collections.append([prob_diffs[i], syn_probs[i], semantic_sims[i], orig_token, synonyms[i]])
    collections.sort(key=lambda x : x[0])
    
    return synonyms[best_idx], syn_probs[best_idx], prob_diffs[best_idx], semantic_sims[best_idx], \
            new_probs[best_idx].cpu().numpy(), collections