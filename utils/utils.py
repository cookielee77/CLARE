import nltk
import torch
import numpy as np

from transformers import GPT2Tokenizer, GPT2LMHeadModel

from models.similarity_model import USE
from .hyper_parameters import thres

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def get_pos(sent, tagset='universal'):
    '''
    :param sent: list of word strings
    tagset: {'universal', 'default'}
    :return: list of pos tags.
    Universal (Coarse) Pos tags has  12 categories
        - NOUN (nouns)
        - VERB (verbs)
        - ADJ (adjectives)
        - ADV (adverbs)
        - PRON (pronouns)
        - DET (determiners and articles)
        - ADP (prepositions and postpositions)
        - NUM (numerals)
        - CONJ (conjunctions)
        - PRT (particles)
        - . (punctuation marks)
        - X (a catch-all for other categories such as abbreviations or foreign words)
    '''
    if tagset == 'default':
        word_n_pos_list = nltk.pos_tag(sent)
    elif tagset == 'universal':
        word_n_pos_list = nltk.pos_tag(sent, tagset=tagset)
    _, pos_list = zip(*word_n_pos_list)
    return pos_list


def pos_filter(ori_pos, new_pos_list):
    same = [True if ori_pos == new_pos or (set([ori_pos, new_pos]) <= set(['NOUN', 'VERB']))
            else False
            for new_pos in new_pos_list]
    return same


class Candidate_Mask(object):
    def __init__(self, args, filter_types):
        self.dataset = args.dataset
        self.filter_types = filter_types
        self.sim_score_threshold = 0.7
        self.sim_score_window = 15
        # build the semantic similarity module
        if self.filter_types['similarity_filter']:
            self.use = USE(args.USE_cache_path)
        if self.filter_types['lm_filter']:
            self.lm_topp = 0.2
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.lm_model = GPT2LMHeadModel.from_pretrained('gpt2').cuda()
        
    def init_sent(self, text_ls, text2):
        # similarity window for later use
        if len(text_ls) < self.sim_score_window:
            self.sim_score_threshold = 0.1  # shut down the similarity thresholding function
        else:
            self.sim_score_threshold = 0.7
        self.half_sim_score_window = (self.sim_score_window - 1) // 2
        # get the pos and verb tense info
        self.pos_ls = get_pos(text_ls)
        # to keep the semantic meaning more similar with text2
        if thres[self.dataset].get('keep_sim', False) and text2 is not None:
            text2_tags = nltk.pos_tag(text2.lower().split(), tagset='universal')
            self.text2_tokens = set()
            for text2_tag in text2_tags:
                if text2_tag[1] in ['NOUN', 'VERB', 'ADV', 'NUM', 'ADJ']:
                    self.text2_tokens.add(text2_tag[0])
        else:
            self.text2_tokens = None

    # to keep the semantic meaning more similar with text2
    def filter_by_text2_token(self, attack_sequences_, text_ls):
        if self.text2_tokens is not None:
            attack_sequences = []
            for attack_op in attack_sequences_:
                idx = attack_op[0]
                if text_ls[idx] in self.text2_tokens:
                        continue
                attack_sequences.append(attack_op)
        else:
            attack_sequences = attack_sequences_
        return attack_sequences

    # calculate the similarity by USE
    def get_semantic_mask(self, idx, text_cache, new_texts):
        if self.filter_types['similarity_filter']:
            len_text = len(text_cache)
            # compute semantic similarity
            if idx >= self.half_sim_score_window and len_text - idx - 1 >= self.half_sim_score_window:
                text_range_min = idx - self.half_sim_score_window
                text_range_max = idx + self.half_sim_score_window + 1
            elif idx < self.half_sim_score_window and len_text - idx - 1 >= self.half_sim_score_window:
                text_range_min = 0
                text_range_max = self.sim_score_window
            elif idx >= self.half_sim_score_window and len_text - idx - 1 < self.half_sim_score_window:
                text_range_min = len_text - self.sim_score_window
                text_range_max = len_text
            else:
                text_range_min = 0
                text_range_max = len_text
            semantic_sims = \
            self.use.semantic_sim([' '.join(text_cache[text_range_min:text_range_max])] * len(new_texts),
                                        list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[0]
        else:
            semantic_sims = np.ones(len(new_texts), dtype=bool)
        return semantic_sims, semantic_sims > self.sim_score_threshold
    
    # prevent incompatible pos
    def get_pos_mask(self, idx, new_texts):
        if self.filter_types['pos_filter']:
            synonyms_pos_ls = [get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
                                if len(new_text) > 10 else get_pos(new_text)[idx] for new_text in new_texts]
            pos_mask = np.array(pos_filter(self.pos_ls[idx], synonyms_pos_ls))
        else:
            pos_mask = np.ones(len(new_texts), dtype=bool)
        return pos_mask
    
    # filter infulent sentences
    def get_lm_mask(self, new_texts):
        if self.filter_types['lm_filter']:
            lm_scores = []
            for text in new_texts:
                text = " ".join(text)
                input_ids = torch.tensor(self.tokenizer.encode(text, add_special_tokens=True))
                input_ids = input_ids.cuda()
                outputs = self.lm_model(input_ids, labels=input_ids)
                lm_loss = outputs[0].mean().item()
                lm_scores.append(lm_loss)
            lm_thres = np.sort(lm_scores)[int(len(new_texts) * self.lm_topp)]
            lm_mask = lm_scores < lm_thres
        else:
            lm_mask = np.ones(len(new_texts), dtype=bool)
        return lm_mask