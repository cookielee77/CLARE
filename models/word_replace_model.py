import random

import torch
import numpy as np
import spacy
import nltk
from nltk.corpus import wordnet as wn

class RandomAttack(object):
    def __init__(self, args):
        self.import_score_threshold = -1.
        self.synonym_num = 50
        self.stop_words_set = set(nltk.corpus.stopwords.words('english'))
        self.perturb_ratio = 0.2
        self.build_vocab(args.counter_fitting_embeddings_path)
        self.filter_types = {'pos_filter': False,
                             'similarity_filter': False,
                             'lm_filter': False}

    # build dictionary via the embedding file
    def build_vocab(self, counter_fitting_embeddings_path):
        self.idx2word = {}
        self.word2idx = {}
        print("Building vocab...")
        with open(counter_fitting_embeddings_path, 'r') as ifile:
            for line in ifile:
                word = line.split()[0]
                if word not in self.idx2word:
                    self.idx2word[len(self.idx2word)] = word
                    self.word2idx[word] = len(self.idx2word) - 1
    
    # build word cos similarity matrix
    def build_word_sim_matrix(self, counter_fitting_cos_sim_path, counter_fitting_embeddings_path):
        print("Building cos sim matrix...")
        if counter_fitting_cos_sim_path:
            # load pre-computed cosine similarity matrix if provided
            print('Load pre-computed cosine similarity matrix from {}'.format(counter_fitting_cos_sim_path))
            cos_sim = np.load(counter_fitting_cos_sim_path)
        else:
            # calculate the cosine similarity matrix
            print('Start computing the cosine similarity matrix!')
            embeddings = []
            with open(counter_fitting_embeddings_path, 'r') as ifile:
                for line in ifile:
                    embedding = [float(num) for num in line.strip().split()[1:]]
                    embeddings.append(embedding)
            embeddings = np.array(embeddings)
            product = np.dot(embeddings, embeddings.T)
            norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
            cos_sim = product / np.dot(norm, norm.T)
        self.cos_sim = cos_sim
        print("Cos sim import finished!")
        
    # get attack_sequence according to the important scores
    def get_attack_sequences(self, predictor, text_ls, text2, marks, orig_probs, orig_label):
        len_text = len(text_ls)
        saliency_scores = np.zeros(len(text_ls))
        len_vocab = len(self.idx2word)
        
        # for each random index, random select 50 words to attack
        perturb_idxes = random.sample(range(len_text), int(len_text * self.perturb_ratio))
        attack_sequences = []
        for idx in perturb_idxes:
            random_word_indices = random.sample(range(len_vocab), self.synonym_num)
            synonyms = [self.idx2word[i] for i in random_word_indices]
            attack_sequences.append((idx, synonyms))
        return saliency_scores, attack_sequences, 0


class Textfooler(RandomAttack):
    def __init__(self, args):
        super().__init__(args)
        self.build_word_sim_matrix(args.counter_fitting_cos_sim_path,
                                   args.counter_fitting_embeddings_path)
        self.filter_types = {'pos_filter': True,
                             'similarity_filter': True,
                             'lm_filter': False}
    
    # get attack_sequence according to the important scores
    def get_attack_sequences(self, predictor, text_ls, text2, marks, orig_probs, orig_label):
        orig_prob = orig_probs.max()
        len_text = len(text_ls)

        # get importance score
        leave_1_texts = [text_ls[:ii] + ['[UNK]'] + text_ls[min(ii + 1, len_text):] for ii in range(len_text)]
        leave_1_probs = predictor(leave_1_texts, text2, marks=marks)
        leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
        saliency_scores = (orig_prob - leave_1_probs[:, orig_label] + (leave_1_probs_argmax != orig_label).float() * (
                    leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0,
                                                                    leave_1_probs_argmax))).data.cpu().numpy()
        
        ranked_words_perturb = []
        for idx, score in sorted(enumerate(saliency_scores), key=lambda x: x[1], reverse=True):
            if score > self.import_score_threshold and text_ls[idx] not in self.stop_words_set:
                ranked_words_perturb.append((idx, text_ls[idx]))
        # find synonyms
        synonym_words, _ = self.pick_most_similar_words_batch(ranked_words_perturb, 0.5)

        attack_sequences = []
        for idx, word in ranked_words_perturb:
            if word in self.word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms:
                    attack_sequences.append((idx, synonyms))
        return saliency_scores, attack_sequences, len(text_ls)

    # find synonyms from the word embedding
    def pick_most_similar_words_batch(self, ranked_words_perturb, threshold=0.5):
        """
        embeddings is a matrix with (d, vocab_size)
        """
        word_indices = [self.word2idx[word] for idx, word in ranked_words_perturb if word in self.word2idx]
        sim_order = np.argsort(-self.cos_sim[word_indices, :])[:, 1:1 + self.synonym_num]
        sim_words, sim_values = [], []
        for idx, src_word in enumerate(word_indices):
            sim_value = self.cos_sim[src_word][sim_order[idx]]
            mask = sim_value >= threshold
            sim_word, sim_value = sim_order[idx][mask], sim_value[mask]
            sim_word = [self.idx2word[id] for id in sim_word]
            sim_words.append(sim_word)
            sim_values.append(sim_value)
        return sim_words, sim_values
    
    
class PWWS(object):
    def __init__(self, args):
        self.supported_pos_tags = set(['CC', 'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP',
                                       'NNPS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG',
                                       'VBN', 'VBP', 'VBZ'])
        self.nlp = spacy.load('en_core_web_sm')
        self.filter_types = {'pos_filter': False,
                             'similarity_filter': False,
                             'lm_filter': False}
    
    def _get_wordnet_pos(self, token):
        '''Wordnet POS tag'''
        pos = token.tag_[0].lower()
        if pos in ['r', 'n', 'v']:  # adv, noun, verb
            return pos
        elif pos == 'j':
            return 'a'  # adj
    
    def _synonym_prefilter_fn(self, token, synonyms):
        '''
        Similarity heuristics go here
        '''
        synonyms_ = set()
        for synonym in synonyms:
            if (len(synonym.text.split()) > 1 or (  # the synonym produced is a phrase
                    synonym[0].lemma == token.lemma) or (  # token and synonym are the same
                    synonym[0].tag != token.tag) or (  # original code, keep the pos tag different?
                    token.text.lower() == 'be')):
                continue
            synonyms_.add(synonym.text)
        return list(synonyms_)
    
    def get_attack_sequences(self, predictor, text_ls, text2, marks, orig_probs, orig_label):
        orig_prob = orig_probs.max()
        len_text = len(text_ls)
        num_query = 0
        leave_1_texts = [text_ls[:ii] + ['[UNK]'] + text_ls[min(ii + 1, len_text):] for ii in range(len_text)]
        leave_1_probs = predictor(leave_1_texts, text2, marks=marks)[:, orig_label]
        num_query += len(leave_1_texts)
        
        saliency_scores = torch.nn.functional.softmax(leave_1_probs - orig_prob, dim=0).cpu().numpy()
        attack_sequences = [[idx, [], saliency_scores[idx]] for idx in range(len(text_ls))]
        
        doc = self.nlp(" ".join(text_ls))
        token_to_spacy_token = {}
        new_texts = []
        start_end_indices = []
        indices = []
        all_synonyms = []
        start_idx = 0
        for x in doc:
            token_to_spacy_token[x.text] = x
        for idx in range(len(text_ls)):
            if text_ls[idx] not in token_to_spacy_token:
                continue
            token = token_to_spacy_token[text_ls[idx]]
            if token.tag_ not in self.supported_pos_tags:
                continue
            wordnet_pos = self._get_wordnet_pos(token)  # 'r', 'a', 'n', 'v' or None
            wordnet_synonyms = []
            synsets = wn.synsets(token.text, pos=wordnet_pos)
            for synset in synsets:
                wordnet_synonyms.extend(synset.lemmas())
                
            synonyms = []
            for wordnet_synonym in wordnet_synonyms:
                spacy_synonym = self.nlp(wordnet_synonym.name().replace('_', ' '))
                synonyms.append(spacy_synonym)
            synonyms = self._synonym_prefilter_fn(token, synonyms)
            if len(synonyms) == 0:
                continue
            
            # calculate substitution strategy
            new_texts_ = [text_ls[:idx] + [synonym] + text_ls[min(idx + 1, len_text):] for synonym in synonyms]
            end_idx = start_idx + len(new_texts_)
            new_texts.extend(new_texts_)
            start_end_indices.append((start_idx, end_idx))
            start_idx = end_idx
            all_synonyms.extend(synonyms)
            indices.extend([idx] * len(synonyms))
        
        if len(new_texts) == 0:
            return saliency_scores, [], 0  
        # make batch prediction
        new_probs = predictor(new_texts, text2, marks=marks)
        num_query += len(new_texts)
        prob_diffs = orig_prob - new_probs[:, orig_label]
        for start_idx, end_idx in start_end_indices:
            max_value, max_idx = torch.max(prob_diffs[start_idx:end_idx], dim=-1)
            max_value = max_value.cpu().item()
            max_idx = max_idx.cpu().item()
            # convert to global indices
            idx = indices[start_idx]
            max_idx = start_idx + max_idx
            attack_sequences[idx][1].append(all_synonyms[max_idx])
            attack_sequences[idx][2] *= max_value
        attack_sequences = [x for x in attack_sequences if len(x[1]) != 0]
        attack_sequences.sort(key=lambda x : x[-1], reverse=True)
        
        return saliency_scores, attack_sequences, num_query      