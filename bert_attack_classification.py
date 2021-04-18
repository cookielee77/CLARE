import os
import random
import time

import nltk
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import RobertaTokenizer

from config import load_arguments
from utils.hyper_parameters import class_names, nclasses, thres
from dataloaders.dataloader import read_corpus
from models.similarity_model import USE
from models.BERT_classifier import BERTinfer
from models.attack_location_search import get_attack_sequences
from models.attack_operations import *
from models.pipeline import FillMaskPipeline
from models.Roberta import RobertaForMaskedLM
from evaluate import evaluate

# for token check
import re
punct_re = re.compile(r'\W')
words_re = re.compile(r'\w')

def attack(example, predictor, stop_words_set, fill_mask, sim_predictor=None,
           synonym_num=50, attack_second=False, attack_loc=None,
           thres_=None):
    true_label = example[0]
    if attack_second:
        text_ls = example[2].split()
        text2 = example[1]
    else:
        text_ls = example[1].split()
        text2 = example[2]
    # first check the prediction of the original text
    orig_probs = predictor([text_ls], text2).squeeze()
    orig_label = torch.argmax(orig_probs).item()
    orig_prob = orig_probs.max()
    if true_label != orig_label:
        return '', 0, orig_label, orig_label, 0, []
    num_queries = 1
    
    # find attack sequences according to predicted probablity change
    attack_sequences, num_query = get_attack_sequences(
        text_ls, fill_mask, predictor, sim_predictor,
        orig_probs, orig_label, stop_words_set, punct_re, words_re,
        text2=text2, attack_loc=attack_loc, thres=thres_)
    num_queries += num_query

    # perform attack sequences
    attack_logs = []
    text_prime = text_ls.copy()
    prev_prob = orig_prob
    insertions = []
    merges = []
    forbid_replaces = set()
    forbid_inserts = set()
    forbid_merges = set(range(5))
    num_changed = 0
    new_label = orig_label
    for attack_info in attack_sequences:
        num_queries += synonym_num
        idx = attack_info[0]
        attack_type = attack_info[1]
        orig_token = attack_info[2]
        # check forbid replace operations
        if attack_type == 'insert' and idx in forbid_inserts:
            continue
        if attack_type == 'merge' and idx in forbid_merges:
            continue
        if attack_type == 'replace' and idx in forbid_replaces:
            continue
        
        # shift the attack index by insertions history
        shift_idx = idx
        for prev_insert_idx in insertions:
            if idx >= prev_insert_idx:
                shift_idx +=1
        for prev_merge_idx in merges:
            if idx >= prev_merge_idx + 1:
                shift_idx -= 1
        
        if attack_type == 'replace':
            synonym, syn_prob, prob_diff, semantic_sim, new_prob, collections = \
                word_replacement(
                    shift_idx, text_prime, fill_mask, predictor,
                    prev_prob, orig_label, sim_predictor, text2, thres=thres_)
        elif attack_type == 'insert':
            synonym, syn_prob, prob_diff, semantic_sim, new_prob, collections = \
                word_insertion(
                    shift_idx, text_prime, fill_mask, predictor,
                    prev_prob, orig_label, punct_re, words_re, sim_predictor, text2, thres=thres_)
        elif attack_type == 'merge':
            synonym, syn_prob, prob_diff, semantic_sim, new_prob, collections = \
                word_merge(
                    shift_idx, text_prime, fill_mask, predictor,
                    prev_prob, orig_label, sim_predictor, text2, thres=thres_)
        
        if prob_diff < 0:
    #         import ipdb; ipdb.set_trace()
            if attack_type == 'replace':
                text_prime[shift_idx] = synonym
                # forbid_inserts.add(idx)
                # forbid_inserts.add(idx+1)
                forbid_merges.add(idx-1)
                forbid_merges.add(idx)
            elif attack_type == 'insert':
                text_prime.insert(shift_idx, synonym)
                # append original attack index
                insertions.append(idx)
                forbid_merges.add(idx-1)
                # forbid_merges.add(idx)
                for i in [-1, 1]:
                    forbid_inserts.add(idx + i)
            elif attack_type == 'merge':
                text_prime[shift_idx] = synonym
                del text_prime[shift_idx+1]
                merges.append(idx)
                # forbid_inserts.add(idx)
                forbid_inserts.add(idx+1)
                # forbid_inserts.add(idx+2)
                # forbid_replaces.add(idx-1)
                forbid_replaces.add(idx)
                forbid_replaces.add(idx+1)
                for i in [-1, 1]:
                    forbid_merges.add(idx + i)
            cur_prob = new_prob[orig_label].item()
            attack_logs.append([idx, attack_type, orig_token, synonym, syn_prob,
                                semantic_sim, prob_diff, cur_prob])
            prev_prob = cur_prob
            num_changed += 1
            # if attack successfully!
            if np.argmax(new_prob) != orig_label:
                new_label = np.argmax(new_prob)
                break

    return ' '.join(text_prime), num_changed, orig_label, new_label, num_queries, attack_logs


def main():
    begin_time = time.time()
    args = load_arguments()
    # get data to attack
    examples = read_corpus(args.attack_file)
    if args.data_size is None:
        args.data_size = len(examples)
    examples = examples[args.data_idx:args.data_idx+args.data_size] # choose how many samples for adversary
    print("Data import finished!")

    # construct the model
    print("Building Model...")
    model = BERTinfer(args.target_model, args.target_model_path,
                      nclasses[args.dataset], args.case,
                      batch_size=args.batch_size,
                      attack_second=args.attack_second)
    predictor = model.text_pred
    print("Model built!")

    # prepare context predictor
    tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')
    model = RobertaForMaskedLM.from_pretrained('distilroberta-base')
    fill_mask = FillMaskPipeline(model, tokenizer, topk=args.synonym_num)

    # build the semantic similarity module
    use = USE(args.USE_cache_path)

    # start attacking
    num_sample = 0
    orig_failures = 0.
    adv_failures = 0.
    skipped_idx = []
    changed_rates = []
    nums_queries = []
    attack_texts = []
    new_texts = []
    label_names = class_names[args.dataset]
    log_file = open(os.path.join(
        args.output_dir,str(args.data_size) + '_results_log'), 'a')
    if args.write_into_tsv:
        folder_path = os.path.join('./data', args.sample_file, args.dataset)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        tsv_name = os.path.join(folder_path, "%d.tsv" % args.data_idx)
        adversarial_file = open(tsv_name, 'w', encoding='utf8')
        header = 'label\ttext1\ttext2\tnum_change\n'
        adversarial_file.write(header)
    else:
        sample_file = open(
            os.path.join(args.output_dir, args.sample_file), 'w', encoding='utf8')

    stop_words_set = set(nltk.corpus.stopwords.words('english'))
    print('Start attacking!')
    for idx, example in enumerate(tqdm(examples)):
        true_label = example[0]
        if example[2] is not None:
            single_sentence = False
            attack_text = example[2] if args.attack_second else example[1]
            ref_text = example[1] if args.attack_second else example[2]
        else:
            single_sentence = True
            attack_text = example[1]
        if len(tokenizer.encode(attack_text)) > args.max_seq_length:
            skipped_idx.append(idx)
            continue
        num_sample += 1

        new_text, num_changed, orig_label, \
        new_label, num_queries, attack_logs = \
            attack(example, predictor, stop_words_set,
                   fill_mask, sim_predictor=use,
                   synonym_num=args.synonym_num,
                   attack_second=args.attack_second,
                   attack_loc=args.attack_loc,
                   thres_=thres[args.dataset])

        if true_label != orig_label:
            orig_failures += 1
        else:
            nums_queries.append(num_queries)

        changed_rate = 1.0 * num_changed / len(attack_text.split())

        if true_label == orig_label and true_label != new_label:
            adv_failures += 1
            attack_texts.append(attack_text)
            new_texts.append(new_text)
            changed_rates.append(changed_rate)
            if args.write_into_tsv:
                text1 = new_text.strip()
                text2 = "" if single_sentence else ref_text.strip()
                if args.attack_second:
                    tmp = text1
                    text1, text2 = text2, tmp
                string_ = "%d\t%s\t%s\t%d\n" % (orig_label, text1, text2, num_changed)
                adversarial_file.write(string_)
            else:
                sample_file.write("Sentence index: %d\n" % idx)
                if not single_sentence:
                    sample_file.write('ref sent: %s\n' % ref_text)
                sample_file.write('orig sent ({}):\t{}\nadv sent ({}):\t{}\n'.format(
                    true_label, attack_text, new_label, new_text))
                sample_file.write('label change: %s ---> %s. num of change: %d\n\n' % \
                    (label_names[orig_label], label_names[new_label], len(attack_logs)))
                for attack_info in attack_logs:
                    output_str = "%d %s %s %s %.4f %.2f %.4f %.4f\n" % tuple(attack_info)
                    sample_file.write(output_str)
                sample_file.write('\n---------------------------------------------\n')

    orig_acc = (1 - orig_failures / num_sample) * 100
    attack_rate = 100 * adv_failures / (num_sample - orig_failures)
    message = 'For Generated model {} / Target model {} : original accuracy: {:.3f}%, attack success: {:.3f}%, ' \
              'avg changed rate: {:.3f}%, num of queries: {:.1f}, num of samples: {:d}, time: {:.1f}\n'.format(
                  args.sample_file, args.target_model, orig_acc, attack_rate,
                  np.mean(changed_rates)*100, np.mean(nums_queries), num_sample, time.time() - begin_time)
    print(message)
    log_file.write(message)
    torch.cuda.empty_cache()
    orig_ppl, adv_ppl, bert_score, sim_score, gram_err = evaluate(attack_texts, new_texts, use)
    message = 'Original ppl: {:.3f}, Adversarial ppl: {:.3f}, BertScore: {:.3f}, SimScore: {:.3f}, gram_err: {:.3f}\n\n'. \
        format(orig_ppl, adv_ppl, bert_score, sim_score, gram_err)
    log_file.write(message)
    print("Skipped indices: ", skipped_idx)
    print("Processing time: %d" % (time.time() - begin_time))

if __name__ == "__main__":
    main()