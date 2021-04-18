import os
import time
import random

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

from config import load_arguments
from utils.utils import Candidate_Mask
from utils.hyper_parameters import class_names, nclasses
from models.BERT_classifier import BERTinfer
from models.word_replace_model import *
from dataloaders.dataloader import read_corpus
from evaluate import evaluate
from transformers import RobertaTokenizer


def attack(example, predictor, synonym_replacer, candidate_mask, attack_second=False):
    # lower the text_ls
    true_label = example[0]
    if attack_second:
        text_ls = example[2].split()
        text2 = example[1]
    else:
        text_ls = example[1].split()
        text2 = example[2]
    # mark the capitalized upper information for attack part
    marks = []
    for i in range(len(text_ls)):
        if text_ls[i].capitalize() == text_ls[i]:
            marks.append('cap')
        elif text_ls[i].isupper():
            marks.append('upper')
        else:
            marks.append("")
    # first check the prediction of the original text
    orig_probs = predictor([text_ls], text2, marks=marks).squeeze()
    num_queries = 1
    orig_label = torch.argmax(orig_probs).item()
    orig_prob = orig_probs.max()
    prev_prob = orig_prob.item()
    if true_label != orig_label:
        return '', 0, orig_label, orig_label, 0, []
    else:
        # all baseline can only deal with lowercase text token
        text_ls = [x.lower() for x in text_ls]
        candidate_mask.init_sent(text_ls, text2)
        
        # get attack_sequence
        saliency_scores, attack_sequences, num_query = synonym_replacer.get_attack_sequences(
            predictor, text_ls, text2, marks, orig_probs, orig_label
        )
        # to keep the semantic meaning more similar with text2
        attack_sequences = candidate_mask.filter_by_text2_token(attack_sequences, text_ls)
        num_queries += num_query

        # start replacing and attacking
        attack_logs = []
        new_label = orig_label
        len_text = len(text_ls)
        text_prime = text_ls[:]
        num_changed = 0
        for attack_info in attack_sequences:
            idx = attack_info[0]
            synonyms = attack_info[1]
            new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
            new_probs = predictor(new_texts, text2, marks=marks)
            num_queries += len(new_texts)
            if len(new_probs.shape) < 2:
                new_probs = new_probs.unsqueeze(0)

            # prevent bad synonyms during similarity check
            semantic_sims, semantic_mask = candidate_mask.get_semantic_mask(idx, text_prime, new_texts)
            # get pos_tag filter
            pos_mask = candidate_mask.get_pos_mask(idx, new_texts)
            # get the lm mask, filter unfluent sentences
            lm_mask = candidate_mask.get_lm_mask(new_texts)
            # mask the prob by pos_mask and semantic_mask
            new_label_probs = new_probs[:, orig_label] + torch.from_numpy(
                    ~semantic_mask + ~pos_mask + ~lm_mask).float().cuda()

            new_label_prob_min, syn_index = torch.min(new_label_probs, dim=-1)
            if new_label_prob_min < orig_prob:
                orig_token, synonym = text_prime[idx], synonyms[syn_index]
                text_prime[idx] = synonyms[syn_index]
                cur_prob = new_probs[:, orig_label][syn_index].item()
                attack_logs.append([idx, orig_token, synonym,
                                    semantic_sims[syn_index], cur_prob-prev_prob, cur_prob])
                prev_prob = cur_prob
                num_changed += 1

                new_label = new_probs[syn_index, :].argmax().item()
                if new_label != orig_label:
                    break
        return text_prime, num_changed, orig_label, new_label, num_queries, attack_logs


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

    # prepare synonym extractor
    if args.baseline_type == 'textfooler':
        synonym_replacer = Textfooler(args)
    elif args.baseline_type == 'pwws':
        synonym_replacer = PWWS(args)
    elif args.baseline_type == 'random':
        synonym_replacer = RandomAttack(args)
    else:
        raise ValueError("%s baseline type is not supported." % args.baseline_type)
    candidate_mask = Candidate_Mask(args, synonym_replacer.filter_types)
    
    # check the input length
    tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')

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
        args.output_dir, str(args.data_size) + '_results_log'), 'a')
    if args.write_into_tsv:
        folder_path = os.path.join('./data', args.sample_file, args.dataset)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        tsv_name = os.path.join(folder_path, "%d.tsv" % args.data_idx)
        adversarial_file = open(tsv_name, 'w', encoding='utf8')
        header = 'label\ttext1\ttext2\n'
        adversarial_file.write(header)
    else:
        sample_file = open(
            os.path.join(args.output_dir, args.sample_file), 'w', encoding='utf8')

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
        new_label, num_queries, attack_logs = attack(
            example, predictor, synonym_replacer, candidate_mask, 
            attack_second=args.attack_second)

        if true_label != orig_label:
            orig_failures += 1
        else:
            nums_queries.append(num_queries)

        text = attack_text.split()
        changed_rate = 1.0 * num_changed / len(text)

        if true_label == orig_label and true_label != new_label:
            adv_failures += 1
            # transfomer the new_text into upper case
            assert len(text) == len(new_text)
            for i in range(len(text)):
                if text[i].capitalize() == text[i]:
                    new_text[i] = new_text[i].capitalize()
                if text[i].isupper():
                    new_text[i] = new_text[i].upper()
            new_text = " ".join(new_text)
            changed_rates.append(changed_rate)
            attack_texts.append(attack_text)
            new_texts.append(new_text)
            if args.write_into_tsv:
                text1 = new_text.strip()
                text2 = "" if single_sentence else ref_text.strip()
                if args.attack_second:
                    tmp = text1
                    text1, text2 = text2, tmp
                string_ = "%d\t%s\t%s\n" % (orig_label, text1, text2)
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
                    output_str = "%d replace %s %s %.2f %.4f %.4f\n" % tuple(attack_info)
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
    from models.similarity_model import USE
    use = USE(args.USE_cache_path)
    orig_ppl, adv_ppl, bert_score, sim_score, gram_err = evaluate(attack_texts, new_texts, use)
    message = 'Original ppl: {:.3f}, Adversarial ppl: {:.3f}, BertScore: {:.3f}, SimScore: {:.3f}, gram_err: {:.3f}\n\n'. \
        format(orig_ppl, adv_ppl, bert_score, sim_score, gram_err)
    log_file.write(message)
    print("Skipped indices: ", skipped_idx)
    print("Processing time: %d" % (time.time() - begin_time))

if __name__ == "__main__":
    main()