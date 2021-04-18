import os
import re
import csv
import random
random.seed(2020)

import numpy as np
import torch

def process_string(string):
    string = re.sub("( )(\'[(m)(d)(t)(ll)(re)(ve)(s)])", r"\2", string)
    string = re.sub("(\d+)( )([,\.])( )(\d+)", r"\1\3\5", string)
    # U . S . -> U.S.
    string = re.sub("(\w)( )(\.)( )(\w)( )(\.)", r"\1\3\5\7", string)
    # reduce left space
    string = re.sub("( )([,\.!?:;)])", r"\2", string)
    # reduce right space
    string = re.sub("([(])( )", r"\1", string)
    string = re.sub("s '", "s'", string)
    # reduce both space
    string = re.sub("(')( )(\S+)( )(')", r"\1\3\5", string)
    string = re.sub("(\")( )(\S+)( )(\")", r"\1\3\5", string)
    string = re.sub("(\w+) (-+) (\w+)", r"\1\2\3", string)
    string = re.sub("(\w+) (/+) (\w+)", r"\1\2\3", string)
    # string = re.sub(" ' ", "'", string)
    return string


def read_corpus(path, text_label_pair=False):
    with open(path, encoding='utf8') as f:
        examples = list(csv.reader(f, delimiter='\t', quotechar=None))[1:]
        second_text = False if examples[1][2] == '' else True
        for i in range(len(examples)):
            examples[i][0] = int(examples[i][0])
            if not second_text:
                examples[i][2] = None

    # label, text1, text2
    if text_label_pair:
        tmp = list(zip(*examples))
        return tmp[0], list(zip(tmp[1], tmp[2]))
    else:
        return examples


def read_adv_corpus(path, text_label_pair=False,
                    dataset=None, max_adv_len=None, min_adv_len=None,
                    adv_data_ratio=1.0, max_num_change=None):
    with open(path, encoding='utf8') as f:
        examples = list(csv.reader(f, delimiter='\t', quotechar=None))[1:]
        second_text = False if examples[1][2] == '' else True
        for i in range(len(examples)):
            examples[i][0] = int(examples[i][0])
            if not second_text:
                examples[i][2] = None

    # filter conditions
    examples_ = []
    for example in examples:
        # filter adversarial examples by num of changes
        if max_num_change is not None and int(example[3]) > max_num_change:
            continue
        # filter adversarial examples by lengths
        text = example[2] if dataset == 'qnli' else example[1]
        text_len = len(text.split())
        if max_adv_len is not None and min_adv_len is not None and (text_len < min_adv_len or text_len > max_adv_len):
            continue
        examples_.append(example)
    examples = examples_

    # filter adv data by ratio
    if adv_data_ratio != 1.0:
        random.shuffle(examples)
        examples = examples[:int(adv_data_ratio * len(examples))]
        assert len(examples) != 0, "%f is too small, the adv data length is zero." % adv_data_ratio

    # label, text1, text2
    if text_label_pair:
        tmp = list(zip(*examples))
        return tmp[0], list(zip(tmp[1], tmp[2]))
    else:
        return examples