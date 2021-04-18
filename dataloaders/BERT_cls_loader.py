import os

import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler, TensorDataset

from transformers import BertTokenizer
from .dataloader import read_corpus, read_adv_corpus


class BERTClsDataloader(Dataset):

    def __init__(self, case):
        if case == 'cased':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        elif case == 'uncased':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        else:
            raise ValueError(case)

    def transform_text(self, data, max_seq_length=256, labels=None,
                       batch_size=32, shuffle=False):
        # data contain list[tuple(text1, text2)]
        # transform data into seq of embeddings
        input_ids, attention_masks, token_ids = self.convert_examples_to_features(
            data, max_seq_length)
        
        if labels is not None:
            assert len(labels) == len(data)
            labels = torch.tensor(labels)
            dataset = TensorDataset(input_ids, attention_masks, token_ids, labels)
        else:
            dataset = TensorDataset(input_ids, attention_masks, token_ids)

        # Run prediction for full data
        if shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        datasetloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

        return datasetloader

    def convert_examples_to_features(self, examples, max_seq_length):
        """Loads a data file into a list of `InputBatch`s."""
        encoded_dict = self.tokenizer.batch_encode_plus(
            examples,
            add_special_tokens = True,
            max_length = max_seq_length,
            pad_to_max_length = True,
            return_attention_mask = True,
            return_tensors='pt', 
        )
        input_ids = encoded_dict['input_ids']
        attention_masks = encoded_dict['attention_mask']
        token_type_ids = encoded_dict['token_type_ids']
        return input_ids, attention_masks, token_type_ids
    
    def get_training_dataloaders(self, args):
        dataloaders = {}
        for split in ['train', 'valid', 'test']:
            shuffle = True if split == 'train' else False
            batch_size_ = args.cls_batchSize if split == 'train' else args.cls_batchSize * 2
            data_path = os.path.join(args.training_dir, split)
            labels, texts = read_corpus(data_path + '.tsv', text_label_pair=True)
            # whether to use adversarial examples for mix training
            if split == 'train' and args.mix_training_data:
                print('%d orig examples loaded for %s.' % (len(texts), split))
                data_path_ = os.path.join(args.training_dir, split + '_adv')
                labels_, texts_ = read_adv_corpus(data_path_ + '.tsv', text_label_pair=True,
                    dataset=args.dataset, max_adv_len=args.max_adv_len, min_adv_len=args.min_adv_len,
                    adv_data_ratio=args.adv_data_ratio, max_num_change=args.max_num_change)
                labels = labels + labels_
                texts = texts + texts_
                assert len(labels) == len(texts)
                print('%d adversarial examples loaded for %s.' % (len(texts_), split))
                print('Mix those two examples...')
            dataloaders[split] = self.transform_text(texts, args.max_seq_length, labels,
                                                     batch_size=batch_size_, shuffle=shuffle)
            print('%d examples loaded for %s.' % (len(texts), split))
        return dataloaders

    def get_adv_dataloaders(self, args, return_dataloader=True):
        assert args.adv_data_ratio != 0
        shuffle = True
        batch_size_ = args.cls_batchSize
        data_path_ = os.path.join(args.training_dir, 'train_adv')
        labels_, texts_ = read_adv_corpus(data_path_ + '.tsv', text_label_pair=True,
            dataset=args.dataset, max_adv_len=args.max_adv_len, min_adv_len=args.min_adv_len,
            adv_data_ratio=args.adv_data_ratio, max_num_change=args.max_num_change)
        assert len(labels_) == len(texts_)
        print('%d adversarial examples loaded from %s. for weighted loss' % (len(texts_), data_path_))
        if return_dataloader:
            return self.transform_text(texts_, args.max_seq_length, labels_,
                                       batch_size=batch_size_, shuffle=shuffle)
        else:
            return texts_, labels_
        