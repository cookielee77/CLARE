import os
import sys
import argparse


def load_arguments(dataset=None):
    argparser = argparse.ArgumentParser()
    ## Required parameters
    argparser.add_argument("--dataset",
                        type=str,
                        required=True,
                        help="Which dataset to attack.")
    argparser.add_argument("--target_model",
                        type=str,
                        choices=['bert', 'textcnn'],
                        default='bert',
                        help="Target models for tasks")
    argparser.add_argument("--case",
                        type=str,
                        default='uncased',
                        choices=['uncased', 'cased'],
                        help="Whether use cased model for BERT model")
    argparser.add_argument("--USE_cache_path",
                        type=str,
                        default='./tmp',
                        help="Path to the USE encoder cache.")
    argparser.add_argument("--output_dir",
                        type=str,
                        default='adv_results',
                        help="The output directory where the attack results will be written.")
    argparser.add_argument("--sample_file",
                        type=str,
                        default='dump',
                        help="name of sample file to write.")
    argparser.add_argument("--attack_file",
                        type=str,
                        default='',
                        help="the attack data path")
    argparser.add_argument("--write_into_tsv",
                        action='store_true',
                        help="whether write adversarial examples into tsv file")
    argparser.add_argument("--data_idx",
                        default=0,
                        type=int,
                        help="The start index in the dataset")
    argparser.add_argument("--data_size",
                        default=None,
                        type=int,
                        help="Data size to create adversaries")
    
    # Baseline attack model parameters
    argparser.add_argument("--baseline_type",
                        type=str,
                        choices=['textfooler', 'pwws', 'random'],
                        default='textfooler',
                        help="baseline attack type")
    argparser.add_argument("--word_embeddings_path",
                        type=str,
                        default='',
                        help="path to the word embeddings for the target model")
    argparser.add_argument("--counter_fitting_embeddings_path",
                        type=str,
                        default='counter-fitted-vectors.txt',
                        help="path to the counter-fitting embeddings we used to find synonyms")
    argparser.add_argument("--counter_fitting_cos_sim_path",
                        type=str,
                        default='cos_sim_counter_fitting.npy',
                        help="pre-compute the cosine similarity scores based on the counter-fitting embeddings")

    ## Model hyperparameters
    argparser.add_argument("--synonym_num",
                        default=50,
                        type=int,
                        help="Number of synonyms to extract")
    argparser.add_argument("--attack_second",
                        action='store_true',
                        help="whether attack the second sentence for two sentences dataset")
    argparser.add_argument("--attack_loc",
                        type=str,
                        choices=['brutal_force', 'silence_score', 'pos_tag_filter'],
                        default='brutal_force',
                        help="methods to find attack locations in BERTAttack")
    argparser.add_argument("--batch_size",
                        default=128,
                        type=int,
                        help="Batch size to get prediction for the target model")
    argparser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="max sequence length for BERT target model")
    
    # Training parameters
    argparser.add_argument('--cls_lr',
                        type=float,
                        default=2e-5,
                        help='classification learning rate')
    argparser.add_argument('--cls_epochs',
                        type=int,
                        default=3,
                        help='max training epochs for classification')
    argparser.add_argument('--cls_batchSize',
                        type=int,
                        default=32,
                        help='batch size for classification training')
    argparser.add_argument('--training_dir',
                        type=str,
                        default="./data/training_data",
                        help='training directory for classifier')
    argparser.add_argument("--load_dataset_from_cache",
                        action='store_true',
                        help="whether load the dataset from pkl cache")
    argparser.add_argument('--target_model_name',
                        type=str,
                        default="",
                        help='the saved model name in traning')
    argparser.add_argument('--save_model',
                        action='store_true',
                        help='whether save the best model after the training')
    
    # adversarial training part
    argparser.add_argument('--mix_training_data',
                        action='store_true',
                        help='whether mix adversarial examples with regular training data')
    argparser.add_argument("--max_num_change",
                        type=int,
                        default=None,
                        help="max number of changes for adversarial examples")
    argparser.add_argument("--max_adv_len",
                        type=int,
                        default=None,
                        help="max length for adversarial exampels.")
    argparser.add_argument("--min_adv_len",
                        type=int,
                        default=None,
                        help="min length for adversarial exampels.")
    argparser.add_argument("--adv_data_ratio",
                        type=float,
                        default=1.0,
                        help="the ratio potion of used adversarial examples")
    argparser.add_argument('--adv_epoch',
                        type=int,
                        default=1,
                        help='The epoch start for adversarial training')
    argparser.add_argument('--adv_weight',
                        type=float,
                        default=0.01,
                        help='The weight for adversarial training examples.')
    argparser.add_argument('--num_limited_files',
                        type=int,
                        default=0,
                        help='The number files for limited training.')

    if dataset:
        args = argparser.parse_args(['--dataset', dataset])
    else:
        args = argparser.parse_args()

    args.output_dir = os.path.join(args.output_dir, args.dataset)
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    else:
        os.makedirs(args.output_dir, exist_ok=True)
    if args.attack_file is "":
        args.attack_file = os.path.join('./data', args.dataset + '.tsv')
    if args.num_limited_files != 0:
        args.training_dir = "%s/%s_%d/%s" % (args.training_dir, args.dataset, args.num_limited_files, 'dataset')
    else:
        args.training_dir = os.path.join(args.training_dir, args.dataset, 'dataset')
    if args.target_model:
        args.sample_file = '_'.join([args.sample_file, args.target_model])
    if args.target_model == 'bert':
        args.target_model = '_'.join([args.target_model, args.case])
        args.sample_file = '_'.join([args.sample_file, args.case])
    if args.target_model_name == "":
        args.target_model_path = os.path.join(
            './saved_model', '_'.join([args.dataset, args.case]))
    else:
        args.target_model_path = os.path.join(
            './saved_model', args.target_model_name)
    print(args)
    return args