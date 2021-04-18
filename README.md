# Contextualized Perturbation for Textual Adversarial Attack

## Introduction
This is a PyTorch implementation of [Contextualized Perturbation for Textual Adversarial Attack](https://arxiv.org/abs/2009.07502) by Dianqi Li, Yizhe Zhang, Hao Peng, Liqun Chen, Chris Brockett, Ming-Ting Sun and Bill Dolan, NAACL 2021.

A third-party implementation of CLARE is available in the [TextAttack](https://github.com/QData/TextAttack).

## Environment
The code is based on python 3.6, tensorflow 1.14 and Pytorch 1.4.0 version. The code is developed and tested using one NVIDIA GTX 1080Ti. 

Please use Conda to setup your environment, and then run
```
conda install -y pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch

bash install_requirement.sh
```

## Data Preparation and Pretrained Classifier
You can download pretrained target classifier and full training data in here (Coming soon).
Alternatively, you can prepare you own training set in the same format as the example under `/data/training_data/${dataset}/dataset/`. The format will look like:
| label | text1 | text2 |
|---|---|---|
| 2 | At the end of 5 years ... | The healthcare agency will be able ... |

For single sentence classification, there is an empty field in `text2`.

After this, please run:
```
python train_BERT_classifier.py --dataset ${dataset} --save_model.
```
It will save pretrained classifer under the director: `/saved_model/${dataset}_uncased/`. The default target classifer is `bert`, you can train other types by setting extra argument: `--target_model textcnn`. Please check out the arguments in `config.py` for more details. 

The text samples to be attacked are store in `/data/${dataset}.tsv` with the same format. 

## Textual Adversarial Attack
Simply run:
```
python bert_attack_classification.py --dataset ${dataset} --sample_file ${dataset}
```
and it will save the results under `/adv_results/`. 

To attack `qnli` dataset, please add an argument `--attack_second` as we attack the longer sentence in two-sentence classification.

You can also modify the attacking hyper-parameters in `hyper_parameters.py` to adjust the trade-off between different aspects. Other details can be refered in `config.py`.

To run the attack from the baseline `textfooler`:
```
python attack_classification.py --dataset ${dataset} --sample_file ${dataset}
```


## Citing
if you find our work is useful in your research, please consider citing: 
```
@InProceedings{li2021contextualized,
  title={Contextualized perturbation for textual adversarial attack},
  author={Li, Dianqi and Zhang, Yizhe and Peng, Hao and Chen, Liqun and Brockett, Chris and Sun, Ming-Ting and Dolan, Bill},
  booktitle={Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics},
  year={2021}
}
```



