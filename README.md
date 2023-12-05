# Hexatagging: Projective Dependency Parsing as Tagging
<p align="center">
  <img src="./header.jpg" width=400>
  <img src="./header-hexa.png" width=400>
</p>
This repository contains reproduction code and scripts for paper: 

[Hexatagging: Projective Dependency Parsing as Tagging
](https://aclanthology.org/2023.acl-short.124/)

## Setting Up The Environment
Here we include more detailed steps on setting up the env. We encountered subtle errors using merely `pip install -r requirements.txt`.
```bash
# create conda env with python=3.10.13
conda create -n hexatag python=3.10.13 -y
conda activate hexatag

# install conda rust env to build `pytokenizations`
conda install -c conda-forge rust -y

# checkout https://pytorch.org/ to determine the right command for installing torch with cuda
pip install torch --index-url https://download.pytorch.org/whl/cu118

# install other requirements
pip install -r requirements.txt
```

## Getting The Data
### Initial Processing
The authors refer to instructions in [this repo](https://github.com/nikitakit/self-attentive-parser/tree/master/data) for initial preprocessing on the English WSJ and Chinese Treebank datasets. Here we list the steps relevant to our reproduction.
#### English WSJ

1. Place a copy of the Penn Treebank
([LDC99T42](https://catalog.ldc.upenn.edu/LDC99T42)) under `data/raw/treebank_3`.
After doing this, `data/raw/treebank_3/parsed/mrg/wsj` should have folders
named `00`-`24`.
2. Place a copy of the revised Penn Treebank
([LDC2015T13](https://catalog.ldc.upenn.edu/LDC2015T13)) under
`data/raw/eng_news_txt_tbnk-ptb_revised`.
3. `cd data/wsj && ./build_corpus.sh`

Processed trees are written to the following files:


| File in `data/wsj/`                | Description                                         |
| ---------------------------------- | --------------------------------------------------- |
| `train_02-21.LDC99T42`             | The standard training split for English constituency parsing publications. |
| `train_02-21.LDC99T42.text`        | Non-destructive tokenization of the training split, meaning that words retain their original spelling (without any escape sequences), and information about whitespace between tokens is preserved. |
| `dev_22.LDC99T42`                  | The standard development/validation split for English constituency parsing publications. |
| `dev_22.LDC99T42.text`             | Non-destructive tokenization of the development split. |
| `test_23.LDC99T42`                 | The standard test split for English constituency parsing publications. |
| `test_23.LDC99T42.text`            | Non-destructive tokenization of the test split. |
| `train_02-21.LDC2015T13`           | English training sentences with revised tokenization and syntactic annotation guidelines. These revised guidelines were also used for other treebanks, including the English Web Treebank and OntoNotes. |
| `train_02-21.LDC2015T13.text`      | Non-destructive revised tokenization of the training sentences. |
| `dev_22.LDC2015T13`                | English development/validation sentences with revised tokenization and syntactic annotation guidelines. |
| `dev_22.LDC2015T13.text`           | Non-destructive revised tokenization of the development sentences. |
| `test_23.LDC2015T13`               | English test sentences with revised tokenization and syntactic annotation guidelines. |
| `test_23.LDC2015T13.text`          | Non-destructive revised tokenization of the test sentences. |
| `train_02-21.LDC99T42.retokenized` | Syntatic annotations (labeled brackets) from the standard training split, overlaid on top of the revised tokenization. |
| `dev_22.LDC99T42.retokenized`      | Syntatic annotations (labeled brackets) from the standard development/validation split, overlaid on top of the revised tokenization. |
| `test_23.LDC99T42.retokenized`     | Syntatic annotations (labeled brackets) from the standard test split, overlaid on top of the revised tokenization. |
#### Chinese Treebank (CTB 5.1)

This prepares the standard Chinese constituency parsing split, following recent papers such as [Liu and Zhang (2017)](https://www.aclweb.org/anthology/Q17-1004/).

1. Place a copy of the Chinese Treebank 5.1
([LDC2005T01](https://catalog.ldc.upenn.edu/LDC2005T01)) in `data/raw/ctb5.1_507K`.
2. `cd data/ctb_5.1 && ./build_corpus.sh`

Processed trees are written to the following files:


| File in `data/ctb_5.1/` | Description                                         |
| ----------------------- | --------------------------------------------------- |
| `ctb.train`             | The standard training split for Chinese constituency parsing publications. |
| `ctb.dev`               | The standard development split for Chinese constituency parsing publications. |
| `ctb.test`              | The standard test split for Chinese constituency parsing publications. |

### Generating Binary-Headed-Trees
Convert CoNLL to Binary Headed Trees:
```bash
python data/dep2bht.py
```
This will generate the phrase-structured BHT trees in the `data/bht` directory. 
(The authors placed the processed files already under the `data/bht` directory.)

## Building The Tagging Vocab
In order to use taggers, we need to build the vocabulary of tags for in-order, pre-order and post-order linearizations. You can cache these vocabularies using:
```bash
python run.py vocab --lang [LANGUAGE] --tagger [TAGGER]
```

For our purposes, we specify tagger to be `hexa` and generate vocab for English and Chinese.
```bash
for lang in English Chinese; do
    python run.py vocab --lang $lang --tagger hexa
done
```

## Training
### PTB (English)
```bash
CUDA_VISIBLE_DEVICES=0 python run.py train --lang English --max-depth 6 --tagger hexa --model bert --epochs 50 --batch-size 32 --lr 2e-5 --model-path xlnet-large-cased --output-path ./checkpoints/ --use-tensorboard True
# model saved at ./checkpoints/English-hexa-bert-2e-05-50
```
### CTB (Chinese)
```bash
CUDA_VISIBLE_DEVICES=0 python run.py train --lang Chinese --max-depth 6 --tagger hexa --model bert --epochs 50 --batch-size 32 --lr 2e-5 --model-path hfl/chinese-xlnet-mid --output-path ./checkpoints/ --use-tensorboard True
# model saved at ./checkpoints/Chinese-hexa-bert-2e-05-50
```

## Evaluation
### PTB
```bash
python run.py evaluate --lang English --max-depth 10 --tagger hexa --bert-model-path xlnet-large-cased --model-name English-hexa-bert-2e-05-50 --batch-size 64 --model-path ./checkpoints/
```

### CTB
```bash
python run.py evaluate --lang Chinese --max-depth 10 --tagger hexa --bert-model-path bert-base-chinese --model-name Chinese-hexa-bert-2e-05-50 --batch-size 64 --model-path ./checkpoints/
```

## Predict
### PTB
```bash
python run.py predict --lang English --max-depth 10 --tagger hexa --bert-model-path xlnet-large-cased --model-name English-hexa-bert-3e-05-50 --batch-size 64 --model-path ./checkpoints/
```

# Citation
If you find this repository useful, please cite our papers:
```bibtex
@inproceedings{amini-cotterell-2022-parsing,
    title = "On Parsing as Tagging",
    author = "Amini, Afra  and
      Cotterell, Ryan",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.607",
    pages = "8884--8900",
}
```

```bibtex
@inproceedings{amini-etal-2023-hexatagging,
    title = "Hexatagging: Projective Dependency Parsing as Tagging",
    author = "Amini, Afra  and
      Liu, Tianyu  and
      Cotterell, Ryan",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-short.124",
    pages = "1453--1464",
}
```

