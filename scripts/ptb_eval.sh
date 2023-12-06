#!/usr/bin/env bash
# caveat: training hyperparameters seem inconsistent
# according to filenaming conventions, it seems that the training should be using lr=3e-05
python run.py \
    evaluate \
    --lang English \
    --max-depth 10 \
    --tagger hexa \
    --bert-model-path xlnet-large-cased \
    --model-name English-hexa-bert-2e-05-50 \
    --batch-size 128 \
    --model-path ./checkpoints/