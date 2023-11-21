#!/usr/bin/env bash
python run.py \
    evaluate \
    --lang Chinese \
    --max-depth 10 \
    --tagger hexa \
    --bert-model-path hfl/chinese-xlnet-mid \
    --model-name Chinese-hexa-bert-2e-05-50 \
    --batch-size 64 \
    --model-path ./checkpoints/