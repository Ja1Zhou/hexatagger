#!/usr/bin/env bash
python \
    run.py train \
    --lang Chinese \
    --max-depth 6 \
    --tagger hexa \
    --model bert \
    --epochs 50 \
    --batch-size 32 \
    --lr 2e-5 \
    --model-path hfl/chinese-xlnet-mid \
    --output-path ./checkpoints/ \
    --use-tensorboard True
