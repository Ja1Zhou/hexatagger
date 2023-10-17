#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=0 python run.py train --lang English --max-depth 6 --tagger hexa --model bert --epochs 50 --batch-size 32 --lr 2e-5 --model-path xlnet-large-cased --output-path ./checkpoints/ --use-tensorboard True
# CUDA_VISIBLE_DEVICES=0 \
python \
    -Xfrozen_modules=off \
    -m debugpy --listen 9999 --wait-for-client \
    run.py train \
    --lang English \
    --max-depth 6 \
    --tagger hexa \
    --model bert \
    --epochs 50 \
    --batch-size 32 \
    --lr 2e-5 \
    --model-path xlnet-large-cased \
    --output-path ./checkpoints/ \
    --use-tensorboard True