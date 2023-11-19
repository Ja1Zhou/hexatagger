python run.py \
    predict \
    --lang English \
    --max-depth 10 \
    --tagger hexa \
    --bert-model-path xlnet-large-cased \
    --model-name English-hexa-bert-2e-05-50 \
    --batch-size 64 \
    --model-path ./checkpoints/