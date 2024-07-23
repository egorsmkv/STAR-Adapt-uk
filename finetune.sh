#!/usr/bin/env bash

source .venv/bin/activate

dataset=yodas2
model_size=large-v3
train_data=/home/yehor/ext-ml-disk/experiments/data-filter/data/kaldi_uk000/train/
dev_data=/home/yehor/ext-ml-disk/experiments/data-filter/data/kaldi_uk000/test/

python finetune.py \
    --MODEL "openai/whisper-${model_size}" \
    --DATASET ${dataset} \
    --TRAIN_DATA ${train_data} \
    --DEV_DATA ${dev_data} \
    --BATCH_SIZE 1 \
    --GRADIENT_ACCUMULATION_STEPS 16 \
    --LEARNING_RATE 1e-5 \
    --EPOCHS 2
