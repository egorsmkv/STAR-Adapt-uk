#!/usr/bin/env bash

source .venv/bin/activate

dataset=yodas2
model_size=large-v3
checkpoint=<checkpoint-path>
test_data=/home/yehor/ext-ml-disk/experiments/data-filter/data/kaldi_uk000/test/

python inference.py \
    --MODEL "openai/whisper-${model_size}" \
    --DATASET ${dataset} \
    --CKPT ${checkpoint} \
    --TEST_DATA ${test_data}
