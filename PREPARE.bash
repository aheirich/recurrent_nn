#!/bin/bash
python ${SINGULARITY_TORCH_RNN_PATH}/scripts/preprocess.py \
--input_txt ${SINGULARITY_TORCH_RNN_PATH}/data/tiny-shakespeare.txt \
--output_h5 ${TORCH_RNN_PATH}/data/tiny-shakespeare.h5 \
--output_json ${TORCH_RNN_PATH}/data/tiny-shakespeare.json


