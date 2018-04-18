#!/bin/bash
export SINGULARITY_TORCH_RNN_PATH=/opt/torch-rnn
export LUA_PATH=${LUA_PATH}:${SINGULARITY_TORCH_RNN_PATH}:/opt/torch/extra/cutorch
export PYTHONPATH=${PYTHONPATH}:${TORCH_RNN_PATH}:${TORCH_RNN_PATH}/trained
