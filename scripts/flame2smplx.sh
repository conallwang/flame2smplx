#!/bin/bash

# cd /path/to/flame2smplx
export CUDA_VISIBLE_DEVICES=0

CFG='./config_files/flame2smplx.yaml'
python -m transfer_model --exp-cfg $CFG
