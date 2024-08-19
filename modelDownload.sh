#!/bin/bash

current_dir=$(pwd)
parentdir="$(dirname "$current_dir")"
mkdir -p $parentdir/hf

export HF_HOME=$parentdir/hf

huggingface-cli login

cd groundtruth
python test.py

cd ../pipeline/utils
python weightSaver.py $parentdir/hf

