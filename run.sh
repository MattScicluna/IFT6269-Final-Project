#!/usr/bin/env bash

# Usage:
#     $ bash run.sh MILA
# or
#     $ bash run.sh WIKI
# or
#     $ bash run.sh MILA CUDA
# or
#     $ bash run.sh WIKI CUDA


if [ "$1" == "MILA" ]; then
    DATASET="data/train.txt"
    MODEL="data/mila.pt"
    SEED="It is trivial to see that: "
elif [ "$1" == "WIKI" ]; then
    DATASET="data/enwik8"
    MODEL="data/wiki.pt"
    SEED="A"
else
  echo "Unrecognised data set '$1'. Options are 'MILA' or 'WIKI'"
  exit -1
fi

if [ "$2" == "CUDA" ]; then
    CUDAFLAG="--cuda"
else
    CUDAFLAG=""
fi

python char-rnn.pytorch/train.py $DATASET $CUDAFLAG --model_file $MODEL
python char-rnn.pytorch/generate.py $MODEL -t 0.5 -l 200 --prime_str $SEED $CUDAFLAG

