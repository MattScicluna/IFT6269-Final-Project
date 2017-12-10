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
    MODEL="models/mila.pt"
    SEED="It is trivial to see that: "
elif [ "$1" == "WIKI" ]; then
    DATASET="data/enwik8"
    MODEL="models/wiki.pt"
    SEED="A"
else
  echo "Unrecognised data set '$1'. Options are 'MILA' or 'WIKI'"
  exit -1
fi

if [ "$2" == "CUDA" ]; then
    CHIPFLAG="--cuda"
else
    CHIPFLAG="--cpu"
fi

python char-rnn.pytorch/train.py $DATASET $CHIPFLAG --model_file $MODEL
python char-rnn.pytorch/generate.py $MODEL -t 0.5 -l 200 --prime_str $SEED $CHIPFLAG

