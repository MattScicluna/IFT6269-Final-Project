#!/usr/bin/env bash
cd char-rnn.pytorch/
python train.py ../data/enwik8 --cuda
python generate.py enwik8.pt --cuda -t 0.5 -l 200 --prime_str "A"
