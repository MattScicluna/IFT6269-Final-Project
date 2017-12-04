#!/usr/bin/env bash
cd char-rnn.pytorch/
python train.py ../data/all_tex_files.txt --cuda
python generate.py all_tex_files.pt --cuda -t 0.5 -l 200 --prime_str "It is trivial to see that: "
