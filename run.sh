#!/usr/bin/env bash
cd char-rnn.pytorch/
python train.py data/all_tex_files.txt --cuda
python generate.py all_tex_files.pt --cuda