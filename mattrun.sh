#!/usr/bin/env bash
python char-rnn.pytorch/train.py --train_set data/train.txt --valid_set data/valid.txt --model lstm --cuda --hidden_size 1000 --batch_size 300