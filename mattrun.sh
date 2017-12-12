#!/usr/bin/env bash
python char-rnn.pytorch/train.py --train_set data/wiki_train.txt --valid_set data/wiki_valid.txt --model gru --cuda --hidden_size 700 --n_layers 7 --batch_size 150 --learning_rate 0.001 --model_file models/gru_epoch5_nlayers7_input100_output100_hs700_trainL1.20_valL1.28.pt
