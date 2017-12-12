# https://github.com/spro/char-rnn.pytorch

# stdlib imports
import argparse
import os
import random
import sys
import time

# thirdparty imports
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

# local imports
from dataset import WordDataset, n_characters, time_since
from evaluate import evaluate
from helpers import read_file
from helpers import char_tensor
from helpers import prep_data
from model import CharRNN

def train(decoder, optimizer, criterion, inp, target, batch_size, chunk_len, cuda):
    hidden = decoder.init_hidden(batch_size)
    if cuda:
        if decoder.model == "gru":
            hidden = hidden.cuda()
        else: # lstm
            hidden = (hidden[0].cuda(), hidden[1].cuda())
    decoder.zero_grad()
    loss = 0

    for c in range(chunk_len):
        output, hidden = decoder(inp[:,c], hidden)
        loss += criterion(output.view(batch_size, -1), target[:,c])

    loss.backward()
    optimizer.step()

    return loss.data[0] / chunk_len

def model_file_name(decoder, epoch, train_loss, val_loss):
    f = 'models/{}_epoch{}_nlayers{}_input{}_output{}_hs{}_trainL{:.2f}_valL{:.2f}.pt'
    return f.format(decoder.model, epoch, decoder.n_layers, decoder.input_size,
                    decoder.output_size, decoder.hidden_size, train_loss, val_loss)

def main():

    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train_set', type=str, required=True)
    argparser.add_argument('--valid_set', type=str, required=True)
    argparser.add_argument('--model', type=str, default="gru")
    argparser.add_argument('--model_file', type=str, default='None')
    argparser.add_argument('--n_epochs', type=int, default=30)
    argparser.add_argument('--hidden_size', type=int, default=200)
    argparser.add_argument('--n_layers', type=int, default=3)
    argparser.add_argument('--learning_rate', type=float, default=0.01)
    argparser.add_argument('--chunk_len', type=int, default=200)
    argparser.add_argument('--batch_size', type=int, default=300)
    argparser.add_argument('--num_workers', type=int, default=8)
    argparser.add_argument('--cuda', action='store_true')
    argparser.add_argument('--cpu', action='store_true')
    args = argparser.parse_args()

    # Initialize models and start training
    
    if args.model_file == 'None':
        decoder = CharRNN(
            n_characters,
            args.hidden_size,
            n_characters,
            model=args.model,
            n_layers=args.n_layers,
        )
        epoch_from = 1
        prev_valid_loss = sys.maxsize
        old_filename = None
    else:
        if args.cpu:
            decoder = torch.load(args.model_file, map_location=lambda storage,
                                                                      loc: storage)
        else:
            decoder = torch.load(args.model_file)
        info = args.model_file.split('_')
        args.model = info[0]
        epoch_from = int(info[1][5:]) + 1
        args.n_layers = int(info[2][7:])
        args.hidden_size = int(info[5][2:])
        prev_valid_loss = float(info[7][4:-3])
        old_filename = args.model_file

        print("successfully loaded model! Continuing from epoch {0} with valid loss {1}"
              .format(epoch_from, prev_valid_loss))

    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    if args.cuda:
        decoder.cuda()

    start = time.time()

    train_dataset = WordDataset(args.train_set, args.chunk_len)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  drop_last=True)

    valid_dataset = WordDataset(args.valid_set, args.chunk_len)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  drop_last=True)

    try:

        print('Training for maximum {} epochs...'.format(args.n_epochs))
        for epoch in range(epoch_from, args.n_epochs + 1):


            train_loss, num_samples = 0, 0
            for s in tqdm(train_dataloader):
                input_, target = prep_data(s['input'], s['target'], args.cuda)
                train_loss += train(decoder, optimizer, criterion, input_,
                                    target, args.batch_size, args.chunk_len,
                                    args.cuda)
                num_samples += 1
            train_loss /= num_samples

            valid_loss, num_samples = 0, 0
            for s in valid_dataloader:
                input_, target = prep_data(s['input'], s['target'], args.cuda)
                valid_loss += evaluate(decoder, criterion, input_, target,
                                       args.batch_size, args.chunk_len, args.cuda)
                num_samples += 1
            valid_loss /= num_samples

            elapsed = time_since(start)
            pcnt = epoch / args.n_epochs * 100
            log = ('{} elapsed - epoch #{} ({:.1f}%) - training loss (BPC) {:.2f} '
                   '- validation loss (BPC) {:.2f}')
            print(log.format(elapsed, epoch, pcnt, train_loss, valid_loss))

            if valid_loss > prev_valid_loss:
                print('No longer learning, just overfitting, stopping here.')
                break
            else:
                filename = model_file_name(decoder, epoch, train_loss, valid_loss)
                torch.save(decoder, filename)
                print('Saved as {}'.format(filename))
                if old_filename:
                    os.remove(old_filename)
                old_filename = filename

            prev_valid_loss = valid_loss

    except KeyboardInterrupt:
        print("Saving before quit...")
        try: valid_loss
        except: valid_loss = 'no_val'
        filename = model_file_name(decoder, epoch, train_loss, valid_loss)
        torch.save(decoder, filename)
        print('Saved as {}'.format(filename))

if __name__ == '__main__':
    main()

