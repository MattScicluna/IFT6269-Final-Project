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

def save(decoder, model_file, filename, epoch, train_loss, valid_loss):
    if model_file:
        filename = model_file
    else:
        #filename = os.path.splitext(os.path.basename(filename))[0] 
        filename = decoder.model + '_epoch' + str(epoch) + '_nlayers' + str(decoder.n_layers) \
                + '_input' + str(decoder.input_size) + '_output' + str(decoder.output_size) \
                + '_hs' + str(decoder.hidden_size) + '_trainL' + str(train_loss)  \
                + '_valL' + str(valid_loss) + '.pt'
    torch.save(decoder, 'models/' + filename)
    print('Saved as {}'.format(filename))


def main():

    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train_set', type=str, required=True)
    argparser.add_argument('--valid_set', type=str, required=True)
    argparser.add_argument('--model', type=str, default="gru")
    argparser.add_argument('--n_epochs', type=int, default=30)
    argparser.add_argument('--hidden_size', type=int, default=200)
    argparser.add_argument('--n_layers', type=int, default=3)
    argparser.add_argument('--learning_rate', type=float, default=0.01)
    argparser.add_argument('--chunk_len', type=int, default=200)
    argparser.add_argument('--batch_size', type=int, default=300)
    argparser.add_argument('--num_workers', type=int, default=8)
    argparser.add_argument('--cuda', action='store_true')
    argparser.add_argument('--cpu', action='store_true')
    argparser.add_argument('--model_file', type=str)
    args = argparser.parse_args()

    # Initialize models and start training

    decoder = CharRNN(
        n_characters,
        args.hidden_size,
        n_characters,
        model=args.model,
        n_layers=args.n_layers,
    )

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
        prev_valid_loss = sys.maxsize

        print('Training for {} epochs...'.format(args.n_epochs))
        for epoch in range(1, args.n_epochs + 1):
            train_loss = 0
            for s in train_dataloader:
                input_, target = prep_data(s['input'], s['target'], args.cuda)
                train_loss += train(decoder, optimizer, criterion, input_,
                                    target, args.batch_size, args.chunk_len,
                                    args.cuda)

            valid_loss, num_samples = 0, 0
            for s in valid_dataloader:
                input_, target = prep_data(s['input'], s['target'], args.cuda)
                valid_loss += evaluate(decoder, criterion, input_, target,
                                       args.batch_size, args.chunk_len, args.cuda)
                num_samples += 1
            bpc = valid_loss / num_samples # bits per character

            elapsed = time_since(start)
            pcnt = epoch / args.n_epochs * 100
            log = ('{} elapsed - epoch #{} ({:.1f}%) - training loss {:.2f} '
                   '- validation loss {:.2f} - BPC {:.2f}')
            print(log.format(elapsed, epoch, pcnt, train_loss, valid_loss, bpc))

            if valid_loss > prev_valid_loss:
                print('No longer learning, just overfitting, stopping here.')
                break

            prev_valid_loss = valid_loss

        print("Saving...")
        save(decoder, args.model_file, args.train_set, epoch, train_loss, valid_loss)

    except KeyboardInterrupt:
        print("Saving before quit...")
        try: valid_loss
        except: valid_loss = 'no_val'
        save(decoder, args.model_file, args.train_set, epoch, train_loss, valid_loss)

if __name__ == '__main__':
    main()

