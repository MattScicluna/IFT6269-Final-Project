# https://github.com/spro/char-rnn.pytorch

# stdlib imports
import argparse
import os
import random
import time

# thirdparty imports
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

# local imports
from dataset import WordDataset, n_characters, time_since
from generate import generate
from helpers import read_file
from helpers import char_tensor
from model import CharRNN

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str)
argparser.add_argument('--model', type=str, default="gru")
argparser.add_argument('--n_epochs', type=int, default=30)
argparser.add_argument('--print_every', type=int, default=2)
argparser.add_argument('--hidden_size', type=int, default=200)
argparser.add_argument('--n_layers', type=int, default=3)
argparser.add_argument('--learning_rate', type=float, default=0.01)
argparser.add_argument('--chunk_len', type=int, default=200)
argparser.add_argument('--batch_size', type=int, default=300)
argparser.add_argument('--num_workers', type=int, default=8)
argparser.add_argument('--cuda', action='store_true')
argparser.add_argument('--full_dataset', action='store_true')
args = argparser.parse_args()

def prep_data(inp, target):
    inp = Variable(inp)
    target = Variable(target)
    if args.cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target

def random_training_set(chunk_len, batch_size):
    inp = torch.LongTensor(batch_size, chunk_len)
    target = torch.LongTensor(batch_size, chunk_len)
    for bi in range(batch_size):
        start_index = random.randint(0, file_len - chunk_len)
        end_index = start_index + chunk_len + 1
        chunk = file[start_index:end_index]
        inp[bi] = char_tensor(chunk[:-1])
        target[bi] = char_tensor(chunk[1:])
    inp = Variable(inp)
    target = Variable(target)
    if args.cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target

def train(inp, target):
    hidden = decoder.init_hidden(args.batch_size)
    if args.cuda:
        hidden = hidden.cuda()
    decoder.zero_grad()
    loss = 0

    for c in range(args.chunk_len):
        output, hidden = decoder(inp[:,c], hidden)
        loss += criterion(output.view(args.batch_size, -1), target[:,c])

    loss.backward()
    decoder_optimizer.step()

    return loss.data[0] / args.chunk_len

def save():
    save_filename = os.path.splitext(os.path.basename(args.filename))[0] + '.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)

# Initialize models and start training

decoder = CharRNN(
    n_characters,
    args.hidden_size,
    n_characters,
    model=args.model,
    n_layers=args.n_layers,
)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()

if args.cuda:
    decoder.cuda()

start = time.time()
loss_avg = 0

if args.full_dataset:
    train_dataset = WordDataset(args.filename, args.chunk_len)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers,
                            drop_last=True)
else:
    file, file_len = read_file(args.filename)

try:
    print('Training for {} epochs...'.format(args.n_epochs))
    for epoch in tqdm(range(1, args.n_epochs + 1)):
        if args.full_dataset:
            loss = 0
            for sample in dataloader:
                loss += train(*prep_data(sample['input'], sample['target']))
                loss_avg += loss
        else:
            loss = train(*random_training_set(args.chunk_len, args.batch_size))
            loss_avg += loss

        if epoch % args.print_every == 0:
            pcnt = epoch / args.n_epochs * 100
            log = '{} ({} {}%) {:.4f}'
            print(log.format(time_since(start), epoch, pcnt, loss))
            seed = 'It is trivial to see that '
            print(generate(decoder, seed, 100, cuda=args.cuda), '\n')

    print("Saving...")
    save()

except KeyboardInterrupt:
    print("Saving before quit...")
    save()

