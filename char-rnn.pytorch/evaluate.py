
import argparse
import unidecode

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import WordDataset
from helpers import char_tensor
from helpers import prep_data
from tqdm import tqdm

def evaluate(decoder, criterion, input_, target, batch_size, chunk_len, cuda):

    hidden = decoder.init_hidden(batch_size)
    if cuda:
        hidden = hidden.cuda()

    loss = 0
    for c in range(chunk_len):
        output, hidden = decoder(input_[:,c], hidden)
        loss += criterion(output.view(batch_size, -1), target[:,c])

    return loss.data[0] / chunk_len

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_file', type=str)
    argparser.add_argument('--cpu', action='store_true')
    argparser.add_argument('--cuda', action='store_true')
    argparser.add_argument('--chunk_len', type=int, default=200)
    argparser.add_argument('--batch_size', type=int, default=300)
    argparser.add_argument('--num_workers', type=int, default=8)
    argparser.add_argument('filename', type=str)
    args = argparser.parse_args()

    if args.cpu:
    	decoder = torch.load(args.model_file, map_location=lambda storage,
                             loc: storage)
    else:
    	decoder = torch.load(args.model_file)

    dataset = WordDataset(args.filename, args.chunk_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            num_workers=args.num_workers, drop_last=True)

    criterion = nn.CrossEntropyLoss()

    loss, num_samples = 0, 0
    for sample in dataloader:
        input_, target = prep_data(sample['input'], sample['target'], args.cuda)
        loss += evaluate(decoder, criterion, input_, target, args.batch_size,
                         args.chunk_len, args.cuda)
        num_samples += 1
    bpc = loss / num_samples # bits per character
    #perplexity = 2 ** (loss / num_samples)

    print('Loss: {:.2f}, BPC: {:.2f}'.format(loss, bpc))
    #print('Loss: {:.2f}, Perplexity: {:.2f}'.format(loss, perplexity))

if __name__ == '__main__':
    main()

