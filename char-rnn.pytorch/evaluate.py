
import argparse
import unidecode

import torch
from torch.autograd import Variable
import torch.nn as nn

from helpers import char_tensor

def evaluate(decoder, input_):
    criterion = nn.CrossEntropyLoss()
    hidden = decoder.init_hidden(1)
    output, hidden = decoder(input_[:-1], hidden)
    loss = criterion(output, input_[1:])
    perplexity = 2 ** loss
    return loss.data[0], perplexity.data[0]

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_file', type=str)
    argparser.add_argument('--cpu', action='store_true')
    argparser.add_argument('--cuda', action='store_true')
    argparser.add_argument('filename', type=str)
    args = argparser.parse_args()

    if args.cpu:
    	decoder = torch.load(args.model_file, map_location=lambda storage,
                             loc: storage)
    else:
    	decoder = torch.load(args.model_file)

    raw_data = unidecode.unidecode(open(args.filename).read())
    input_ = Variable(char_tensor(raw_data)[:100])

    loss, perplexity = evaluate(decoder, input_)
    print('Loss: {:.2f}, Perplexity: {:.2f}'.format(loss, perplexity))

if __name__ == '__main__':
    main()

