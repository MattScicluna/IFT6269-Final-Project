from torch.utils.data import Dataset
import unidecode
import torch
import string
import time
import math

all_characters = string.printable
n_characters = len(all_characters)

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = all_characters.index(string[c])
        except:
            continue
    return tensor

class WordDataset(Dataset):
    """Our dataset."""

    def __init__(self, filename, chunk_len):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.filename = filename
        self.raw_data = unidecode.unidecode(open(filename).read())
        self.chunk_len = chunk_len

    def __len__(self):
        return len(self.raw_data) // self.chunk_len

    def __getitem__(self, idx):
        start_index = self.chunk_len*idx
        end_index = start_index + self.chunk_len + 1
        chunk = self.raw_data[start_index:end_index]
        inp = char_tensor(chunk[:-1])
        target = char_tensor(chunk[1:])
        return {'input': inp, 'target': target}
