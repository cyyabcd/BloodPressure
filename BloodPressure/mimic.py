import os
import os.path
import numpy as np
import torch
import codecs
import torch.utils.data as data

train_len = 100
test_len =10

class MIMIC(data.Dataset):
    def __init__(self, root, train = True):
        return super().__init__(*args, **kwargs)
    def __getitem__(self, index):
        return super().__getitem__(index)
    def __len__(self):
        if self.train:
            return train_len
        else:
            return test_len 
