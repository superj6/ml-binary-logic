import torch
from torch.utils.data import Dataset
import csv
import numpy as np

class XorDataset(Dataset):
    def __init__(self, csv_file):
        self.data = []
        with open(csv_file, 'r') as f:
            r = csv.reader(f)
            for row in r:
                x = list(map(int, row))
                x = (x[:2], x[2])
                self.data.append(x)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = torch.tensor(self.data[idx][0], dtype = torch.float)
        y = torch.tensor([self.data[idx][1]], dtype = torch.float)
        return X, y

class GatesDataset(Dataset):    
    def __init__(self, csv_file):
        self.data = []
        with open(csv_file, 'r') as f:
            r = csv.reader(f)
            for row in r:
                x = (list(map(int, row[:2])), row[2], int(row[3]))
                self.data.append(x)

        self.ops = '|&^'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        op_array = [1 if self.data[idx][1] == self.ops[i] else 0 for i in range(len(self.ops))]
        data_in = self.data[idx][0] + op_array
        
        X = torch.tensor(data_in, dtype = torch.float)
        y = torch.tensor([self.data[idx][2]], dtype = torch.float)

        return X, y
