import torch
from torch import nn
import torch.nn.functional as F

class XorModel(nn.Module):
    #(x, y) -> (x xor y)

    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(2, 2)
        self.lin2 = nn.Linear(2, 1)

    def forward(self, x):
        x = F.sigmoid(self.lin1(x))
        x = self.lin2(x)
        return x

class GatesModel(nn.Module):
    #(x, y, op_and, op_or, op_xor, op_nor) -> (x op y)

    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(5, 2)
        self.lin2 = nn.Linear(2, 1)

    def forward(self, x, yeet):
        x = F.sigmoid(yeet * self.lin1(x))
        x = self.lin2(x)
        return x

