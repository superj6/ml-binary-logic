from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from . import dataset

def get_xor_loader(csv_file):
    trainset = dataset.XorDataset(csv_file)
    trainloader = DataLoader(trainset, shuffle = True, batch_size = 1)

    return trainloader

def get_gates_loader(csv_file):
    trainset = dataset.GatesDataset(csv_file)
    trainloader = DataLoader(trainset, shuffle = True, batch_size = 1)

    return trainloader
