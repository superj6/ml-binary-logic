from torch.utils.data import DataLoader
import torch
from torch import nn

from data import loader
from models import gates

def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            # initialize the weight tensor, here we use a normal distribution
            m.weight.data.normal_(0, 1)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X, yeet)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X, yeet)
            test_loss += loss_fn(pred, y).item()
            correct += ((pred > 0.5) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == '__main__':
    trainloader = loader.get_gates_loader('./data/csv/gates.csv')
    model = gates.GatesModel()

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.02, momentum = 0.9)

    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    weights_init(model)

    yeet = 1
    for i in range(1500):
        train(trainloader, model, loss_fn, optimizer)
        
        if i % 100 == 0:
            test(trainloader, model, loss_fn)
            yeet += (i / 400) * (i / 400)


    for X, y in trainloader:
        X, y = X.to(device), y.to(device)
        pred = model(X, yeet)
        print(X, y, '-', pred)

    print(list(model.parameters()))
