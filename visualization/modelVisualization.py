import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from models.binaryNN import BinaryNeuralNetwork
import torch.nn.functional as F
import torch.optim as optim
from torchviz import make_dot

# Check mps maybe if working in MacOS
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

'''
Importing MNIST dataset
'''
print(f'IMPORT DATASET\n')

training_data = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

neuronPerLayer = 20
nHiddenLayers = 3

binaryNNModel = BinaryNeuralNetwork(neuronPerLayer).to(device)

'''
Train and test functions
'''


def train(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = F.nll_loss(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += F.nll_loss(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


'''
Train and test
'''
print(f'TRAINING\n')

optBinary = optim.Adamax(binaryNNModel.parameters(), lr=3e-3, weight_decay=1e-4)

epochs = 1

print('Train and test binary NN')
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, binaryNNModel, optBinary)
    test(test_dataloader, binaryNNModel)

batch = next(iter(train_dataloader))
yhat = binaryNNModel(batch[0])

make_dot(yhat, params=dict(list(binaryNNModel.named_parameters()))).render("binaryNN_torchviz", format="png")
