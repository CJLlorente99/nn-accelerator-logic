import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from binaryNN import BinaryNeuralNetwork
from fpNN import FPNeuralNetwork
from hpNN import HPNeuralNetwork
import torch.nn.functional as F
import torch.optim as optim
from ttGenerator import TTGenerator

# Check mps maybe if working in MacOS
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

'''
Importing MNIST dataset
'''
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

'''
Show basic info of the dataset
'''
print(f'Classes and idx \n {training_data.class_to_idx}')
print(f'Number of samples in training data is {len(training_data)}')
print(f'Number of samples in test data is {len(test_data)}')
print(f'Size of samples is {training_data[0][0].size()}')

'''
Create DataLoader
'''
batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

'''
Instantiate NN models
'''
fpNNModel = FPNeuralNetwork().to(device)
binaryNNModel = BinaryNeuralNetwork(5).to(device)
hpNNModel = HPNeuralNetwork().to(device)

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
optFP = optim.Adamax(fpNNModel.parameters(), lr=3e-3, weight_decay=1e-4)
optHP = optim.Adamax(hpNNModel.parameters(), lr=3e-3, weight_decay=1e-4)
optBinary = optim.Adamax(binaryNNModel.parameters(), lr=3e-3, weight_decay=1e-4)

epochs = 1

# print('Train and test FP NN')
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train(train_dataloader, fpNNModel, optFP)
#     test(test_dataloader, fpNNModel)
#
# print('Train and test HP NN')
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train(train_dataloader, hpNNModel, optHP)
#     test(test_dataloader, hpNNModel)

print('Train and test binary NN')
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, binaryNNModel, optBinary)
    test(test_dataloader, binaryNNModel)

'''
Generate TT
'''

accLayers = TTGenerator.getAccLayers(binaryNNModel)
neurons = TTGenerator.getNeurons(accLayers)


'''
Calculate importance per class per neuron
'''

'''
For a batch of samples, do the following
1) Input training sample
2) Compute backward
3) Go neuron by neuron looking for the grad
4) Compute importance score as absolute value of output value times gradient
5) Add value to list of class (one list per class) if above threshold
6) Compute importance as percentage (length of the list divided by batch size)
7) Compute importance as sum of percentages
'''


