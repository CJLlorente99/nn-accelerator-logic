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
from torch.autograd import Variable
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from ttGenerator import BinaryOutputNeuron

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

# TODO. Target distribution can be critical for importance calculation

'''
Create DataLoader
'''
batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

'''
Instantiate NN models
'''
neuronPerLayer = 100

fpNNModel = FPNeuralNetwork().to(device)
binaryNNModel = BinaryNeuralNetwork(neuronPerLayer).to(device)
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

epochs = 10

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

# Create backward hook to get gradients
gradientsSTE1 = []
gradientsSTE2 = []
gradientsSTE3 = []


def backward_hook_ste1(module, grad_input, grad_output):
    gradientsSTE1.append(grad_input[0].cpu().detach().numpy()[0])


def backward_hook_ste2(module, grad_input, grad_output):
    gradientsSTE2.append(grad_input[0].cpu().detach().numpy()[0])


def backward_hook_ste3(module, grad_input, grad_output):
    gradientsSTE3.append(grad_input[0].cpu().detach().numpy()[0])


# Create forward hook to get values per neuron
valueSTE1 = []
valueSTE2 = []
valueSTE3 = []


def forward_hook_ste1(module, val_input, val_output):
    valueSTE1.append(val_output[0].cpu().detach().numpy())


def forward_hook_ste2(module, val_input, val_output):
    valueSTE2.append(val_output[0].cpu().detach().numpy())


def forward_hook_ste3(module, val_input, val_output):
    valueSTE3.append(val_output[0].cpu().detach().numpy())


# Register hooks

binaryNNModel.ste1.register_full_backward_hook(backward_hook_ste1)
binaryNNModel.ste2.register_full_backward_hook(backward_hook_ste2)
binaryNNModel.ste3.register_full_backward_hook(backward_hook_ste3)

binaryNNModel.ste1.register_forward_hook(forward_hook_ste1)
binaryNNModel.ste2.register_forward_hook(forward_hook_ste2)
binaryNNModel.ste3.register_forward_hook(forward_hook_ste3)

# Input samples and get gradients and values in each neuron

binaryNNModel.eval()

sampleSize = int(0.1 * len(training_data.data))

for i in range(sampleSize):
    X = training_data.data[i]
    y = training_data.targets[i]
    x = torch.reshape(Variable(X).type(torch.FloatTensor), (1, 28, 28))
    binaryNNModel.zero_grad()
    pred = binaryNNModel(x)
    pred[0, y.item()].backward()

    if (i+1) % 1000 == 0:
        print(f"[{i+1:>5d}/{sampleSize:>5d}]")

# Compute importance

gradientsSTE1 = np.array(gradientsSTE1).squeeze().reshape(len(gradientsSTE1), neuronPerLayer)
gradientsSTE2 = np.array(gradientsSTE2).squeeze().reshape(len(gradientsSTE2), neuronPerLayer)
gradientsSTE3 = np.array(gradientsSTE3).squeeze().reshape(len(gradientsSTE3), neuronPerLayer)

valueSTE1 = np.array(valueSTE1).squeeze().reshape(len(valueSTE1), neuronPerLayer)
valueSTE2 = np.array(valueSTE2).squeeze().reshape(len(valueSTE2), neuronPerLayer)
valueSTE3 = np.array(valueSTE3).squeeze().reshape(len(valueSTE3), neuronPerLayer)

importanceSTE1 = abs(np.multiply(gradientsSTE1, valueSTE1))
importanceSTE2 = abs(np.multiply(gradientsSTE2, valueSTE2))
importanceSTE3 = abs(np.multiply(gradientsSTE3, valueSTE3))

# Give each neuron its importance values
for neuron in neurons:
    layer = neuron.nLayer
    nNeuron = neuron.nNeuron

    if layer == 1:
        neuron.giveImportance(importanceSTE1[:, nNeuron], training_data.targets.tolist())
    elif layer == 2:
        neuron.giveImportance(importanceSTE2[:, nNeuron], training_data.targets.tolist())
    elif layer == 3:
        neuron.giveImportance(importanceSTE3[:, nNeuron], training_data.targets.tolist())

# Plot ordered importance of neurons per layer
fig = make_subplots(rows=1, cols=3,
                    subplot_titles=('Layer 1', 'Layer 2', 'Layer 3'))

neuronsL1 = BinaryOutputNeuron.neuronsPerLayer(neurons, 1)
neuronsL2 = BinaryOutputNeuron.neuronsPerLayer(neurons, 2)
neuronsL3 = BinaryOutputNeuron.neuronsPerLayer(neurons, 3)

fig.add_trace(
    go.Bar(x=np.arange(len(neuronsL1)), y=BinaryOutputNeuron.listImportance(neuronsL1)),
    row=1, col=1
)

fig.add_trace(
    go.Bar(x=np.arange(len(neuronsL2)), y=BinaryOutputNeuron.listImportance(neuronsL2)),
    row=1, col=2
)

fig.add_trace(
    go.Bar(x=np.arange(len(neuronsL3)), y=BinaryOutputNeuron.listImportance(neuronsL3)),
    row=1, col=3
)

fig.show()



