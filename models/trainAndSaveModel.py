import random
import pandas as pd
import torch
from models.auxFunctions import trainAndTest, ToBlackAndWhite
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader
from models.binaryNN import BinaryNeuralNetwork
from models.fpNN import FPNeuralNetwork
import torch.optim as optim
import torch.nn as nn

batch_size = 64
neuronPerLayer = 100
epochs = 100
criterionName = 'nll'
# criterionName = 'cel'
precision = 'bin'
# precision = 'full'

# Check mps maybe if working in MacOS
device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
Importing MNIST dataset
'''
print(f'IMPORT DATASET\n')

training_data = datasets.MNIST(
    root='C:/Users/carlo/OneDrive/Documentos/Universidad/MUIT/Segundo/TFM/Code/data',
    train=True,
    download=False,
    transform=Compose([
            ToTensor(),
            ToBlackAndWhite()
        ])
)

test_data = datasets.MNIST(
    root='C:/Users/carlo/OneDrive/Documentos/Universidad/MUIT/Segundo/TFM/Code/data',
    train=False,
    download=False,
    transform=Compose([
        ToTensor(),
        ToBlackAndWhite()
    ])
)

'''
Create DataLoader
'''
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

'''
Instantiate NN models
'''
print(f'MODEL INSTANTIATION\n')

if precision == 'full':
    model = FPNeuralNetwork(neuronPerLayer).to(device)
elif precision == 'bin':
    model = BinaryNeuralNetwork(neuronPerLayer).to(device)

'''
Train and test
'''
print(f'TRAINING\n')

opt = optim.Adamax(model.parameters(), lr=3e-3, weight_decay=1e-6)

if criterionName == 'nll':
    criterion = nn.NLLLoss()
elif criterionName == 'cel':
    criterion = nn.CrossEntropyLoss()

trainAndTest(epochs, train_dataloader, test_dataloader, model, opt, criterion)

'''
Save
'''

torch.save(model.state_dict(), f'savedModels/MNIST{precision}NN{epochs}Epoch{neuronPerLayer}NPL{criterionName}Criterion')


