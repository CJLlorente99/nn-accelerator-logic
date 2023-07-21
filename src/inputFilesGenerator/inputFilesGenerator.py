import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader
from modules.binaryEnergyEfficiency import BinaryNeuralNetwork
import torch.nn as nn
import pandas as pd
import numpy as np
from modelsCommon.auxTransformations import *
import torch.nn.functional as F

batch_size = 1
neuronPerLayer = 100
modelFilename = f'data\savedModels\eeb_pruned_100ep_100npl'
outputFilenameTrain = f'data/inputs/trainlayer1'
outputFilenameTest = f'data/inputs/testlayer1'


# Check mps maybe if working in MacOS
device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
Importing MNIST dataset
'''
print(f'IMPORT DATASET\n')

training_data = datasets.MNIST(
    root='data',
    train=True,
    download=False,
    transform=Compose([
            ToTensor(),
            ToBlackAndWhite(),
            ToSign()
        ])
    )

test_data = datasets.MNIST(
    root='data',
    train=False,
    download=False,
    transform=Compose([
            ToTensor(),
            ToBlackAndWhite(),
            ToSign()
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

model = BinaryNeuralNetwork(neuronPerLayer).to(device)
model.load_state_dict(torch.load(modelFilename, map_location=torch.device(device)))

'''
Load the simulated inputs to the last layer (provided by minimized network)
'''

model.eval()
with open(outputFilenameTrain, 'w') as f:
    for X, y in train_dataloader:
        X = torch.flatten(X, start_dim=1)
        predL0 = model.forwardOneLayer(X , 0)
        x = predL0.detach().cpu().numpy()
        x[x < 0] = 0
        x = list(x.astype(int)[0])
        f.write(''.join(str(e) for e in x))
        f.write('\n')

with open(outputFilenameTest, 'w') as f:
    for X, y in test_dataloader:
        X = torch.flatten(X, start_dim=1)
        predL0 = model.forwardOneLayer(X , 0)
        x = predL0.detach().cpu().numpy()
        x[x < 0] = 0
        x = list(x.astype(int)[0])
        f.write(''.join(str(e) for e in x))
        f.write('\n')