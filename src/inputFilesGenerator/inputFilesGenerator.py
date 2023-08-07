import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader
from modules.binaryBNN import BNNBinaryNeuralNetwork
import torch.nn as nn
import pandas as pd
import numpy as np
from modelsCommon.auxTransformations import *
import torch.nn.functional as F
import os

batch_size = 1
neuronPerLayer = 4096
modelName = 'bnn/bnn_prunedBT14_100ep_4096npl'
modelFilename = f'data\savedModels\{modelName}'
outputFilenameTrain = f'data/inputs/{modelName}/trainlayer1'
outputFilenameTest = f'data/inputs/{modelName}/testlayer1'
prunedBT = True


# Check mps maybe if working in MacOS
device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
Create folder if it doesn't exist
'''
if not os.path.exists(f'data/inputs/{modelName}'):
    os.makedirs(f'data/inputs/{modelName}')

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

if 'eeb' in modelFilename:
    model = BinaryNeuralNetwork(neuronPerLayer).to(device) if not prunedBT else BinaryNeuralNetwork(neuronPerLayer, 1).to(device)
elif 'bnn' in modelFilename:
    model = BNNBinaryNeuralNetwork(neuronPerLayer).to(device) if not prunedBT else BNNBinaryNeuralNetwork(neuronPerLayer, 1).to(device)
model.load_state_dict(torch.load(modelFilename, map_location=torch.device(device)))

'''
Load the simulated inputs to the last layer (provided by minimized network)
'''

model.eval()

aux = []
count = 0
for X, y in train_dataloader:
    X = torch.flatten(X, start_dim=1)
    predL0 = model.forwardOneLayer(X , 0)
    x = predL0.detach().cpu().numpy()
    x[x < 0] = 0
    aux.append(x.astype(int)[0])
    count += 1
    if count % 5000 == 0:
        print(f'Training forward [{count:>5d}/{len(training_data.targets):>5d}]')

columns = [f'N{i:04d}' for i in range(neuronPerLayer)]
df = pd.DataFrame(np.array(aux), columns=columns)
df.to_csv(f'{outputFilenameTrain}.csv')
print(f'{outputFilenameTrain} created successfully')

aux = []
count = 0
for X, y in test_dataloader:
    X = torch.flatten(X, start_dim=1)
    predL0 = model.forwardOneLayer(X , 0)
    x = predL0.detach().cpu().numpy()
    x[x < 0] = 0
    aux.append(x.astype(int)[0])
    count += 1
    if count % 5000 == 0:
        print(f'Test forward [{count:>5d}/{len(test_data.targets):>5d}]')

columns = [f'N{i:04d}' for i in range(neuronPerLayer)]
df = pd.DataFrame(np.array(aux), columns=columns)
df.to_csv(f'{outputFilenameTest}.csv')
print(f'{outputFilenameTest} created successfully')