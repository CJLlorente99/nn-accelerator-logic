import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize, RandomHorizontalFlip, RandomCrop, Resize
from torch.utils.data import DataLoader
from modules.binaryVggSmall import binaryVGGSmall
from modules.binaryVggVerySmall import binaryVGGVerySmall
import torch.nn as nn
import pandas as pd
import numpy as np
from modelsCommon.auxTransformations import *
import torch.nn.functional as F
import os

batch_size = 1
modelName = f'binaryVggSmall/binaryVGGSmall_prunedBT6_4'
modelFilename = f'data\savedModels\{modelName}'
prunedBT = True
resizeFactor = 4

# Check mps maybe if working in MacOS
device = 'cuda' if torch.cuda.is_available() else 'cpu'

outputFilenameTrain = f'data/inputs/{modelName}/trainlayer42'
outputFilenameTest = f'data/inputs/{modelName}/testlayer42'

'''
Create folder if it doesn't exist
'''
if not os.path.exists(f'data/inputs/{modelName}'):
    os.makedirs(f'data/inputs/{modelName}')

'''
Importing MNIST dataset
'''

print(f'DOWNLOAD DATASET\n')
training_data = datasets.CIFAR10(root='data', train=True, transform=Compose([
        RandomHorizontalFlip(),
        RandomCrop(32, 4),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        Resize(resizeFactor*32, antialias=False)]),
    download=False)

test_data = datasets.CIFAR10(root='data', train=False, transform=Compose([
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        Resize(resizeFactor*32, antialias=False)]),
    download=False)

'''
Create DataLoader
'''
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

'''
Instantiate NN models
'''
print(f'MODEL INSTANTIATION\n')

connectionsToPrune = 0
if prunedBT:
	connectionsToPrune = 1

if modelName.startswith('binaryVggVerySmall'):
    relus = [1, 1, 1, 1, 0, 0, 0, 0]
    model = binaryVGGVerySmall(resizeFactor=resizeFactor, relus=relus, connectionsAfterPrune=connectionsToPrune)
elif modelName.startswith('binaryVggSmall'):
    relus = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    model = binaryVGGSmall(resizeFactor=resizeFactor, relus=relus, connectionsAfterPrune=connectionsToPrune)
        
model.load_state_dict(torch.load(modelFilename, map_location=torch.device(device)))

'''
Load the simulated inputs to the last layer (provided by minimized network)
'''

model.eval()

aux = []
count = 0
for X, y in train_dataloader:
    predL0 = model.forwardOneLayer(X , 0)
    x = predL0.detach().cpu().numpy()
    x[x < 0] = 0
    aux.append(x.astype(int)[0])
    count += 1
    if count % 5000 == 0:
        print(f'Training forward [{count:>5d}/{len(training_data.targets):>5d}]')

columns = [f'N{i:04d}' for i in range(8192)]
df = pd.DataFrame(np.array(aux), columns=columns)
df.to_csv(f'{outputFilenameTrain}.csv')
print(f'{outputFilenameTrain} created successfully')

aux = []
count = 0
for X, y in test_dataloader:
    predL0 = model.forwardOneLayer(X , 0)
    x = predL0.detach().cpu().numpy()
    x[x < 0] = 0
    aux.append(x.astype(int)[0])
    count += 1
    if count % 5000 == 0:
        print(f'Test forward [{count:>5d}/{len(test_data.targets):>5d}]')

columns = [f'N{i:04d}' for i in range(8192)]
df = pd.DataFrame(np.array(aux), columns=columns)
df.to_csv(f'{outputFilenameTest}.csv')
print(f'{outputFilenameTest} created successfully')
