import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from modelsCommon.auxTransformations import ToBlackAndWhite, ToSign
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize, RandomHorizontalFlip, RandomCrop, Resize
from torch.utils.data import DataLoader
import os

modelName = f'binaryVggSmall/binaryVGGSmall_prunedBT10_4'
resizeFactor = 4
nLayers = 3

'''
Importing MNIST dataset
'''
print(f'IMPORT DATASET\n')

training_data = datasets.CIFAR10(root='data', train=True, transform=Compose([
	ToTensor(),
	Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
 	Resize(resizeFactor*32, antialias=False)]),
								 download=False)

# Load importance
importanceList = {}
for i in range(nLayers):
    importanceList[i] = pd.read_csv(f'data/importance/{modelName}/PerEntrylayer{i}.csv').to_numpy()

# Intialize containers of importance per class
print(f'INITIALIZE IMPORTANCE PER CLASS\n')
importancePerClass = {}
for iImp in range(len(importanceList)):
    importancePerClass[iImp] = {}
    for i in range(10):
        importancePerClass[iImp][i] = []
        
    
# Assign importance per class
print(f'ASSIGN IMPORTANCE PER CLASS\n')
for iImp in range(len(importanceList)):
    for i in range(len(importanceList[iImp])):
        importancePerClass[iImp][training_data.targets[i]].append(importanceList[iImp][i, :])

# From list to numpy array
print(f'FROM LIST TO NUMPY ARRAY\n')
for iImp in range(len(importanceList)):
    for i in range(10):
        importancePerClass[iImp][i] = np.array(importancePerClass[iImp][i])
        
# Save importance per class
print(f'CLASS-IMPORTANCE SCORE CALCULATION\n')
for iImp in range(len(importanceList)):
    nEntries = 0
    totalEntries = 0
    for i in range(10):
        totalEntries += importancePerClass[iImp][i].size
        aux = importancePerClass[iImp][i].sum(0) / len(importancePerClass[iImp][i])
        nEntries += len(importancePerClass[iImp][i]) * (aux > 0).sum()
        importancePerClass[iImp][i] = aux
    print(f'importance number {iImp} has {nEntries} with relevant classes out of {totalEntries}')

# Group all importances in same array
for iImp in range(len(importanceList)):
    importancePerClass[iImp] = np.row_stack(tuple(importancePerClass[iImp].values()))

# Create folder if not exists
if not os.path.exists(f'img/importance/{modelName}'):
    os.makedirs(f'img/importance/{modelName}')

# Print results
for imp in importancePerClass:
    # Print aggregated importance
    aux = importancePerClass[imp].sum(0)
    aux.sort()

    fig = plt.figure()
    plt.hist(aux, bins=100, range=(0, 10), color='b')
    plt.xlabel('Neuron importance score')
    plt.ylabel('Number of neurons')
    plt.title(f'Layer {imp} Importance Score')
    fig.savefig(f'img/importance/{modelName}/accImportanceLayer{imp}.png', transparent=True)

    # Print classes that are important
    aux = (importancePerClass[imp] > 0).sum(0)
    aux.sort()

    fig = plt.figure()
    plt.hist(aux, bins=100, range=(0, 10), color='b')
    plt.xlabel('Number of important classes')
    plt.ylabel('Number of neurons')
    plt.title(f'Layer {imp} Number of Important Classes')
    fig.savefig(f'img/importance/{modelName}/importanClassesLayer{imp}.png', transparent=True)