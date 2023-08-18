import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from modelsCommon.auxTransformations import ToBlackAndWhite, ToSign
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader
import os

nLayers = 3
neuronPerLayer = 100
for modelName in ['eeb/eeb_prunedBT6_100ep_100npl', 'eeb/eeb_prunedBT8_100ep_100npl', 'eeb/eeb_prunedBT10_100ep_100npl', 'eeb/eeb_prunedBT12_100ep_100npl']:

    
    print(f'{modelName}')

    training_data = datasets.MNIST(
        root='./data',
        train=True,
        download=False,
        transform=Compose([
                ToTensor(),
                ToBlackAndWhite(),
                ToSign()
            ])
    )

    # Load importance
    importanceList = {}
    for i in range(nLayers):
        importanceList[i] = pd.read_csv(f'data/importance/{modelName}/PerEntrylayer{i}.csv').to_numpy()
        print(f'Per entry approach importance number {i}. Entries spared {importanceList[i].sum().sum()} ({importanceList[i].sum().sum() / importanceList[i].size * 100}%) of total {importanceList[i].size}')

    # Intialize containers of importance per class
    importancePerClass = {}
    for iImp in range(len(importanceList)):
        importancePerClass[iImp] = {}
        for i in range(10):
            importancePerClass[iImp][i] = []
            
        
    # Assign importance per class
    for iImp in range(len(importanceList)):
        for i in range(len(importanceList[iImp])):
            importancePerClass[iImp][training_data.targets[i].item()].append(importanceList[iImp][i, :])

    # From list to numpy array
    for iImp in range(len(importanceList)):
        for i in range(10):
            importancePerClass[iImp][i] = np.array(importancePerClass[iImp][i])
            
    # Save importance per class
    for iImp in range(len(importanceList)):
        nEntries = 0
        totalEntries = 0
        for i in range(10):
            totalEntries += importancePerClass[iImp][i].size
            aux = importancePerClass[iImp][i].sum(0) / len(importancePerClass[iImp][i])
            nEntries += len(importancePerClass[iImp][i]) * (aux > 0).sum()
            importancePerClass[iImp][i] = aux
        print(f'Per Class approach importance number {iImp} has {nEntries} ({nEntries / totalEntries * 100}%) with relevant classes out of {totalEntries}')

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
        plt.close(fig)

        # Print classes that are important
        aux = (importancePerClass[imp] > 0).sum(0)
        aux.sort()

        fig = plt.figure()
        plt.hist(aux, bins=100, range=(0, 10), color='b')
        plt.xlabel('Number of important classes')
        plt.ylabel('Number of neurons')
        plt.title(f'Layer {imp} Number of Important Classes')
        fig.savefig(f'img/importance/{modelName}/importanClassesLayer{imp}.png', transparent=True)
        plt.close(fig)
