import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize, RandomHorizontalFlip, RandomCrop, Resize
from torch.utils.data import DataLoader
from modules.binaryVggVerySmall import binaryVGGVerySmall
from modules.binaryVggSmall import binaryVGGSmall
from ttUtilities.isfRealization import *
import numpy as np
import pandas as pd
import os

# Check mps maybe if working in MacOS
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Global parameters
prunedBT = True
perGradientSampling = 1
resizeFactor = 4

# for modelName in ['binaryVggSmall/binaryVGGSmall_prunedBT6_4', 'binaryVggSmall/binaryVGGSmall_prunedBT8_4',
#                   'binaryVggSmall/binaryVGGSmall_prunedBT10_4', 'binaryVggSmall/binaryVGGSmall_prunedBT12_4']:
    
for modelName in ['binaryVggVerySmall/binaryVGGVerySmall_prunedBT6_4', 'binaryVggVerySmall/binaryVGGVerySmall_prunedBT8_4',
                  'binaryVggVerySmall/binaryVGGVerySmall_prunedBT10_4', 'binaryVggVerySmall/binaryVGGVerySmall_prunedBT12_4']:

    modelFilename = f'data/savedModels/{modelName}'

    '''
    Importing CIFAR10 dataset
    '''
    print(f'DOWNLOAD DATASET\n')
    training_data = datasets.CIFAR10(root='data', train=True, transform=Compose([
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        Resize(resizeFactor*32, antialias=False)]),
                                    download=False)
    
    train_dataloader = DataLoader(training_data, 1)
    sampleSize = int(perGradientSampling * len(training_data.data)) 

    '''
    Load model
    '''
    connectionsToPrune = 0
    if prunedBT:
        connectionsToPrune = 1

    if modelName.startswith('binaryVggVerySmall'):
        relus = [1, 1, 1, 1, 0, 0, 0, 0]
        model = binaryVGGVerySmall(resizeFactor=resizeFactor, relus=relus, connectionsAfterPrune=connectionsToPrune)
    elif modelName.startswith('binaryVggSmall'):
        relus = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
        model = binaryVGGSmall(resizeFactor=resizeFactor, relus=relus, connectionsAfterPrune=connectionsToPrune)

    model.load_state_dict(torch.load(modelFilename, map_location=device))

    """
    Enumeration for realization and importance calculation
    """

    model.registerHooks()
    model.eval()

    start = 0
    for i in range(start, sampleSize):
        X, y = train_dataloader.dataset[i]
        model.zero_grad()
        pred = model(X[None, :, :, :])
        pred[0, y].backward()

        if (i+1) % 500 == 0:
            print(f'Get Gradients and Activation Values [{i + 1 :>5d}/{sampleSize :>5d}]')

    # Folder for gradients and activations
    if not os.path.exists(f'./data/activation/{modelName}'):
        os.makedirs(f'./data/activation/{modelName}')
    if not os.path.exists(f'./data/gradients/{modelName}'):
        os.makedirs(f'./data/gradients/{modelName}')

    model.listToArray()
    model.saveActivations()
    model.saveGradients()

    # Free memory (only gradients needed)
    del model.valueSTE42
    del model.valueSTEL0
    del model.valueSTEL1
    del model.valueSTEL2

    # Compute importance
    importanceList = model.computeImportance()
    del model.gradientsSTEL0
    del model.gradientsSTEL1
    del model.gradientsSTEL2