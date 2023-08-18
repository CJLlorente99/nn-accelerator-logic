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
threshold = 10e-5

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

    # Compute importance
    importanceList = model.computeImportance()
    del model.gradientsSTEL0
    del model.gradientsSTEL1
    del model.gradientsSTEL2

     # Function to get duplicates
    def getIdxDuplicates(arr):
        vals, inverse, count = np.unique(arr, axis=0, return_inverse=True,
                                        return_counts=True)

        idx_vals_repeated = np.where(count > 1)[0]

        rows, cols = np.where(inverse == idx_vals_repeated[:, np.newaxis])
        _, inverse_rows = np.unique(rows, return_index=True)
        res = np.split(cols, inverse_rows[1:])
        return res

    # Get info about the activations
    dupList = []
    dupList.append(getIdxDuplicates(model.valueSTEL0, model))
    uniqueSTEL0 = np.unique(model.valueSTEL0, axis=0)
    print(f'Original length {model.valueSTEL0.shape[0]}, only unique length {uniqueSTEL0.shape[0]}, number of sample {sampleSize}')
    dupList.append(getIdxDuplicates(model.valueSTEL1))
    uniqueSTEL1 = np.unique(model.valueSTEL1, axis=0)
    print(f'Original length {model.valueSTEL1.shape[0]}, only unique length {uniqueSTEL1.shape[0]}, number of sample {sampleSize}')
    dupList.append(getIdxDuplicates(model.valueSTEL2))
    uniqueSTEL2 = np.unique(model.valueSTEL2, axis=0)
    print(f'Original length {model.valueSTEL2.shape[0]}, only unique length {uniqueSTEL2.shape[0]}, number of sample {sampleSize}')

    # Activations values no longer needed
    del model.valueSTE42
    del model.valueSTEL0
    del model.valueSTEL1
    del model.valueSTEL2

    # Apply threshold
    print(f'APPLY THRESHOLD\n')
    for iImp in range(len(importanceList)):
        importanceList[iImp] = importanceList[iImp] > threshold
        # Save importance for minimization per entry
        columnsTags = [f'N{i}' for i in range(importanceList[iImp].shape[1])]
        df = pd.DataFrame(importanceList[iImp], columns=columnsTags).astype(int)
        if not os.path.exists(f'data/importance/{modelName}/'):
            os.makedirs(f'data/importance/{modelName}/')
        df.to_csv(f'data/importance/{modelName}/PerEntrylayer{iImp}.csv', index=False)
        print(f'File data/importance/{modelName}/PerEntrylayer{iImp}.csv created')
        for dup in dupList[iImp]:
            if len(dup) != 0:
                importanceList[iImp][dup[0], :] = np.sum(importanceList[iImp][dup, :], axis=0)
        print(f'importance number {iImp} has shape {importanceList[iImp].shape}')
        print(f'importance number {iImp} has {importanceList[iImp].sum().sum()} ({(importanceList[iImp].sum().sum() / importanceList[iImp].size * 100):0.2f}%) entries above threshold {threshold} out of {importanceList[iImp].size}')
        
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
        for i in range(sampleSize):
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
        print(f'importance number {iImp} has {nEntries} ({(nEntries / totalEntries * 100):.2f}%) with relevant classes out of {totalEntries}')

    # Save class-based importance for minimization per class
    for iImp in range(len(importanceList)):
        dict_list = []
        for i in range(sampleSize):
            data = importancePerClass[iImp][training_data.targets[i]] > 0
            dict_data = {f'N{i}': data[i] for i in range(importanceList[iImp].shape[1])}
            dict_list.append(dict_data)
            if (i+1) % 500 == 0:
                print(f"Layer {iImp} entry {i+1:>5d}/{sampleSize:>5d}")

        df = pd.DataFrame.from_dict(dict_list)
        df = df.astype(int)
        df.to_csv(f'data/importance/{modelName}/PerClasslayer{iImp}.csv', index=False)
        print(f'File data/importance/{modelName}/PerClasslayer{iImp}.csv created')