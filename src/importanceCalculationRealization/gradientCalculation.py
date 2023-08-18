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

    # ! not even neccessary to load the model

    # Load activations
    model.loadActivations(f'./data/activations/{modelName}')

    # Create TT per layer (not optimized)
    for i in range(1, 4):
        if not os.path.exists(f'./data/plas/{modelName}/ABC/layer{i}/'):
            os.makedirs(f'./data/plas/{modelName}/ABC/layer{i}/')
        if not os.path.exists(f'./data/plas/{modelName}/ESPRESSO/layer{i}/'):
            os.makedirs(f'./data/plas/{modelName}/ESPRESSO/layer{i}/')

    # Load pruning data
    dfPrunedLayer = pd.read_csv(f'data/savedModels/{modelName}_prunedInfo0.csv')
    print(f'data/savedModels/{modelName}_prunedInfo0.csv read')

    # Layer 1
    columnsTags = [f'IN{i}' for i in range(model.valueSTE42.shape[1])]
    df = pd.DataFrame(model.valueSTE42, columns=columnsTags)
    df[df == -1] = 0
    df = df.astype(int)
    for neuron in range(model.valueSTEL0.shape[1]):
        df[f'OUT{neuron:04d}'] = model.valueSTEL0[:, neuron]
        df.replace({f'OUT{neuron:04d}': -1}, 0, inplace=True)

        aux = pruneAndDrop(df, dfPrunedLayer, f'N{neuron:04d}')

        createPLAFileABC(aux, f'./data/plas/{modelName}/ABC/layer1/N{neuron:04d}')
        createPLAFileEspresso(aux, f'./data/plas/{modelName}/ESPRESSO/layer1/N{neuron:04d}')
        df.drop(f'OUT{neuron:04d}', inplace=True, axis=1)

    # Load pruning data
    dfPrunedLayer = pd.read_csv(f'data/savedModels/{modelName}_prunedInfo1.csv')
    print(f'data/savedModels/{modelName}_prunedInfo1.csv read')

    # Layer 2
    columnsTags = [f'IN{i}' for i in range(model.valueSTEL0.shape[1])]
    df = pd.DataFrame(model.valueSTEL0, columns=columnsTags)
    df[df == -1] = 0
    df = df.astype(int)
    for neuron in range(model.valueSTEL1.shape[1]):
        df[f'OUT{neuron:04d}'] = model.valueSTEL1[:, neuron]
        df.replace({f'OUT{neuron:04d}': -1}, 0, inplace=True)
        aux = df.copy()

        aux = pruneAndDrop(df, dfPrunedLayer, f'N{neuron:04d}')

        createPLAFileABC(aux, f'./data/plas/{modelName}/ABC/layer2/N{neuron:04d}')
        createPLAFileEspresso(aux, f'./data/plas/{modelName}/ESPRESSO/layer2/N{neuron:04d}')
        df.drop(f'OUT{neuron:04d}', inplace=True, axis=1)

    # Load pruning data

    # Layer 3
    columnsTags = [f'IN{i}' for i in range(model.valueSTEL1.shape[1])]
    df = pd.DataFrame(model.valueSTEL1, columns=columnsTags)
    df[df == -1] = 0
    df = df.astype(int)
    for neuron in range(model.valueSTEL2.shape[1]):
        df[f'OUT{neuron:04d}'] = model.valueSTEL2[:, neuron]
        df.replace({f'OUT{neuron:04d}': -1}, 0, inplace=True)
        aux = df.copy()

        aux = pruneAndDrop(df, dfPrunedLayer, f'N{neuron:04d}')

        createPLAFileABC(aux, f'./data/plas/{modelName}/ABC/layer3/N{neuron:04d}')
        createPLAFileEspresso(aux, f'./data/plas/{modelName}/ESPRESSO/layer3/N{neuron:04d}')
        df.drop(f'OUT{neuron:04d}', inplace=True, axis=1)

    ###########################################################################################3
    # Create TT per layer (optimized per class)
    for i in range(1, 4):
        if not os.path.exists(f'./data/plas/{modelName}/ABCOptimizedPerClass/layer{i}/'):
            os.makedirs(f'./data/plas/{modelName}/ABCOptimizedPerClass/layer{i}/')

        if not os.path.exists(f'./data/plas/{modelName}/ESPRESSOOptimizedPerClass_0/layer{i}/'):
            os.makedirs(f'./data/plas/{modelName}/ESPRESSOOptimizedPerClass_0/layer{i}/')
        if not os.path.exists(f'./data/plas/{modelName}/ESPRESSOOptimizedPerClass_1/layer{i}/'):
            os.makedirs(f'./data/plas/{modelName}/ESPRESSOOptimizedPerClass_1/layer{i}/')
        if not os.path.exists(f'./data/plas/{modelName}/ESPRESSOOptimizedPerClass_2/layer{i}/'):
            os.makedirs(f'./data/plas/{modelName}/ESPRESSOOptimizedPerClass_2/layer{i}/')

    # Load pruning data
    dfPrunedLayer = pd.read_csv(f'data/savedModels/{modelName}_prunedInfo0.csv')
    print(f'data/savedModels/{modelName}_prunedInfo0.csv read')

    # Layer 1
    columnsTags = [f'IN{i}' for i in range(model.valueSTE42.shape[1])]
    df = pd.DataFrame(model.valueSTE42, columns=columnsTags)
    importancePerEntry = pd.read_csv(f'./data/importance/{modelName}/PerClasslayer0.csv')
    df[df == -1] = 0
    df = df.astype(int)
    for neuron in range(model.valueSTEL0.shape[1]):
        dfImportance = importancePerEntry[f'N{neuron}']    
        df[f'OUT{neuron:04d}'] = model.valueSTEL0[:, neuron]
        df.replace({f'OUT{neuron:04d}': -1}, 0, inplace=True)
        aux = df.copy()

        aux = aux[dfImportance > 0]
        aux.reset_index(drop=True, inplace=True)

        aux = pruneAndDrop(df, dfPrunedLayer, f'N{neuron:04d}')
        
        createPLAFileEspresso(aux, f'./data/plas/{modelName}/ESPRESSOOptimizedPerClass_0/layer1/N{neuron:04d}', conflictMode=0)
        createPLAFileEspresso(aux, f'./data/plas/{modelName}/ESPRESSOOptimizedPerClass_1/layer1/N{neuron:04d}', conflictMode=1)
        createPLAFileEspresso(aux, f'./data/plas/{modelName}/ESPRESSOOptimizedPerClass_2/layer1/N{neuron:04d}', conflictMode=2)
        df.drop(f'OUT{neuron:04d}', inplace=True, axis=1)

    # Load pruning data
    dfPrunedLayer = pd.read_csv(f'data/savedModels/{modelName}_prunedInfo1.csv')
    print(f'data/savedModels/{modelName}_prunedInfo1.csv read')

    # Layer 2
    columnsTags = [f'IN{i}' for i in range(model.valueSTEL0.shape[1])]
    df = pd.DataFrame(model.valueSTEL0, columns=columnsTags)
    importancePerEntryPre = pd.read_csv(f'./data/importance/{modelName}/PerClasslayer0.csv', dtype=int)
    importancePerEntry = pd.read_csv(f'./data/importance/{modelName}/PerClasslayer1.csv')
    df = pd.DataFrame(np.where(importancePerEntryPre == 0, 2, df), columns=columnsTags)
    df[df == -1] = 0
    df = df.astype(int)
    for neuron in range(model.valueSTEL1.shape[1]):
        dfImportance = importancePerEntry[f'N{neuron}']    
        df[f'OUT{neuron:04d}'] = model.valueSTEL1[:, neuron]
        df.replace({f'OUT{neuron:04d}': -1}, 0, inplace=True)
        aux = df.copy()

        aux = aux[dfImportance > 0]
        aux.reset_index(drop=True, inplace=True)

        aux = pruneAndDrop(df, dfPrunedLayer, f'N{neuron:04d}')
        
        createPLAFileEspresso(aux, f'./data/plas/{modelName}/ESPRESSOOptimizedPerClass_0/layer2/N{neuron:04d}', conflictMode=0)
        createPLAFileEspresso(aux, f'./data/plas/{modelName}/ESPRESSOOptimizedPerClass_1/layer2/N{neuron:04d}', conflictMode=1)
        createPLAFileEspresso(aux, f'./data/plas/{modelName}/ESPRESSOOptimizedPerClass_2/layer2/N{neuron:04d}', conflictMode=2)
        df.drop(f'OUT{neuron:04d}', inplace=True, axis=1)

    # Load pruning data
    dfPrunedLayer = pd.read_csv(f'data/savedModels/{modelName}_prunedInfo3.csv')
    print(f'data/savedModels/{modelName}_prunedInfo3.csv read')

    # Layer 3
    columnsTags = [f'IN{i}' for i in range(model.valueSTEL1.shape[1])]
    df = pd.DataFrame(model.valueSTEL1, columns=columnsTags)
    importancePerEntryPre = pd.read_csv(f'./data/importance/{modelName}/PerClasslayer1.csv', dtype=int)
    importancePerEntry = pd.read_csv(f'./data/importance/{modelName}/PerClasslayer2.csv')
    df = pd.DataFrame(np.where(importancePerEntryPre == 0, 2, df), columns=columnsTags)
    df[df == -1] = 0
    df = df.astype(int)
    for neuron in range(model.valueSTEL2.shape[1]):
        dfImportance = importancePerEntry[f'N{neuron}']    
        df[f'OUT{neuron:04d}'] = model.valueSTEL2[:, neuron]
        df.replace({f'OUT{neuron:04d}': -1}, 0, inplace=True)
        aux = df.copy()

        aux = aux[dfImportance > 0]
        aux.reset_index(drop=True, inplace=True)

        aux = pruneAndDrop(df, dfPrunedLayer, f'N{neuron:04d}')
        
        createPLAFileEspresso(aux, f'./data/plas/{modelName}/ESPRESSOOptimizedPerClass_0/layer3/N{neuron:04d}', conflictMode=0)
        createPLAFileEspresso(aux, f'./data/plas/{modelName}/ESPRESSOOptimizedPerClass_1/layer3/N{neuron:04d}', conflictMode=1)
        createPLAFileEspresso(aux, f'./data/plas/{modelName}/ESPRESSOOptimizedPerClass_2/layer3/N{neuron:04d}', conflictMode=2)
        df.drop(f'OUT{neuron:04d}', inplace=True, axis=1)
