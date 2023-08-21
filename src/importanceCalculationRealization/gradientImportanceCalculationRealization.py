import torch
from modelsCommon.auxTransformations import ToBlackAndWhite, ToSign
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader
from modules.binaryBNN import BNNBinaryNeuralNetwork
from ttUtilities.isfRealization import createPLAFileABC, createPLAFileEspresso, pruneAndDrop
import numpy as np
import pandas as pd
import os

# Check mps maybe if working in MacOS
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Global parameters
neuronPerLayer = 4096
threshold = 10e-5
perGradientSampling = 1  # Percentage of the training dataset to enumerate
prunedBT = True  # True if BT, False if AT

# Global parameters
# ! Do not change
batch_size = 1

for modelName in ['bnn/bnn_prunedBT12_100ep_4096npl']:
# for modelName in ['bnn/bnn_prunedBT10_100ep_4096npl', 'bnn/bnn_prunedBT12_100ep_4096npl']:

    print(f'{modelName}')
    modelFilename = f'data/savedModels/{modelName}'
    
    '''
    Importing MNIST dataset
    '''
    print(f'IMPORT DATASET\n')

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

    sampleSize = int(perGradientSampling * len(training_data.data))  # sample size to be used for importance calculation

    '''
    Create DataLoader
    '''
    train_dataloader = DataLoader(training_data, batch_size=batch_size)

    connectionsToPrune = 0
    if prunedBT:
        connectionsToPrune = 1
    model = BNNBinaryNeuralNetwork(neuronPerLayer, connectionsToPrune)
    model.load_state_dict(torch.load(modelFilename, map_location=torch.device(device)))

    '''
    Calculate importance per class per neuron
    '''

    # Input samples and get gradients and values in each neuron
    # print(f'GET GRADIENTS AND ACTIVATION VALUES\n')

    # model.registerHooks()
    # model.eval()

    # for i in range(sampleSize):
    #     X, y = train_dataloader.dataset[i]
    #     model.zero_grad()
    #     pred = model(X)
    #     pred[0, y].backward()

    #     if (i+1) % 500 == 0:
    #         print(f"Get Gradients and Activation Values [{i+1:>5d}/{sampleSize:>5d}]")

    # model.listToArray(neuronPerLayer)  # Hopefully improves memory usage

    # # Create folder to store the activations and gradients
    # if not os.path.exists(f'./data/activations/{modelName}/'):
    #     os.makedirs(f'./data/activations/{modelName}/')
    # if not os.path.exists(f'./data/gradients/{modelName}/'):
    #     os.makedirs(f'./data/gradients/{modelName}/')

    # # Save gradients and activations
    # model.saveActivations(f'./data/activations/{modelName}/')
    # model.saveGradients(f'./data/gradients/{modelName}/')

    # Function to get duplicates
    # def getIdxDuplicates(arr):
    #     vals, inverse, count = np.unique(arr, axis=0, return_inverse=True,
    #                                     return_counts=True)

    #     idx_vals_repeated = np.where(count > 1)[0]

    #     rows, cols = np.where(inverse == idx_vals_repeated[:, np.newaxis])
    #     _, inverse_rows = np.unique(rows, return_index=True)
    #     res = np.split(cols, inverse_rows[1:])
    #     return res
    
    # # Load activations to calculate dupllicated
    # model.loadActivations(f'./data/activations/{modelName}/')

    # # Get info about the activations
    # dupList = []
    # dupList.append(getIdxDuplicates(model.valueSTE0))
    # uniqueSTE0 = np.unique(model.valueSTE0, axis=0)
    # print(f'Original length {model.valueSTE0.shape[0]}, only unique length {uniqueSTE0.shape[0]}, number of sample {sampleSize}')
    # dupList.append(getIdxDuplicates(model.valueSTE1))
    # uniqueSTE1 = np.unique(model.valueSTE1, axis=0)
    # print(f'Original length {model.valueSTE1.shape[0]}, only unique length {uniqueSTE1.shape[0]}, number of sample {sampleSize}')
    # dupList.append(getIdxDuplicates(model.valueSTE2))
    # uniqueSTE2 = np.unique(model.valueSTE2, axis=0)
    # print(f'Original length {model.valueSTE2.shape[0]}, only unique length {uniqueSTE2.shape[0]}, number of sample {sampleSize}')

    # # Only gradients are needed to compute importance (free activations)
    # del model.valueSTE0 
    # del model.valueSTE1
    # del model.valueSTE2
    # del model.valueSTE3

    # # Load gradients and compute importance
    # model.loadGradients(f'./data/gradients/{modelName}/')
    # importanceList = model.computeImportance(neuronPerLayer)

    # del model.gradientsSTE0
    # del model.gradientsSTE1
    # del model.gradientsSTE2
    # del model.gradientsSTE3

    # # Apply threshold
    # print(f'APPLY THRESHOLD\n')
    # for iImp in range(len(importanceList)):
    #     importanceList[iImp] = importanceList[iImp] > threshold
    #     # Save importance for minimization per entry
    #     columnsTags = [f'N{i}' for i in range(importanceList[iImp].shape[1])]
    #     df = pd.DataFrame(importanceList[iImp], columns=columnsTags).astype(int)
    #     if not os.path.exists(f'data/importance/{modelName}/'):
    #         os.makedirs(f'data/importance/{modelName}/')
    #     df.to_csv(f'data/importance/{modelName}/PerEntrylayer{iImp}.csv', index=False)
    #     print(f'File data/importance/{modelName}/PerEntrylayer{iImp}.csv created')
    #     for dup in dupList[iImp]:
    #         if len(dup) != 0:
    #             importanceList[iImp][dup[0], :] = np.sum(importanceList[iImp][dup, :], axis=0)
    #     print(f'importance number {iImp} has shape {importanceList[iImp].shape}')
    #     print(f'importance number {iImp} has {importanceList[iImp].sum().sum()} ({(importanceList[iImp].sum().sum() / importanceList[iImp].size * 100):0.2f}%) entries above threshold {threshold} out of {importanceList[iImp].size}')
        
    # # Intialize containers of importance per class
    # print(f'INITIALIZE IMPORTANCE PER CLASS\n')
    # importancePerClass = {}
    # for iImp in range(len(importanceList)):
    #     importancePerClass[iImp] = {}
    #     for i in range(10):
    #         importancePerClass[iImp][i] = []
            
        
    # # Assign importance per class
    # print(f'ASSIGN IMPORTANCE PER CLASS\n')
    # for iImp in range(len(importanceList)):
    #     for i in range(sampleSize):
    #         importancePerClass[iImp][training_data.targets[i].item()].append(importanceList[iImp][i, :])

    # # From list to numpy array
    # print(f'FROM LIST TO NUMPY ARRAY\n')
    # for iImp in range(len(importanceList)):
    #     for i in range(10):
    #         importancePerClass[iImp][i] = np.array(importancePerClass[iImp][i])
            
    # # Save importance per class
    # print(f'CLASS-IMPORTANCE SCORE CALCULATION\n')
    # for iImp in range(len(importanceList)):
    #     nEntries = 0
    #     totalEntries = 0
    #     for i in range(10):
    #         totalEntries += importancePerClass[iImp][i].size
    #         aux = importancePerClass[iImp][i].sum(0) / len(importancePerClass[iImp][i])
    #         nEntries += len(importancePerClass[iImp][i]) * (aux > 0).sum()
    #         importancePerClass[iImp][i] = aux
    #     print(f'importance number {iImp} has {nEntries} ({(nEntries / totalEntries * 100):.2f}%) with relevant classes out of {totalEntries}')

    # # Save class-based importance for minimization per class
    # for iImp in range(len(importanceList)):
    #     dict_list = []
    #     for i in range(sampleSize):
    #         data = importancePerClass[iImp][training_data.targets[i].item()] > 0
    #         dict_data = {f'N{i}': data[i] for i in range(importanceList[iImp].shape[1])}
    #         dict_list.append(dict_data)
    #         if (i+1) % 500 == 0:
    #             print(f"Layer {iImp} entry {i+1:>5d}/{sampleSize:>5d}")

    #     df = pd.DataFrame.from_dict(dict_list)
    #     df = df.astype(int)
    #     df.to_csv(f'data/importance/{modelName}/PerClasslayer{iImp}.csv', index=False)
    #     print(f'File data/importance/{modelName}/PerClasslayer{iImp}.csv created')

    # # Get rid of everything that won't be used
    # del importanceList

    # Only activations are needed now
    model.loadActivations(f'./data/activations/{modelName}/')

    ################################################################################
    # Create TT per layer (not optimized)
    # for i in range(1, 4):
    #     if not os.path.exists(f'./data/plas/{modelName}/ABC/layer{i}/'):
    #         os.makedirs(f'./data/plas/{modelName}/ABC/layer{i}/')
    #     if not os.path.exists(f'./data/plas/{modelName}/ESPRESSO/layer{i}/'):
    #         os.makedirs(f'./data/plas/{modelName}/ESPRESSO/layer{i}/')

    # # Pruned info files
    # dfPrunedLayer = pd.read_csv(f'data/savedModels/{modelName}_prunedInfol1.csv')
    # print(f'data/savedModels/{modelName}_prunedInfol1.csv read')

    # # Layer 1
    # columnsTags = [f'IN{i}' for i in range(model.valueSTE0.shape[1])]
    # df = pd.DataFrame(model.valueSTE0, columns=columnsTags)
    # df[df == -1] = 0
    # df = df.astype(int)
    # for neuron in range(model.valueSTE1.shape[1]):
    #     df[f'OUT{neuron:04d}'] = model.valueSTE1[:, neuron]
    #     df.replace({f'OUT{neuron:04d}': -1}, 0, inplace=True)

    #     aux = pruneAndDrop(df, dfPrunedLayer, f'N{neuron:04d}')

    #     createPLAFileABC(aux, f'./data/plas/{modelName}/ABC/layer1/N{neuron:04d}')
    #     createPLAFileEspresso(aux, f'./data/plas/{modelName}/ESPRESSO/layer1/N{neuron:04d}')
    #     df.drop(f'OUT{neuron:04d}', inplace=True, axis=1)

    # # Pruned info files
    # dfPrunedLayer = pd.read_csv(f'data/savedModels/{modelName}_prunedInfol2.csv')
    # print(f'data/savedModels/{modelName}_prunedInfol2.csv read')

    # # Layer 2
    # columnsTags = [f'IN{i}' for i in range(model.valueSTE1.shape[1])]
    # df = pd.DataFrame(model.valueSTE1, columns=columnsTags)
    # df[df == -1] = 0
    # df = df.astype(int)
    # for neuron in range(model.valueSTE2.shape[1]):
    #     df[f'OUT{neuron:04d}'] = model.valueSTE2[:, neuron]
    #     df.replace({f'OUT{neuron:04d}': -1}, 0, inplace=True)
    #     aux = df.copy()

    #     aux = pruneAndDrop(df, dfPrunedLayer, f'N{neuron:04d}')

    #     createPLAFileABC(aux, f'./data/plas/{modelName}/ABC/layer2/N{neuron:04d}')
    #     createPLAFileEspresso(aux, f'./data/plas/{modelName}/ESPRESSO/layer2/N{neuron:04d}')
    #     df.drop(f'OUT{neuron:04d}', inplace=True, axis=1)

    # # Pruned info files
    # dfPrunedLayer = pd.read_csv(f'data/savedModels/{modelName}_prunedInfol3.csv')
    # print(f'data/savedModels/{modelName}_prunedInfol3.csv read')

    # # Layer 3
    # columnsTags = [f'IN{i}' for i in range(model.valueSTE2.shape[1])]
    # df = pd.DataFrame(model.valueSTE2, columns=columnsTags)
    # df[df == -1] = 0
    # df = df.astype(int)
    # for neuron in range(model.valueSTE3.shape[1]):
    #     df[f'OUT{neuron:04d}'] = model.valueSTE3[:, neuron]
    #     df.replace({f'OUT{neuron:04d}': -1}, 0, inplace=True)
    #     aux = df.copy()

    #     aux = pruneAndDrop(df, dfPrunedLayer, f'N{neuron:04d}')

    #     createPLAFileABC(aux, f'./data/plas/{modelName}/ABC/layer3/N{neuron:04d}')
    #     createPLAFileEspresso(aux, f'./data/plas/{modelName}/ESPRESSO/layer3/N{neuron:04d}')
    #     df.drop(f'OUT{neuron:04d}', inplace=True, axis=1)

    #################################################################################################
    # # Create PLAs per layer (optimized per class)
    for i in range(1, 4):
        if not os.path.exists(f'./data/plas/{modelName}/ESPRESSOOptimizedPerClass_0/layer{i}/'):
            os.makedirs(f'./data/plas/{modelName}/ESPRESSOOptimizedPerClass_0/layer{i}/')
        if not os.path.exists(f'./data/plas/{modelName}/ESPRESSOOptimizedPerClass_1/layer{i}/'):
            os.makedirs(f'./data/plas/{modelName}/ESPRESSOOptimizedPerClass_1/layer{i}/')
        if not os.path.exists(f'./data/plas/{modelName}/ESPRESSOOptimizedPerClass_2/layer{i}/'):
            os.makedirs(f'./data/plas/{modelName}/ESPRESSOOptimizedPerClass_2/layer{i}/')

    # Pruned info files
    # dfPrunedLayer = pd.read_csv(f'data/savedModels/{modelName}_prunedInfol1.csv')
    # print(f'data/savedModels/{modelName}_prunedInfol1.csv read')

    # # Layer 1
    # columnsTags = [f'IN{i}' for i in range(model.valueSTE0.shape[1])]
    # df = pd.DataFrame(model.valueSTE0, columns=columnsTags)
    # importancePerEntry = pd.read_csv(f'./data/importance/{modelName}/PerClasslayer0.csv')
    # df[df == -1] = 0
    # df = df.astype(int)
    # for neuron in range(model.valueSTE1.shape[1]):
    #     dfImportance = importancePerEntry[f'N{neuron}']    
    #     df[f'OUT{neuron:04d}'] = model.valueSTE1[:, neuron]
    #     df.replace({f'OUT{neuron:04d}': -1}, 0, inplace=True)
    #     aux = df.copy()

    #     aux = aux[dfImportance > 0]
    #     aux.reset_index(drop=True, inplace=True)

    #     aux = pruneAndDrop(df, dfPrunedLayer, f'N{neuron:04d}')
        
    #     createPLAFileEspresso(aux, f'./data/plas/{modelName}/ESPRESSOOptimizedPerClass_0/layer1/N{neuron:04d}', conflictMode=0)
    #     createPLAFileEspresso(aux, f'./data/plas/{modelName}/ESPRESSOOptimizedPerClass_1/layer1/N{neuron:04d}', conflictMode=1)
    #     createPLAFileEspresso(aux, f'./data/plas/{modelName}/ESPRESSOOptimizedPerClass_2/layer1/N{neuron:04d}', conflictMode=2)
    #     df.drop(f'OUT{neuron:04d}', inplace=True, axis=1)

    dfPrunedLayer = pd.read_csv(f'data/savedModels/{modelName}_prunedInfol2.csv')
    print(f'data/savedModels/{modelName}_prunedInfol2.csv read')

    # Layer 2
    columnsTags = [f'IN{i}' for i in range(model.valueSTE1.shape[1])]
    df = pd.DataFrame(model.valueSTE1, columns=columnsTags)
    importancePerEntryPre = pd.read_csv(f'./data/importance/{modelName}/PerClasslayer0.csv', dtype=int)
    importancePerEntry = pd.read_csv(f'./data/importance/{modelName}/PerClasslayer1.csv')
    df = pd.DataFrame(np.where(importancePerEntryPre == 0, 2, df), columns=columnsTags)
    df[df == -1] = 0
    df = df.astype(int)
    for neuron in range(2675, model.valueSTE2.shape[1]):
        dfImportance = importancePerEntry[f'N{neuron}']    
        df[f'OUT{neuron:04d}'] = model.valueSTE2[:, neuron]
        df.replace({f'OUT{neuron:04d}': -1}, 0, inplace=True)
        aux = df.copy()

        aux = aux[dfImportance > 0]
        aux.reset_index(drop=True, inplace=True)

        aux = pruneAndDrop(df, dfPrunedLayer, f'N{neuron:04d}')
        
        createPLAFileEspresso(aux, f'./data/plas/{modelName}/ESPRESSOOptimizedPerClass_0/layer2/N{neuron:04d}', conflictMode=0)
        createPLAFileEspresso(aux, f'./data/plas/{modelName}/ESPRESSOOptimizedPerClass_1/layer2/N{neuron:04d}', conflictMode=1)
        createPLAFileEspresso(aux, f'./data/plas/{modelName}/ESPRESSOOptimizedPerClass_2/layer2/N{neuron:04d}', conflictMode=2)
        df.drop(f'OUT{neuron:04d}', inplace=True, axis=1)

    dfPrunedLayer = pd.read_csv(f'data/savedModels/{modelName}_prunedInfol3.csv')
    print(f'data/savedModels/{modelName}_prunedInfol3.csv read')

    # Layer 3
    columnsTags = [f'IN{i}' for i in range(model.valueSTE2.shape[1])]
    df = pd.DataFrame(model.valueSTE2, columns=columnsTags)
    importancePerEntryPre = pd.read_csv(f'./data/importance/{modelName}/PerClasslayer1.csv', dtype=int)
    importancePerEntry = pd.read_csv(f'./data/importance/{modelName}/PerClasslayer2.csv')
    df = pd.DataFrame(np.where(importancePerEntryPre == 0, 2, df), columns=columnsTags)
    df[df == -1] = 0
    df = df.astype(int)
    for neuron in range(model.valueSTE3.shape[1]):
        dfImportance = importancePerEntry[f'N{neuron}']    
        df[f'OUT{neuron:04d}'] = model.valueSTE3[:, neuron]
        df.replace({f'OUT{neuron:04d}': -1}, 0, inplace=True)
        aux = df.copy()

        aux = aux[dfImportance > 0]
        aux.reset_index(drop=True, inplace=True)

        aux = pruneAndDrop(df, dfPrunedLayer, f'N{neuron:04d}')
        
        createPLAFileEspresso(aux, f'./data/plas/{modelName}/ESPRESSOOptimizedPerClass_0/layer3/N{neuron:04d}', conflictMode=0)
        createPLAFileEspresso(aux, f'./data/plas/{modelName}/ESPRESSOOptimizedPerClass_1/layer3/N{neuron:04d}', conflictMode=1)
        createPLAFileEspresso(aux, f'./data/plas/{modelName}/ESPRESSOOptimizedPerClass_2/layer3/N{neuron:04d}', conflictMode=2)
        df.drop(f'OUT{neuron:04d}', inplace=True, axis=1)
