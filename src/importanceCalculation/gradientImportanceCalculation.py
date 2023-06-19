import torch
from modelsCommon.auxTransformations import ToBlackAndWhite, ToSign
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader
from modules.binaryEnergyEfficiency import BinaryNeuralNetwork
from ttUtilities.helpLayerNeuronGenerator import HelpGenerator
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

neuronPerLayer = 100
modelFilename = f'/home/carlosl/Dokumente/nn-accelerator-logic/src/modelCreation/savedModels/MNIST_bin_100NPL'
batch_size = 1
perGradientSampling = 1
# Check mps maybe if working in MacOS
device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
Importing MNIST dataset
'''
print(f'IMPORT DATASET\n')

training_data = datasets.MNIST(
    root='/home/carlosl/Dokumente/nn-accelerator-logic/data',
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

model = BinaryNeuralNetwork(neuronPerLayer)
model.load_state_dict(torch.load(modelFilename))

'''
Calculate importance per class per neuron
'''

# Input samples and get gradients and values in each neuron
print(f'GET GRADIENTS AND ACTIVATION VALUES\n')

model.registerHooks()
model.eval()

for i in range(sampleSize):
    X, y = train_dataloader.dataset[i]
    model.zero_grad()
    pred = model(X)
    pred[0, y].backward()

    if (i+1) % 500 == 0:
        print(f"Get Gradients and Activation Values [{i+1:>5d}/{sampleSize:>5d}]")

model.listToArray(neuronPerLayer)  # Hopefully improves memory usage
importanceList = model.computeImportance(neuronPerLayer)


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
dupList.append(getIdxDuplicates(model.input0))
uniqueInput0 = np.unique(model.input0, axis=0)
print(f'Original length {model.input0.shape[0]}, only unique length {uniqueInput0.shape[0]}, number of sample {sampleSize}')
dupList.append(getIdxDuplicates(model.valueSTE0))
uniqueSTE0 = np.unique(model.valueSTE0, axis=0)
print(f'Original length {model.valueSTE0.shape[0]}, only unique length {uniqueSTE0.shape[0]}, number of sample {sampleSize}')
dupList.append(getIdxDuplicates(model.valueSTE1))
uniqueSTE1 = np.unique(model.valueSTE1, axis=0)
print(f'Original length {model.valueSTE1.shape[0]}, only unique length {uniqueSTE1.shape[0]}, number of sample {sampleSize}')
dupList.append(getIdxDuplicates(model.valueSTE2))
uniqueSTE2 = np.unique(model.valueSTE2, axis=0)
print(f'Original length {model.valueSTE2.shape[0]}, only unique length {uniqueSTE2.shape[0]}, number of sample {sampleSize}')

# Apply threshold
print(f'APPLY THRESHOLD\n')
threshold = 10e-5
for iImp in range(len(importanceList)):
    importanceList[iImp] = importanceList[iImp] > threshold
    for dup in dupList[iImp]:
        if len(dup) != 0:
            importanceList[iImp][dup[0], :] = np.sum(importanceList[iImp][dup, :], axis=0)
    print(f'importance number {iImp} has shape {importanceList[iImp].shape}')
    print(f'importance number {iImp} has {importanceList[iImp].sum().sum()} entries above threshold {threshold} out of {importanceList[iImp].size}')
    
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
        importancePerClass[iImp][training_data.targets[i].item()].append(importanceList[iImp][i, :])

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
    
# Print results
for imp in importancePerClass:
    # Print aggregated importance
    aux = importancePerClass[imp].sum(0)
    aux.sort()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(aux))), y=aux))
    fig.update_layout(title=f'Layer {imp} total importance per neuron (EE-{neuronPerLayer})',
                      xaxis_title="Neuron index",
                      yaxis_title="Neuron importance score",
                      paper_bgcolor='rgba(0,0,0,0)')
    fig.show()

    # Print classes that are important
    aux = (importancePerClass[imp] > 0).sum(0)
    aux.sort()

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=aux))
    fig.update_layout(title=f'Layer {imp} number of important classes per neuron (EE-{neuronPerLayer})',
                      xaxis_title="Neuron index",
                      yaxis_title="Number of important classes",
                      paper_bgcolor='rgba(0,0,0,0)')
    fig.show()
