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

neuronPerLayer = 1024
modelFilename = f'/home/carlosl/Dokumente/nn-accelerator-logic/src/modelCreation/savedModels/MNISTSignbinNN100Epoch1024NPLnllCriterion'
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
    
# Apply threshold
print(f'APPLY THRESHOLD\n')
threshold = 10e-5
for iImp in range(len(importanceList)):
    importanceList[iImp] = importanceList[iImp] > threshold
    print(f'importance number {iImp} has shape {importanceList[iImp].shape}')
    
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
    for i in range(10):
        importancePerClass[iImp][i] = importancePerClass[iImp][i].sum(0) / len(importancePerClass[iImp][i])
        
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
    fig.update_layout(title=f'Layer {imp} total importance per neuron (EEB-{neuronPerLayer})',
                      xaxis_title="Neuron index",
                      yaxis_title="Neuron importance score",
                      paper_bgcolor='rgba(0,0,0,0)')
    fig.show()

    # Print classes that are important
    aux = (importancePerClass[imp] > 0).sum(0)
    aux.sort()

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=aux))
    fig.update_layout(title=f'Layer {imp} number of important classes per neuron (EEB-{neuronPerLayer})',
                      xaxis_title="Neuron index",
                      yaxis_title="Number of important classes",
                      paper_bgcolor='rgba(0,0,0,0)')
    fig.show()

