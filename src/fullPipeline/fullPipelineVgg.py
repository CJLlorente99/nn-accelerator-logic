from datetime import datetime
import torch
from modelsCommon.auxFunc import trainAndTest, getAccInfo
from modelsCommon.auxTransformations import *
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize, RandomHorizontalFlip, RandomCrop, Resize
from torch.utils.data import DataLoader
from modules.binaryVggVerySmall import binaryVGGVerySmall
from modules.binaryVggVerySmall2 import binaryVGGVerySmall2
from modules.vggVerySmall import VGGVerySmall
from modules.vggSmall import VGGSmall
import torch.optim as optim
import torch.nn as nn
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import os

today = datetime.now().strftime("%Y_%m_%d_%H_%M")

'''
1) Create and train the model
'''
# Check mps maybe if working in MacOS
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ! PARAMETERS
batch_size = 200
epochs = 20
resize_multiplier = 2
model = VGGVerySmall().to(device)
modelName = 'VGGVerySmall' + today

'''
2) Get gradients and activations
'''
# ! PARAMETERS
perGradientSampling = 0.2

'''
3) Calculate importance scores and get valuable info
'''
# ! PARAMETERS
threshold = 10e-50
printImportancePerFilter = True
printClassesImportant = True
importancePerClass = False

'''
4) Create the TT per neuron and simplify them
'''
# ! PARAMETERS
# NONE

'''
5) Create PLAs
'''
# ! PARAMETERS
espresso = False
abc = True

# Execution
print(f'DOWNLOAD DATASET\n')
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=Compose([
        RandomHorizontalFlip(),
        RandomCrop(32, 4),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        Resize(resize_multiplier*32, antialias=False)]),
    download=False)

test_dataset = datasets.CIFAR10(root='./data', train=False, transform=Compose([
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        Resize(resize_multiplier*32, antialias=False)]),
    download=False)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Execution 1)
print(f'TRAINING\n')

opt = optim.SGD(model.parameters(), lr=0.02, weight_decay=0.0005, momentum=0.9)
criterion = nn.CrossEntropyLoss()

trainAndTest(epochs, train_dataloader, test_dataloader, model, opt, criterion)
with open(f'data/logs/{modelName}', 'w') as f:
    f.write(f'{getAccInfo(train_dataloader, test_dataloader, model, criterion)}\n')
    f.close()

torch.save(model.state_dict(), f'./src/modelCreation/savedModels/{modelName}')

# Execution 2)
sampleSize = int(perGradientSampling * len(train_dataset.data))  # sample size to be used for importance calculation
with open(f'data/logs/{modelName}', 'a') as f:
    f.write(f'Sample size used is {sampleSize}, {perGradientSampling * 100}\n')
    f.close()

print(f'GET GRADIENTS AND ACTIVATION VALUES\n')

model.registerHooks()
model.eval()

for i in range(sampleSize):
    X, y = train_dataloader.dataset[i]
    model.zero_grad()
    pred = model(X[None, :, :, :])
    pred[0, y].backward()

    if (i+1) % 500 == 0:
        print(f"Get Gradients and Activation Values [{i+1:>5d}/{sampleSize:>5d}]")

model.listToArray()  # Hopefully improves memory usage

model.saveActivations(f'data/activations/{modelName}')
model.saveGradients(f'data/gradients/{modelName}')

with open(f'data/logs/{modelName}', 'a') as f:
    f.write(f'Activations saved in data/activations/{modelName}\n')
    f.write(f'Gradients saved in data/gradients/{modelName}\n')
    f.close()
    
# Execution 3)
importances = model.computeImportance(True, f'data/importances/{modelName}')

for i in range(len(importances)):
    importances[i] = (importances[i] > threshold)

importancePerClassFilter = {}
importancePerClassNeuron = {}
# Initialize for 10 classes and layers
for grad in model.helpHookList:
    importancePerClassFilter[grad] = {}
    importancePerClassNeuron[grad] = {}
    for i in range(10):
        importancePerClassFilter[grad][i] = []
        importancePerClassNeuron[grad][i] = {}

# Iterate through the calculated importances and assign depending on the class
imp = 0
for grad in model.helpHookList:
    importance = importances[imp]
    if importance.ndim > 2:
        for i in range(importance.shape[1]):
            importancePerClassFilter[grad][train_dataset.targets[i]].append(importance[:, i, :])
    else:
        for i in range(importance.shape[0]):
            importancePerClassFilter[grad][train_dataset.targets[i]].append(importance[i, :])
    imp += 1

# Change from list to array each entry of importancePerClassFilter
for grad in model.helpHookList:
    for nClass in importancePerClassFilter[grad]:
        importancePerClassFilter[grad][nClass] = np.array(importancePerClassFilter[grad][nClass])

# Iterate through the importance scores and convert them into importance values
for grad in model.helpHookList:
    for nClass in importancePerClassFilter[grad]:
        if importancePerClassFilter[grad][nClass].ndim > 2: # Then it comes from a conv layer
            importancePerClassFilter[grad][nClass] = importancePerClassFilter[grad][nClass].sum(0) / len(importancePerClassFilter[grad][nClass])
            importancePerClassNeuron[grad][nClass] = importancePerClassFilter[grad][nClass].flatten() # To store information about all neurons
            # Take the max score per filter
            importancePerClassFilter[grad][nClass] = importancePerClassFilter[grad][nClass].max(0)
        else: # Then it comes from a neural layer
            importancePerClassFilter[grad][nClass] = importancePerClassFilter[grad][nClass].sum(0) / len(importancePerClassFilter[grad][nClass])
    
# Join the lists so each row is a class and each column a filter/neuron
for grad in importancePerClassFilter:
    importancePerClassFilter[grad] = np.row_stack(tuple(importancePerClassFilter[grad].values()))
    importancePerClassNeuron[grad] = np.row_stack(tuple(importancePerClassNeuron[grad].values()))
    
for grad in importancePerClassFilter:
    # Print aggregated importance
    aux = importancePerClassFilter[grad].sum(0)
    with open(f'data/logs/{modelName}', 'a') as f:
        f.write(f'Number of neurons not important in {grad} is {(aux == 0).sum()}\n')
        f.close()
    aux.sort()

    if printImportancePerFilter:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(aux))), y=aux))
        fig.update_layout(title=f'{grad} total importance per filter ({len(aux)})')
        fig.show()
    
    # Print classes that are important
    if grad.startswith('relul'):
        aux = (importancePerClassFilter[grad] > 0).sum(0)
        aux.sort()
        
        for i in range(11):
            with open(f'data/logs/{modelName}', 'a') as f:
                f.write(f'Number of filters with {i} classes important in {grad} is {(aux == i).sum()}\n')
        f.close()
        
        if printClassesImportant:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=aux))
            fig.update_layout(title=f'{grad} number of important classes per neuron ({len(aux)})')
            fig.show()
    else:
        aux = (importancePerClassNeuron[grad] > 0).sum(0)
        aux.sort()
        
        for i in range(11):
            with open(f'data/logs/{modelName}', 'a') as f:
                f.write(f'Number of neurons with {i} classes important in {grad} is {(aux == i).sum()}\n')
        f.close()
        
        if printClassesImportant:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=aux))
            fig.update_layout(title=f'{grad} number of important classes per neuron ({len(aux)})')
            fig.show()

    # Print importance per class
    if importancePerClass:
        aux = pd.DataFrame(importancePerClassFilter[grad],
                        columns=[f'filter{i}' for i in range(importancePerClassFilter[grad].shape[1])])
        fig = go.Figure()
        for index, row in aux.iterrows():
            fig.add_trace(go.Bar(name=f'class{index}', y=row, x=aux.columns))
        fig.update_layout(title=f'{grad} importance per class', barmode='stack')
        fig.show()
           
# Execution 4)
previousGrad = ''
iImportance = 0
for grad in model.helpHookList:
    if grad.startswith('ste') and previousGrad.startswith('ste'): # Input and output binary conv to conv
        previousGrad = grad

        # If conv layer, change to (samples, filters)
        if not previousGrad.startswith('stel'):
        # TODO reduce the third dimension (samples, filters)
            inputToLayer = model.dataFromHooks[previousGrad]['forward'].flatten()
        if not grad.startswith('stel'):
            outputToLayer = model.dataFromHooks[grad]['forward'].flatten()
            imp = importances[iImportance].flatten()

        columnTags = [f'F{i}' for i in range(inputToLayer.shape[1])]
        for iFilter in outputToLayer.shape[1]: # loop through the filters
            samples = []
            for iSample in inputToLayer.shape[0]: # loop through the samples
                # Standard way to reduce the truth table (looking which class is related to the entry and noting if it's important)
                if importancePerClassFilter[grad][train_dataset.targets[iSample]][iFilter] != 0: # Then add entry
                    samples.append(iSample)
                # Alternative way to reduce the truth table (looking if the entry was impotant)
                # if imp[iSample, iFilter] != 0: # Then add entry
                # 	samples.append(iSample)
            df = pd.DataFrame(inputToLayer[samples, :], columns=columnTags)
            df[f'output{grad}_{iFilter}'] = outputToLayer[samples, iFilter]
            with open(f'data/logs/{modelName}', 'a') as f:
                f.write(f'Change number of entries in {grad}/{iFilter} is {inputToLayer.shape[0]} -> {len(df)}\n')
                f.close()
            # Check folder exists
            if not os.path.exists(f'data/optimizedTT/{modelName}/{grad}'):
                os.makedirs(f'data/optimizedTT/{modelName}/{grad}')
            df.to_feather(f'data/optimizedTT/{modelName}/{grad}/F{iFilter}')
    iImportance += 1
