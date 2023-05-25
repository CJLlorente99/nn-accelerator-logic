import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from modules.binaryEnergyEfficiency import BinaryNeuralNetwork
import torch.nn as nn
import pandas as pd
import numpy as np

batch_size = 100
neuronPerLayer = 100
mod = True  # Change accordingly in modelFilename too
modelFilename = f'../models/savedModels/MNISTbinNN100Epoch{neuronPerLayer}NPLnllCriterion'
precision = 'bin'
lastLayerInputsTrainFilename = f'example'
lastLayerInputsTestFilename = f'example'


# Check mps maybe if working in MacOS
device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
Importing MNIST dataset
'''
print(f'IMPORT DATASET\n')

training_data = datasets.MNIST(
    root='C:/Users/carlo/OneDrive/Documentos/Universidad/MUIT/Segundo/TFM/Code/data',
    train=True,
    download=False,
    transform=ToTensor()
    )

test_data = datasets.MNIST(
    root='C:/Users/carlo/OneDrive/Documentos/Universidad/MUIT/Segundo/TFM/Code/data',
    train=False,
    download=False,
    transform=ToTensor()
    )

'''
Create DataLoader
'''
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

'''
Instantiate NN models
'''
print(f'MODEL INSTANTIATION\n')

model = BinaryNeuralNetwork(neuronPerLayer).to(device)
model.load_state_dict(torch.load(modelFilename))

'''
Load the simulated inputs to the last layer (provided by minimized network)
'''
columnTags = [f'N{i}' for i in range(10)]
dfInputsLastLayerTrain = pd.DataFrame(columns=columnTags)

with open(lastLayerInputsTrainFilename) as f:
        while True:
            line = np.array(list(f.readline()), dtype=np.float)
            if not line: break
            
            dfInputsLastLayerTrain = pd.concat(dfInputsLastLayerTrain,
                                          pd.DataFrame(line, columns=columnTags))

dfInputsLastLayerTest = pd.DataFrame(columns=columnTags)
        
with open(lastLayerInputsTestFilename) as f:
        while True:
            line = np.array(list(f.readline()), dtype=np.float)
            if not line: break
            
            dfInputsLastLayerTest = pd.concat(dfInputsLastLayerTest,
                                          pd.DataFrame(line, columns=columnTags))

'''
Test
'''
print(f'TEST Train\n')
sizeTotal = len(train_dataloader.dataset) + len(test_dataloader.dataset)
model.eval()
totalCorrect = 0

correct = 0
size = len(train_dataloader.dataset)
count = 0
for index, row in dfInputsLastLayerTrain.iterrows():
    with torch.no_grad():
        pred = model.forwardLastLayer(row)
        correct += (pred.argmax(1) == train_dataloader.target[count]).type(torch.float).sum().item()
        count += 1

totalCorrect += correct
print(f"Train Error: \n Accuracy: {(100 * correct / size):>0.2f}%\n")
    
print(f'TEST Test\n')

correct = 0
size = len(test_dataloader.dataset)
count = 0
for index, row in dfInputsLastLayerTest.iterrows():
    with torch.no_grad():
        pred = model.forwardLastLayer(row)
        correct += (pred.argmax(1) == test_dataloader.target[count]).type(torch.float).sum().item()
        count += 1

totalCorrect += correct
print(f"Test Error: \n Accuracy: {(100 * correct / size):>0.2f}%\n")

print(f"Total Test Error: \n Accuracy: {(100 * totalCorrect / sizeTotal):>0.2f}%\n")
