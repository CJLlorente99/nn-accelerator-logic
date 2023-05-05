import random
import numpy as np
import pandas as pd
import torch
from modelsBNNPaper.auxFunctions import trainAndTest, ToBlackAndWhite, test
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader
from modelsBNNPaper.binaryNN import BNNBinaryNeuralNetwork
import torch.optim as optim
import torch.nn as nn
from torchmetrics.classification import MulticlassHingeLoss

batch_size = 100
neuronPerLayer = 4096
mod = True  # Change accordingly in modelFilename too
modelFilename = f'../modelsBNNPaper/savedModels/MNISTbinNNMod50Epoch{neuronPerLayer}NPLhingeCriterion'


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
    transform=Compose([
            ToBlackAndWhite(),
            ToTensor()
        ])
)

test_data = datasets.MNIST(
    root='C:/Users/carlo/OneDrive/Documentos/Universidad/MUIT/Segundo/TFM/Code/data',
    train=False,
    download=False,
    transform=Compose([
        ToBlackAndWhite(),
        ToTensor()
    ])
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

model = BNNBinaryNeuralNetwork(neuronPerLayer, mod).to(device)
model.load_state_dict(torch.load(modelFilename))

'''
Test
'''
print(f'TEST\n')

criterion = MulticlassHingeLoss(num_classes=10, squared=True, multiclass_mode='one-vs-all')

test(test_dataloader, train_dataloader, model, criterion)

