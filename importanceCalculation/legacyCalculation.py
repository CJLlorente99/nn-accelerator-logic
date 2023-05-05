import random
import pandas as pd
import torch
from models.auxFunctions import trainAndTest, ToBlackAndWhite
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader
from models.binaryNN import BinaryNeuralNetwork
from models.fpNN import FPNeuralNetwork
import torch.optim as optim
from ttUtilities.helpLayerNeuronGenerator import HelpGenerator
from torch.autograd import Variable
import numpy as np

modelFilename = '../models/savedModels/fullNN20Epoch100NPLBlackAndWhite'
batch_size = 64
neuronPerLayer = 100
perGradientSampling = 1

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

# model = FPNeuralNetwork(neuronPerLayer)
# model.load_state_dict(torch.load(modelFilename))

model = BinaryNeuralNetwork(neuronPerLayer)
model.load_state_dict(torch.load(modelFilename))

'''
Generate AccLayers and Neuron objects
'''

accLayers = HelpGenerator.getAccLayers(model)
accLayers.pop()  # Pop last element (binary-float)
HelpGenerator.getNeurons(accLayers)

'''
Calculate importance per class per neuron
'''

# Input samples and get gradients and values in each neuron
print(f'GET GRADIENTS AND ACTIVATION VALUES\n')

sampleSize = int(perGradientSampling * len(training_data.data))  # sample size to be used for importance calculation
model.legacyImportance = True
model.registerHooks()
model.eval()

importance = {}  # dict with one entry per layer, each layer is represented through a matrix (one row per sample,
# one column per neuron)
for layer in range(len(accLayers)):
    impMatrix = np.zeros((sampleSize, neuronPerLayer))
    for i in range(sampleSize):
        aux = []
        X = training_data.data[i]
        y = training_data.targets[i]
        x = torch.reshape(Variable(X).type(torch.FloatTensor), (1, 28, 28))
        for n in range(neuronPerLayer):
            model.neuronSwitchedOff = (-1, -1)  # So no output is short to 0
            pred_withoutTurnedOff = model(x).detach().squeeze().numpy()
            model.neuronSwitchedOff = (layer + 2, n)
            pred_withTurnedOff = model(x).detach().squeeze().numpy()
            aux.append(abs(pred_withoutTurnedOff - pred_withTurnedOff).sum())
        impMatrix[i, :] = np.array(aux)

        if (i+1) % 500 == 0:
            print(f"Get Gradients and Activation Values. Layer [{layer+1:>2d}/{len(accLayers):>2d}] "
                  f"Sample [{i+1:>5d}/{sampleSize:>5d}]")
    importance[layer] = impMatrix

# Give each neuron its importance values
for j in range(len(importance)):
    for i in range(neuronPerLayer):
        accLayers[j].neurons[i].giveImportance(importance[j][:, i], training_data.targets.tolist())

# Plot importance of neurons per layer

for i in range(len(accLayers)):
    # accLayers[i].plotImportancePerNeuron('FP')
    # accLayers[i].plotImportancePerClass('FP')
    accLayers[i].saveImportance(f'../data/layersImportance/2023_04_18_layer{i}ImportanceLegacyFull20epoch100nnpl')