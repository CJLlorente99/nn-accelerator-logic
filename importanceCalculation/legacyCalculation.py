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

modelFilename = '../models/savedModels/fullNN100Epoch100NPLBlackAndWhite'
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
            ToTensor(),
            ToBlackAndWhite()
        ])
)

test_data = datasets.MNIST(
    root='C:/Users/carlo/OneDrive/Documentos/Universidad/MUIT/Segundo/TFM/Code/data',
    train=False,
    download=False,
    transform=Compose([
        ToTensor(),
        ToBlackAndWhite()
    ])
)

'''
Create DataLoader
'''
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

model = FPNeuralNetwork(neuronPerLayer)
model.load_state_dict(torch.load(modelFilename))

# model = BinaryNeuralNetwork(neuronPerLayer)
# model.load_state_dict(torch.load('models/savedModels/binaryNN100Epoch100NPLBlackAndWhite'))

'''
Generate AccLayers and Neuron objects
'''

accLayers = HelpGenerator.getAccLayers(model)
accLayers.pop(0)  # Pop first element (float-float)
accLayers.pop(0)  # Pop second element (float-binary
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

# Get activations-class per layer
# neuronTags = ['n' + str(i) for i in range(neuronPerLayer)]
# Instead of storing one 8bit word per activation, decompose as sum of products of 2
# binaryNNModel.individualActivationsToUniqueValue()
#
# neuronTags = ['activation' + str(i) for i in range(binaryNNModel.valueSTE1.shape[1])]
# classTags = ['class' + str(i) for i in range(nClasses)]
#
# valueSTE1 = np.hstack((binaryNNModel.valueSTE1, training_data.targets[:sampleSize].detach().numpy().reshape((sampleSize, 1))))
# valueSTE1 = np.hstack((valueSTE1, np.zeros((sampleSize, nClasses))))
# valueSTE2 = np.hstack((binaryNNModel.valueSTE2, training_data.targets[:sampleSize].detach().numpy().reshape((sampleSize, 1))))
# valueSTE2 = np.hstack((valueSTE2, np.zeros((sampleSize, nClasses))))
# valueSTE3 = np.hstack((binaryNNModel.valueSTE3, training_data.targets[:sampleSize].detach().numpy().reshape((sampleSize, 1))))
# valueSTE3 = np.hstack((valueSTE3, np.zeros((sampleSize, nClasses))))
#
# dfLayer1 = pd.DataFrame(valueSTE1, columns=neuronTags + ['class'] + classTags)
# dfLayer2 = pd.DataFrame(valueSTE2, columns=neuronTags + ['class'] + classTags)
# dfLayer3 = pd.DataFrame(valueSTE3, columns=neuronTags + ['class'] + classTags)


# def classToClassesUnpack(row):
#     row['class' + str(int(row['class']))] = 1
#
#
# dfLayer1.apply(classToClassesUnpack, axis=1)
# dfLayer1 = dfLayer1.drop(['class'], axis=1)
#
# dfLayer2.apply(classToClassesUnpack, axis=1)
# dfLayer2 = dfLayer2.drop(['class'], axis=1)
#
# dfLayer3.apply(classToClassesUnpack, axis=1)
# dfLayer3 = dfLayer3.drop(['class'], axis=1)
#
# # Group by neuron activations and sum class columns
# typeDict = {}
# for tag in classTags:
#     typeDict[tag] = 'uint8'
#
# # for tag in neuronTags:
# #     typeDict[tag] = 'float64'
#
# dfLayer1 = dfLayer1.groupby(neuronTags).aggregate('sum').reset_index().copy()
# dfLayer1 = dfLayer1.astype(typeDict)
#
# dfLayer2 = dfLayer2.groupby(neuronTags).aggregate('sum').reset_index().copy()
# dfLayer2 = dfLayer2.astype(typeDict)
#
# dfLayer3 = dfLayer3.groupby(neuronTags).aggregate('sum').reset_index().copy()
# dfLayer3 = dfLayer3.astype(typeDict)
#
# # Assign to each layer object
# accLayers[0].tt = dfLayer1
# accLayers[1].tt = dfLayer2
# accLayers[2].tt = dfLayer3
#
# # Some data over the activation performed
#
# print(f'In layer 1, there are a total of {len(dfLayer1)} input combinations from the {2**neuronPerLayer} possible')
# print(f'In layer 2, there are a total of {len(dfLayer2)} input combinations from the {2**neuronPerLayer} possible')
# print(f'In layer 3, there are a total of {len(dfLayer3)} input combinations from the {2**neuronPerLayer} possible')
#
# # Create the TT per neuron
#
# for layer in accLayers:
#     layer.fillTT()

# Plot importance per entry of a neuron as an example

# Example belonging to layer 1

# exampleNeuron = accLayers[0].neurons[random.randint(0, len(accLayers[0].neurons) - 1)]
# exampleNeuron.showImportancePerEntry()
# exampleNeuron.showImportancePerClassPerEntry()
#
# # Example belonging to layer 2
#
# exampleNeuron = accLayers[1].neurons[random.randint(0, len(accLayers[0].neurons) - 1)]
# exampleNeuron.showImportancePerEntry()
# exampleNeuron.showImportancePerClassPerEntry()

# Plot importance of neurons per layer

for i in range(len(accLayers)):
    accLayers[i].plotImportancePerNeuron('FP')
    accLayers[i].plotImportancePerClass('FP')
