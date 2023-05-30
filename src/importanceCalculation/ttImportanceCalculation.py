import random
import pandas as pd
import torch
from modelsCommon.auxTransformations import *
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader
from modules.binaryEnergyEfficiency import BinaryNeuralNetwork
from ttUtilities.helpLayerNeuronGenerator import HelpGenerator
from ttUtilities.auxFunctions import integerToBinaryArray
import numpy as np

neuronPerLayer = 100
modelUsed = f'MNISTSignbinNN100Epoch100NPLnllCriterion'
modelUsedImportance = f'SignBNN50epochs{neuronPerLayer}npl'
modelFilename = f'src\modelCreation\savedModels\MNISTSignbinNN100Epoch100NPLnllCriterion'
layerActivationFilename = f'./data/activations/activationsSignBin100epochs100npl'
batch_size = 64
perGradientSampling = 1
nClasses = 10
threshold = '1e1'


def layerFilename(name):
	return f'./data/layersTT/{name}{modelUsed}'


def layerImportanceFilename(layer):
	return f'./data/layersImportance/layer{layer}Importance{threshold}GradientBinary{modelUsedImportance}'


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
		ToBlackAndWhite(),
		ToSign()
	])
)

'''
Create DataLoader
'''
train_dataloader = DataLoader(training_data, batch_size=batch_size)

sampleSize = int(perGradientSampling * len(training_data.data))

model = BinaryNeuralNetwork(neuronPerLayer)
model.load_state_dict(torch.load(modelFilename))

'''
Generate AccLayers and Neuron objects
'''

accLayers = HelpGenerator.getAccLayers(model)
HelpGenerator.getNeurons(accLayers)

'''
Load from files
'''
dfImportance = {}
for i in range(len(accLayers) - 1):  # No importance in the last layer
	accLayers[i].fillImportanceDf(pd.read_feather(layerImportanceFilename(i)))

dfActivations = {}
dfActivations['input0'] = pd.read_feather(layerActivationFilename + 'Input0')
model.input0 = dfActivations['input0'].to_numpy()
dfActivations['input1'] = pd.read_feather(layerActivationFilename + 'Input1')
model.valueSTE0 = dfActivations['input1'].to_numpy()
dfActivations['input2'] = pd.read_feather(layerActivationFilename + 'Input2')
model.valueSTE1 = dfActivations['input2'].to_numpy()
dfActivations['input3'] = pd.read_feather(layerActivationFilename + 'Input3')
model.valueSTE2 = dfActivations['input3'].to_numpy()
dfActivations['input4'] = pd.read_feather(layerActivationFilename + 'Input4')
model.valueSTE3 = dfActivations['input4'].to_numpy()

# Instead of storing one 8bit word per activation, decompose as sum of products of 2
model.signToBinary()
model.individualActivationsToUniqueValue()

neuronTags = ['activation' + str(i) for i in range(model.activationSize)]
neuronTags = neuronTags + ['lengthActivation' + str(i) for i in range(model.activationSize)]
inputNeuronTags = ['activation' + str(i) for i in range(model.activationSizeInput)]
inputNeuronTags = inputNeuronTags + ['lengthActivation' + str(i) for i in range(model.activationSizeInput)]
classTags = ['class' + str(i) for i in range(nClasses)]

valueInput0 = np.hstack((model.input0, training_data.targets[:sampleSize].detach().numpy().reshape((sampleSize, 1))))
valueInput0 = np.hstack((valueInput0, np.zeros((sampleSize, nClasses))))
valueSTE0 = np.hstack((model.valueSTE0, training_data.targets[:sampleSize].detach().numpy().reshape((sampleSize, 1))))
valueSTE0 = np.hstack((valueSTE0, np.zeros((sampleSize, nClasses))))
valueSTE1 = np.hstack((model.valueSTE1, training_data.targets[:sampleSize].detach().numpy().reshape((sampleSize, 1))))
valueSTE1 = np.hstack((valueSTE1, np.zeros((sampleSize, nClasses))))
valueSTE2 = np.hstack((model.valueSTE2, training_data.targets[:sampleSize].detach().numpy().reshape((sampleSize, 1))))
valueSTE2 = np.hstack((valueSTE2, np.zeros((sampleSize, nClasses))))
valueSTE3 = np.hstack((model.valueSTE3, training_data.targets[:sampleSize].detach().numpy().reshape((sampleSize, 1))))
valueSTE3 = np.hstack((valueSTE3, np.zeros((sampleSize, nClasses))))

df = []

df.append(pd.DataFrame(valueInput0, columns=inputNeuronTags + ['class'] + classTags))
df.append(pd.DataFrame(valueSTE0, columns=neuronTags + ['class'] + classTags))
df.append(pd.DataFrame(valueSTE1, columns=neuronTags + ['class'] + classTags))
df.append(pd.DataFrame(valueSTE2, columns=neuronTags + ['class'] + classTags))
df.append(pd.DataFrame(valueSTE3, columns=neuronTags + ['class'] + classTags))

def classToClassesUnpack(row):
	row['class' + str(int(row['class']))] = 1


for i in range(len(df)):
	df[i].apply(classToClassesUnpack, axis=1)
	df[i] = df[i].drop(['class'], axis=1)

	if (i + 1) % 1 == 0:
		print(f"Class to Classes Unpack [{i + 1:>2d}/{len(df):>2d}]")

# Group by neuron activations and sum class columns
typeDict = {}
for tag in classTags:
	typeDict[tag] = 'uint8'

for i in range(len(df)):
	if i == 0:
		df[i] = df[i].groupby(inputNeuronTags).aggregate('sum').reset_index().copy()
	else:
		df[i] = df[i].groupby(neuronTags).aggregate('sum').reset_index().copy()
	df[i] = df[i].astype(typeDict)

	if (i + 1) % 1 == 0:
		print(f"Group Entries [{i + 1:>2d}/{len(df):>2d}]")

# Assign to each layer object
for i in range(len(df)):
	accLayers[i].tt = df[i]

	if (i + 1) % 1 == 0:
		print(f"Assign to Layers [{i + 1:>2d}/{len(df):>2d}]")

# Some data over the activation performed

i = 0
for frame in df:
	print(f'In layer {i}, there are a total of {len(frame)} unique input combinations')
	i += 1

# Create the TT per neuron

for i in range(len(accLayers)):
	if i < len(accLayers) - 1:
		accLayers[i].fillTT(dfActivations[f'input{i+1}'])

# Plot importance per entry of a neuron as an example

# Example belonging to layer 1

exampleNeuron = accLayers[0].neurons[random.randint(0, len(accLayers[0].neurons) - 1)]
exampleNeuron.showImportancePerEntry()
exampleNeuron.showImportancePerClassPerEntry()

# Example belonging to layer 2

exampleNeuron = accLayers[1].neurons[random.randint(0, len(accLayers[1].neurons) - 1)]
exampleNeuron.showImportancePerEntry()
exampleNeuron.showImportancePerClassPerEntry()

# Save the layers

for i in range(len(df)):
	accLayers[i].saveTT(layerFilename(f'layer{i}_'))
