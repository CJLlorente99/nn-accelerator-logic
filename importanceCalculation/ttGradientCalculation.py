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

neuronPerLayer = 100
modelFilename = f'../models/savedModels/binaryNN100Epoch{neuronPerLayer}NPLBlackAndWhite'
batch_size = 64
perGradientSampling = 1
nClasses = 10

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

model = BinaryNeuralNetwork(neuronPerLayer)
model.load_state_dict(torch.load(modelFilename))

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
model.registerHooks()
model.eval()

for i in range(sampleSize):
	X = training_data.data[i]
	y = training_data.targets[i]
	x = torch.reshape(Variable(X).type(torch.FloatTensor), (1, 28, 28))
	model.zero_grad()
	pred = model(x)
	pred[0, y.item()].backward()

	if (i + 1) % 500 == 0:
		print(f"Get Gradients and Activation Values [{i + 1:>5d}/{sampleSize:>5d}]")

# Compute importance

importance = model.computeImportance(neuronPerLayer)

# Give each neuron its importance values
for j in range(len(importance)):
	for i in range(neuronPerLayer):
		accLayers[j].neurons[i].giveImportance(importance[j][:, i], training_data.targets.tolist())

# Instead of storing one 8bit word per activation, decompose as sum of products of 2
model.individualActivationsToUniqueValue()

neuronTags = ['activation' + str(i) for i in range(model.activationSize)]
classTags = ['class' + str(i) for i in range(nClasses)]

# valueSTE1 = np.hstack((model.valueSTE1, training_data.targets[:sampleSize].detach().numpy().reshape((sampleSize, 1))))
# valueSTE1 = np.hstack((valueSTE1, np.zeros((sampleSize, nClasses))))
valueSTE2 = np.hstack((model.valueSTE2, training_data.targets[:sampleSize].detach().numpy().reshape((sampleSize, 1))))
valueSTE2 = np.hstack((valueSTE2, np.zeros((sampleSize, nClasses))))
valueSTE3 = np.hstack((model.valueSTE3, training_data.targets[:sampleSize].detach().numpy().reshape((sampleSize, 1))))
valueSTE3 = np.hstack((valueSTE3, np.zeros((sampleSize, nClasses))))

df = []

# dfLayer1 = pd.DataFrame(valueSTE1, columns=neuronTags + ['class'] + classTags)
dfLayer2 = pd.DataFrame(valueSTE2, columns=neuronTags + ['class'] + classTags)
dfLayer3 = pd.DataFrame(valueSTE3, columns=neuronTags + ['class'] + classTags)

df.append(dfLayer2)
df.append(dfLayer3)


def classToClassesUnpack(row):
	row['class' + str(int(row['class']))] = 1


for i in range(len(df)):
	df[i].apply(classToClassesUnpack, axis=1)
	df[i] = df[i].drop(['class'], axis=1)

# Group by neuron activations and sum class columns
typeDict = {}
for tag in classTags:
	typeDict[tag] = 'uint8'

# for tag in neuronTags:
#     typeDict[tag] = 'float64'

for i in range(len(df)):
	df[i] = df[i].groupby(neuronTags).aggregate('sum').reset_index().copy()
	df[i] = df[i].astype(typeDict)

# Assign to each layer object
for i in range(len(df)):
	accLayers[i].tt = df[i]

# Some data over the activation performed

i = 1
for frame in df:
	print(f'In layer {i}, there are a total of {len(frame)} input combinations from the {2 ** neuronPerLayer} possible')
	i += 1

# Create the TT per neuron

for layer in accLayers:
	layer.fillTT()

# Plot importance per entry of a neuron as an example

# Example belonging to layer 1

exampleNeuron = accLayers[0].neurons[random.randint(0, len(accLayers[0].neurons) - 1)]
exampleNeuron.showImportancePerEntry()
exampleNeuron.showImportancePerClassPerEntry()

# Example belonging to layer 2

exampleNeuron = accLayers[1].neurons[random.randint(0, len(accLayers[0].neurons) - 1)]
exampleNeuron.showImportancePerEntry()
exampleNeuron.showImportancePerClassPerEntry()
