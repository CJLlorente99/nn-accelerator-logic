import random
import pandas as pd
import torch
from models.auxFunctions import trainAndTest, ToBlackAndWhite
from modelsBNNPaper.auxFunctions import ToSign
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader
from models.binaryNN import BinaryNeuralNetwork
from models.fpNN import FPNeuralNetwork
from modelsBNNPaper.binaryNN import BNNBinaryNeuralNetwork
import torch.optim as optim
from ttUtilities.helpLayerNeuronGenerator import HelpGenerator
from torch.autograd import Variable
import numpy as np

neuronPerLayer = 4096
mod = False
modelUsed = f'MNISTSignbinNN50Epoch{neuronPerLayer}NPLhingeCriterion'
modelUsedImportance = f'SignBNN50epochs{neuronPerLayer}npl'
modelFilename = f'../modelsBNNPaper/savedModels/{modelUsed}'
layerActivationFilename = f'C:/Users/carlo/OneDrive/Documentos/Universidad/MUIT/Segundo/TFM/Code/data/activations/activations{modelUsedImportance}'
batch_size = 64
perGradientSampling = 1
nClasses = 10


def layerFilename(name):
	return f'C:/Users/carlo/OneDrive/Documentos/Universidad/MUIT/Segundo/TFM/Code/data/layersTT/{name}{modelUsed}'


def layerImportanceFilename(layer):
	return f'C:/Users/carlo/OneDrive/Documentos/Universidad/MUIT/Segundo/TFM/Code/data/layersImportance/layer{layer}ImportanceGradientBinary{modelUsedImportance}'


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

test_data = datasets.MNIST(
	root='C:/Users/carlo/OneDrive/Documentos/Universidad/MUIT/Segundo/TFM/Code/data',
	train=False,
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
test_dataloader = DataLoader(test_data, batch_size=batch_size)

sampleSize = int(perGradientSampling * len(training_data.data))

model = BNNBinaryNeuralNetwork(neuronPerLayer, mod=mod)
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
# dfActivations['out5'] = pd.read_feather(layerActivationFilename + 'Out5')
# model.valueIden = dfActivations['out5'].to_numpy()

# Instead of storing one 8bit word per activation, decompose as sum of products of 2
model.signToBinary()
model.individualActivationsToUniqueValue()

neuronTags = ['activation' + str(i) for i in range(model.activationSize)]
neuronTags = neuronTags + ['lengthActivation' + str(i) for i in range(model.activationSize)]
inputNeuronTags = ['activation' + str(i) for i in range(model.activationSizeInput)]
inputNeuronTags = inputNeuronTags + ['lengthActivation' + str(i) for i in range(model.activationSizeInput)]
# outputNeuronTags = ['activation' + str(i) for i in range(model.activationSizeOutput)]
# outputNeuronTags = outputNeuronTags + ['lengthActivation' + str(i) for i in range(model.activationSizeOutput)]
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
if mod:
	valueSTE4 = np.hstack(
		(model.valueSTE4, training_data.targets[:sampleSize].detach().numpy().reshape((sampleSize, 1))))
	valueSTE4 = np.hstack((valueSTE4, np.zeros((sampleSize, nClasses))))
# valueIden = np.hstack((model.valueIden, training_data.targets[:sampleSize].detach().numpy().reshape((sampleSize, 1))))
# valueIden = np.hstack((valueIden, np.zeros((sampleSize, nClasses))))

df = []

df.append(pd.DataFrame(valueInput0, columns=inputNeuronTags + ['class'] + classTags))
df.append(pd.DataFrame(valueSTE0, columns=neuronTags + ['class'] + classTags))
df.append(pd.DataFrame(valueSTE1, columns=neuronTags + ['class'] + classTags))
df.append(pd.DataFrame(valueSTE2, columns=neuronTags + ['class'] + classTags))
df.append(pd.DataFrame(valueSTE3, columns=neuronTags + ['class'] + classTags))
if mod:
	df.append(pd.DataFrame(valueSTE4, columns=neuronTags + ['class'] + classTags))
# df.append(pd.DataFrame(valueIden, columns=outputNeuronTags + ['class'] + classTags))


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

# for tag in neuronTags:
#     typeDict[tag] = 'float64'

for i in range(len(df)):
	if i == 0:
		df[i] = df[i].groupby(inputNeuronTags).aggregate('sum').reset_index().copy()
	# elif i == 5:
	# 	df[i] = df[i].groupby(outputNeuronTags).aggregate('sum').reset_index().copy()
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
	print(f'In layer {i}, there are a total of {len(frame)} input combinations from the {2 ** accLayers[i].nNeurons} possible')
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

exampleNeuron = accLayers[1].neurons[random.randint(0, len(accLayers[1].neurons) - 1)]
exampleNeuron.showImportancePerEntry()
exampleNeuron.showImportancePerClassPerEntry()

# Save the layers

for i in range(len(df)):
	accLayers[i].saveTT(layerFilename(f'layer{i}_'))
