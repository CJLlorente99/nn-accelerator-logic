import pandas as pd
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from models.binaryNN import BinaryNeuralNetwork
import torch.optim as optim
from ttUtilities.helpLayerNeuronGenerator import HelpGenerator
from models.auxFunctions import trainAndTest
from torch.autograd import Variable
import numpy as np

# TODO. Change according to mnist_quantization changes

batch_size = 64
tryNeurons = [5, 10, 20, 40, 80, 100]
epochs = 10
perGradientSampling = 1

# Check mps maybe if working in MacOS
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

'''
Importing MNIST dataset
'''
print(f'IMPORT DATASET\n')

training_data = datasets.MNIST(
	root='data',
	train=True,
	download=True,
	transform=ToTensor()
)

test_data = datasets.MNIST(
	root='data',
	train=False,
	download=True,
	transform=ToTensor()
)

'''
Create DataLoader
'''

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

'''
Train and test functions
'''

nClasses = len(training_data.class_to_idx)

for neuronPerLayer in tryNeurons:
	importanceCharacteristics = pd.DataFrame()
	print(f'Trying NN with {neuronPerLayer} neurons per layer')

	binaryNNModel = BinaryNeuralNetwork(neuronPerLayer).to(device)
	'''
	Train and test
	'''
	print(f'TRAINING\n')

	optBinary = optim.Adamax(binaryNNModel.parameters(), lr=3e-3, weight_decay=1e-4)

	print('Train and test binary NN')
	trainAndTest(epochs, train_dataloader, test_dataloader, binaryNNModel, optBinary)

	'''
	Generate TT
	'''

	accLayers = HelpGenerator.getAccLayers(binaryNNModel)
	accLayers.pop(0)  # Pop first element
	accLayers.pop()  # Pop last element
	HelpGenerator.getNeurons(accLayers)

	'''
	Calculate importance per class per neuron
	'''

	# Input samples and get gradients and values in each neuron
	print(f'GET GRADIENTS AND ACTIVATION VALUES\n')

	sampleSize = int(perGradientSampling * len(training_data.data))  # sample size to be used for importance calculation
	binaryNNModel.registerHooks()
	binaryNNModel.eval()

	for i in range(sampleSize):
		X = training_data.data[i]
		y = training_data.targets[i]
		x = torch.reshape(Variable(X).type(torch.FloatTensor), (1, 28, 28))
		binaryNNModel.zero_grad()
		pred = binaryNNModel(x)
		pred[0, y.item()].backward()

		if (i+1) % 500 == 0:
			print(f"Get Gradients and Activation Values [{i+1:>5d}/{sampleSize:>5d}]")

	# Compute importance

	importanceSTE1, importanceSTE2, importanceSTE3 = binaryNNModel.computeImportance(neuronPerLayer)

	for i in range(neuronPerLayer):
		accLayers[0].neurons[i].giveImportance(importanceSTE1[:, i], training_data.targets.tolist())
		accLayers[1].neurons[i].giveImportance(importanceSTE2[:, i], training_data.targets.tolist())
		accLayers[2].neurons[i].giveImportance(importanceSTE3[:, i], training_data.targets.tolist())

	# Get activations-class per layer
	neuronTags = ['n' + str(i) for i in range(neuronPerLayer)]
	classTags = ['class' + str(i) for i in range(nClasses)]

	valueSTE1 = np.hstack(
		(binaryNNModel.valueSTE1, training_data.targets[:sampleSize].detach().numpy().reshape((sampleSize, 1))))
	valueSTE1 = np.hstack((valueSTE1, np.zeros((sampleSize, nClasses))))
	valueSTE2 = np.hstack(
		(binaryNNModel.valueSTE2, training_data.targets[:sampleSize].detach().numpy().reshape((sampleSize, 1))))
	valueSTE2 = np.hstack((valueSTE2, np.zeros((sampleSize, nClasses))))
	valueSTE3 = np.hstack(
		(binaryNNModel.valueSTE3, training_data.targets[:sampleSize].detach().numpy().reshape((sampleSize, 1))))
	valueSTE3 = np.hstack((valueSTE3, np.zeros((sampleSize, nClasses))))

	dfLayer1 = pd.DataFrame(valueSTE1, columns=neuronTags + ['class'] + ['class' + str(n) for n in range(nClasses)])
	dfLayer2 = pd.DataFrame(valueSTE2, columns=neuronTags + ['class'] + ['class' + str(n) for n in range(nClasses)])
	dfLayer3 = pd.DataFrame(valueSTE3, columns=neuronTags + ['class'] + ['class' + str(n) for n in range(nClasses)])


	def classToClassesUnpack(row):
		row['class' + str(int(row['class']))] = 1


	dfLayer1.apply(classToClassesUnpack, axis=1)
	dfLayer1 = dfLayer1.drop(['class'], axis=1)

	dfLayer2.apply(classToClassesUnpack, axis=1)
	dfLayer2 = dfLayer2.drop(['class'], axis=1)

	dfLayer3.apply(classToClassesUnpack, axis=1)
	dfLayer3 = dfLayer3.drop(['class'], axis=1)

	# Group by neuron activations and sum class columns
	dfLayer1 = dfLayer1.groupby(neuronTags).aggregate('sum').reset_index().copy()
	dfLayer1 = dfLayer1.astype('uint8')

	dfLayer2 = dfLayer2.groupby(neuronTags).aggregate('sum').reset_index().copy()
	dfLayer2 = dfLayer2.astype('uint8')

	dfLayer3 = dfLayer3.groupby(neuronTags).aggregate('sum').reset_index().copy()
	dfLayer3 = dfLayer3.astype('uint8')

	# Assign to each layer object
	accLayers[0].tt = dfLayer1
	accLayers[1].tt = dfLayer2
	accLayers[2].tt = dfLayer3

	# Some data over the activation performed

	print(f'In layer 1, there are a total of {len(dfLayer1)} input combinations from the {2**neuronPerLayer} possible')
	print(f'In layer 2, there are a total of {len(dfLayer2)} input combinations from the {2**neuronPerLayer} possible')
	print(f'In layer 3, there are a total of {len(dfLayer3)} input combinations from the {2**neuronPerLayer} possible')

	# Create the TT per neuron

	for layer in accLayers:
		layer.fillTT()

	# Retrieve distribution of importances
	for layer in accLayers:
		for neuron in layer.neurons:
			infoImportance = neuron.returnTTImportanceStats()
			importanceCharacteristics = pd.concat([importanceCharacteristics,
												   pd.DataFrame({'layer': infoImportance[0],
																 'name': infoImportance[1],
																 'numImportance': infoImportance[2],
																 'ttSize': infoImportance[3]}, index=[0])],
												  ignore_index=True)

	importanceCharacteristics.to_csv('./data/importanceCharacterization/' + str(neuronPerLayer) + 'neurons.csv')

