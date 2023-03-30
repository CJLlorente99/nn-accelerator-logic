import pandas as pd
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from models.binaryNN import BinaryNeuralNetwork
import torch.nn.functional as F
import torch.optim as optim
from ttUtilities.ttGenerator import TTGenerator
from torch.autograd import Variable
import numpy as np

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
batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

'''
Train and test functions
'''


def train(dataloader, model, optimizer):
	size = len(dataloader.dataset)
	model.train()
	for batch, (X, y) in enumerate(dataloader):
		X, y = X.to(device), y.to(device)

		# Compute prediction error
		pred = model(X)
		loss = F.nll_loss(pred, y)

		# Backpropagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if batch % 100 == 0:
			loss, current = loss.item(), (batch + 1) * len(X)
			print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model):
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	model.eval()
	test_loss, correct = 0, 0
	with torch.no_grad():
		for X, y in dataloader:
			X, y = X.to(device), y.to(device)
			pred = model(X)
			test_loss += F.nll_loss(pred, y).item()
			correct += (pred.argmax(1) == y).type(torch.float).sum().item()
	test_loss /= num_batches
	correct /= size
	print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


nClasses = len(training_data.class_to_idx)
tryNeurons = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
nHiddenLayers = 3

for neuronPerLayer in tryNeurons:
	importanceCharacteristics = pd.DataFrame()
	print(f'Trying NN with {neuronPerLayer} neurons per layer')

	binaryNNModel = BinaryNeuralNetwork(neuronPerLayer).to(device)
	'''
	Train and test
	'''
	print(f'TRAINING\n')

	optBinary = optim.Adamax(binaryNNModel.parameters(), lr=3e-3, weight_decay=1e-4)

	epochs = 5

	print('Train and test binary NN')
	for t in range(epochs):
		print(f"Epoch {t+1}\n-------------------------------")
		train(train_dataloader, binaryNNModel, optBinary)
		test(test_dataloader, binaryNNModel)

	'''
	Generate TT
	'''

	accLayers = TTGenerator.getAccLayers(binaryNNModel)
	accLayers.pop(0)  # Pop first element
	accLayers.pop()  # Pop last element
	neurons = TTGenerator.getNeurons(accLayers)

	# Assign list of neurons to each layer
	i = 0
	for layer in neurons:
		accLayers[i] = neurons[layer]
		i += 1

	'''
	Calculate importance per class per neuron
	'''

	# Create backward hook to get gradients
	gradientsSTE1 = []
	gradientsSTE2 = []
	gradientsSTE3 = []


	def backward_hook_ste1(module, grad_input, grad_output):
		gradientsSTE1.append(grad_input[0].cpu().detach().numpy()[0])


	def backward_hook_ste2(module, grad_input, grad_output):
		gradientsSTE2.append(grad_input[0].cpu().detach().numpy()[0])


	def backward_hook_ste3(module, grad_input, grad_output):
		gradientsSTE3.append(grad_input[0].cpu().detach().numpy()[0])


	# Create forward hook to get values per neuron
	valueSTE1 = []
	valueSTE2 = []
	valueSTE3 = []


	def forward_hook_ste1(module, val_input, val_output):
		valueSTE1.append(val_output[0].cpu().detach().numpy())


	def forward_hook_ste2(module, val_input, val_output):
		valueSTE2.append(val_output[0].cpu().detach().numpy())


	def forward_hook_ste3(module, val_input, val_output):
		valueSTE3.append(val_output[0].cpu().detach().numpy())


	# Register hooks

	binaryNNModel.ste1.register_full_backward_hook(backward_hook_ste1)
	binaryNNModel.ste2.register_full_backward_hook(backward_hook_ste2)
	binaryNNModel.ste3.register_full_backward_hook(backward_hook_ste3)

	binaryNNModel.ste1.register_forward_hook(forward_hook_ste1)
	binaryNNModel.ste2.register_forward_hook(forward_hook_ste2)
	binaryNNModel.ste3.register_forward_hook(forward_hook_ste3)

	# Input samples and get gradients and values in each neuron
	print(f'GET GRADIENTS AND ACTIVATION VALUES\n')

	sampleSize = int(0.5 * len(training_data.data))  # sample size to be used for importance calculation
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

	gradientsSTE1 = np.array(gradientsSTE1).squeeze().reshape(len(gradientsSTE1), neuronPerLayer)
	gradientsSTE2 = np.array(gradientsSTE2).squeeze().reshape(len(gradientsSTE2), neuronPerLayer)
	gradientsSTE3 = np.array(gradientsSTE3).squeeze().reshape(len(gradientsSTE3), neuronPerLayer)

	valueSTE1 = np.array(valueSTE1).squeeze().reshape(len(valueSTE1), neuronPerLayer)
	valueSTE2 = np.array(valueSTE2).squeeze().reshape(len(valueSTE2), neuronPerLayer)
	valueSTE3 = np.array(valueSTE3).squeeze().reshape(len(valueSTE3), neuronPerLayer)

	importanceSTE1 = abs(np.multiply(gradientsSTE1, valueSTE1))
	importanceSTE2 = abs(np.multiply(gradientsSTE2, valueSTE2))
	importanceSTE3 = abs(np.multiply(gradientsSTE3, valueSTE3))

	# Give each neuron its importance values
	for i in range(neuronPerLayer):
		neurons['layer1'][i].giveImportance(importanceSTE1[:, i], training_data.targets.tolist())
		neurons['layer2'][i].giveImportance(importanceSTE2[:, i], training_data.targets.tolist())
		neurons['layer3'][i].giveImportance(importanceSTE3[:, i], training_data.targets.tolist())

	# Get activations-class per layer
	neuronTags = ['n' + str(i) for i in range(neuronPerLayer)]
	classTags = ['class' + str(i) for i in range(nClasses)]

	valueSTE1 = np.hstack((valueSTE1, training_data.targets[:sampleSize].detach().numpy().reshape((sampleSize, 1))))
	valueSTE1 = np.hstack((valueSTE1, np.zeros((sampleSize, nClasses))))
	valueSTE2 = np.hstack((valueSTE2, training_data.targets[:sampleSize].detach().numpy().reshape((sampleSize, 1))))
	valueSTE2 = np.hstack((valueSTE2, np.zeros((sampleSize, nClasses))))
	valueSTE3 = np.hstack((valueSTE3, training_data.targets[:sampleSize].detach().numpy().reshape((sampleSize, 1))))
	valueSTE3 = np.hstack((valueSTE3, np.zeros((sampleSize, nClasses))))

	dfLayer1 = pd.DataFrame(valueSTE1, columns=neuronTags + ['class'] + ['class' + str(n) for n in range(nClasses)])
	dfLayer2 = pd.DataFrame(valueSTE2, columns=neuronTags + ['class'] + ['class' + str(n) for n in range(nClasses)])
	dfLayer3 = pd.DataFrame(valueSTE3, columns=neuronTags + ['class'] + ['class' + str(n) for n in range(nClasses)])

	for index, row in dfLayer1.iterrows():
		row['class' + str(int(row['class']))] = 1
	dfLayer1 = dfLayer1.drop(['class'], axis=1)
	for index, row in dfLayer2.iterrows():
		row['class' + str(int(row['class']))] = 1
	dfLayer2 = dfLayer2.drop(['class'], axis=1)
	for index, row in dfLayer3.iterrows():
		row['class' + str(int(row['class']))] = 1
	dfLayer3 = dfLayer3.drop(['class'], axis=1)

	# Group by neuron activations and sum class columns
	dfLayer1 = dfLayer1.groupby(neuronTags).aggregate('sum').reset_index()
	dfLayer1 = dfLayer1.astype('uint8')
	dfLayer2 = dfLayer2.groupby(neuronTags).aggregate('sum').reset_index()
	dfLayer2 = dfLayer2.astype('uint8')
	dfLayer3 = dfLayer3.groupby(neuronTags).aggregate('sum').reset_index()
	dfLayer3 = dfLayer3.astype('uint8')

	# Some data over the activation performed

	print(f'In layer 1, there are a total of {len(dfLayer1)} input combinations from the {2**neuronPerLayer} possible')
	print(f'In layer 2, there are a total of {len(dfLayer2)} input combinations from the {2**neuronPerLayer} possible')
	print(f'In layer 3, there are a total of {len(dfLayer3)} input combinations from the {2**neuronPerLayer} possible')

	# Create the TT per neuron

	i = 0
	for layer in neurons:
		for neuron in neurons[layer]:
			i += 1
			if (i + 1) % 25 == 0:
				print(f"Neuron TT [{i + 1:>4d}/{neuronPerLayer * nHiddenLayers:>4d}]")

			if layer == 'layer1':
				neuron.createTT(dfLayer1)
			elif layer == 'layer2':
				neuron.createTT(dfLayer2)
			elif layer == 'layer3':
				neuron.createTT(dfLayer3)

			# Retrieve distribution of importances
			infoImportance = neuron.returnTTImportanceStats()
			importanceCharacteristics = pd.concat([importanceCharacteristics,
												   pd.DataFrame({'layer': infoImportance[0],
																 'name': infoImportance[1],
																 'numImportance': infoImportance[2],
																 'ttSize': infoImportance[3]}, index=[0])],
												  ignore_index=True)

	importanceCharacteristics.to_csv('./data/importanceCharacterization/' + str(neuronPerLayer) + 'neurons.csv')

