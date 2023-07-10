import torch
from modelsCommon.auxTransformations import ToBlackAndWhite, ToSign
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize, RandomHorizontalFlip, RandomCrop
from torch.utils.data import DataLoader
from modules.vggSmall import VGGSmall
from modules.binaryVggVerySmall import binaryVGGVerySmall
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import os

modelName = 'binaryVGGVerySmall_1000001_2'
modelFilename = f'src\modelCreation\savedModels\{modelName}'
outputOptmizedFolder = f'data/optimizedTT/{modelName}'
outputNonOptmizedFolder = f'data/nonOptimizedTT/{modelName}'
# Check mps maybe if working in MacOS
device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
Layers to synthesize (only binarized linear)
'''
synthLayers = {'stel0': 'ste41'}  # Layer and predecessor layer

'''
Importing CIFAR10 dataset
'''
print(f'DOWNLOAD DATASET\n')
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=Compose([
	RandomHorizontalFlip(),
	RandomCrop(32, 4),
	ToTensor(),
	Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
								 download=False)

'''
Load model
'''
model = binaryVGGVerySmall()
model.load_state_dict(torch.load(modelFilename))

"""
Process by chunks
"""
numberRowsPerChunk = 2.5 * 10 ** 3  # 2500
gradientBaseFilename = f'data/gradients/{modelName}'
activationBaseFilename = f'data/activations/{modelName}'

# Create list for reader
readerFromHooks = {}
for i in (list(synthLayers.keys()) + list(synthLayers.values())):
	readerFromHooks[i] = {'forward': None, 'backward': None}

# Load readers for activations
for grad in (list(synthLayers.keys()) + list(synthLayers.values())):
	if grad.startswith('relul') or grad.startswith('stel'):
		readerFromHooks[grad]['forward'] = pd.read_csv(f'{activationBaseFilename}/{grad}', index_col=False, header=None, dtype=np.float16, chunksize=numberRowsPerChunk)
	else:
		readerFromHooks[grad]['forward'] = []
		for file in os.scandir(f'{activationBaseFilename}/{grad}'):
			readerFromHooks[grad]['forward'].append(pd.read_csv(f'{file.path}', index_col=False, header=None, dtype=np.float16, chunksize=numberRowsPerChunk))
	print(f'Activations reader from {grad} loaded')

# Load readers for gradients
for grad in (list(synthLayers.keys()) + list(synthLayers.values())):
	if grad.startswith('relul') or grad.startswith('stel'):
		readerFromHooks[grad]['backward'] = pd.read_csv(f'{gradientBaseFilename}/{grad}', index_col=False, header=None, dtype=np.float16, chunksize=numberRowsPerChunk)
	else:
		readerFromHooks[grad]['backward'] = []
		for file in os.scandir(f'{gradientBaseFilename}/{grad}'):
			readerFromHooks[grad]['backward'].append(pd.read_csv(f'{file.path}', index_col=False, header=None, dtype=np.float16, chunksize=numberRowsPerChunk))
	print(f'Gradients reader from {grad} loaded')

# Iterate through the chunks
nChunk = 0
targetsOffset = 0
lenImportanceAccum = {}
importancePerClassFilterGlobal = {}
importancePerClassNeuronGlobal = {}
entriesBelowThreshold = {}
idxEntriesBelowThreshold = {}
totalEntries = {}
# Initialize for 10 classes and layers
for grad in (list(synthLayers.keys()) + list(synthLayers.values())):
	importancePerClassFilterGlobal[grad] = {}
	importancePerClassNeuronGlobal[grad] = {}
	lenImportanceAccum[grad] = {}
	entriesBelowThreshold[grad] = 0
	totalEntries[grad] = 0
	idxEntriesBelowThreshold[grad] = []
	for i in range(10):
		importancePerClassFilterGlobal[grad][i] = []
		importancePerClassNeuronGlobal[grad][i] = []
		lenImportanceAccum[grad][i] = 0

header = True
while True:
	try:
		# Load activations
		for grad in (list(synthLayers.keys()) + list(synthLayers.values())):
			if grad.startswith('relul') or grad.startswith('stel'):
				model.dataFromHooks[grad]['forward'] = next(readerFromHooks[grad]['forward']).to_numpy()
			else:
				model.dataFromHooks[grad]['forward'] = []
				i = 0
				for file in os.scandir(f'{activationBaseFilename}/{grad}'):
					model.dataFromHooks[grad]['forward'].append(next(readerFromHooks[grad]['forward'][i]).to_numpy())
					i += 1
				model.dataFromHooks[grad]['forward'] = np.array(model.dataFromHooks[grad]['forward'])
				model.dataFromHooks[grad]['forward'] = np.moveaxis(model.dataFromHooks[grad]['forward'], 0, 1)
			print(f'Activations from {grad} loaded')
			print(model.dataFromHooks[grad]['forward'].shape)
		
		# Load gradients
		for grad in list(synthLayers.keys()):
			if grad.startswith('relul') or grad.startswith('stel'):
				model.dataFromHooks[grad]['backward'] = next(readerFromHooks[grad]['backward']).to_numpy()
			else:
				model.dataFromHooks[grad]['backward'] = []
				i = 0
				for file in os.scandir(f'{gradientBaseFilename}/{grad}'):
					model.dataFromHooks[grad]['backward'].append(next(readerFromHooks[grad]['backward'][i]).to_numpy())
					i += 1
				model.dataFromHooks[grad]['backward'] = np.array(model.dataFromHooks[grad]['backward'])
				model.dataFromHooks[grad]['backward'] = np.moveaxis(model.dataFromHooks[grad]['backward'], 0, 1)
			print(f'Gradients from {grad} loaded')
			print(model.dataFromHooks[grad]['backward'].shape)

		for grad in list(synthLayers.keys()):
			importance = model.computeImportanceLayer(grad)

			'''
			Calculate importance scores and sum the entries below the threshold
			'''
			threshold = 10e-50
			importance = (importance > threshold)

			'''
			Generate file with entry-based criteria
			'''
			if not synthLayers[grad].startswith('stel'):
				inputs = model.dataFromHooks[synthLayers[grad]]['forward'].reshape(model.dataFromHooks[synthLayers[grad]]['forward'].shape[0], -1)  # this must be 2D (sample, neurons)
			else:
				inputs = model.dataFromHooks[synthLayers[grad]]['forward']
			outputs = model.dataFromHooks[grad]['forward']  # this must be 2D (sample, neurons)
			outputOptimizedEntries = np.where(importance > 0, outputs, -5)

			inputColumnTags = [f'IN{i}' for i in range(inputs.shape[1])]
			inputs = pd.DataFrame(inputs, columns=inputColumnTags)
			outputColumnTags = [f'OUT{i}' for i in range(outputs.shape[1])]
			outputs = pd.DataFrame(outputs, columns=outputColumnTags)
			outputOptimizedEntries = pd.DataFrame(outputOptimizedEntries, columns=outputColumnTags)

			nonOptimizedDf = pd.concat([inputs, outputs], axis=1)
			optimizedDf = pd.concat([inputs, outputOptimizedEntries], axis=1)

			mode = 'w' if header else 'a'
			nonOptimizedDf.to_csv(f'{outputNonOptmizedFolder}/{grad}.csv', mode=mode, header=header, index=False)
			optimizedDf.to_csv(f'{outputOptmizedFolder}/{grad}_entryBased.csv', mode=mode, header=header, index=False)

			'''
			Calculate importance per class
			'''
			importancePerClassFilter = {}
			importancePerClassNeuron = {}
			# Initialize for 10 classes and layers
			importancePerClassFilter[grad] = {}
			for i in range(10):
				importancePerClassFilter[grad][i] = []

			# Iterate through the calculated importances and assign depending on the class
			for i in range(importance.shape[0]):
				importancePerClassFilter[grad][train_dataset.targets[i + targetsOffset]].append(importance[i])
			targetsOffset += i+1  # Should have the last sample in the importance class characterization

				
			# Change from list to array each entry of importancePerClassFilter
			for nClass in importancePerClassFilter[grad]:
				importancePerClassFilter[grad][nClass] = np.array(importancePerClassFilter[grad][nClass])
			
			# Iterate through the importance scores and convert them into importance values
			for nClass in importancePerClassFilterGlobal[grad]:
				if importancePerClassFilter[grad][nClass].ndim > 2: # Then it comes from a conv layer
					importancePerClassFilterGlobal[grad][nClass].append(importancePerClassFilter[grad][nClass].sum(0))
					lenImportanceAccum[grad][nClass] += len(importancePerClassFilter[grad][nClass])
					importancePerClassNeuronGlobal[grad][nClass].append(importancePerClassFilter[grad][nClass].sum(0).flatten()) # To store information about all neurons
				else: # Then it comes from a neural layer
					importancePerClassFilterGlobal[grad][nClass].append(importancePerClassFilter[grad][nClass].sum(0))
					lenImportanceAccum[grad][nClass] += len(importancePerClassFilter[grad][nClass])
		header = False
		print(f"Processed entry-based and importance samples [{int((nChunk+1)*numberRowsPerChunk)}/50000]")
		nChunk += 1
	except StopIteration:  # No more chunks
		break

# List to array
for grad in list(synthLayers.keys()):
	for nClass in importancePerClassFilterGlobal[grad]:
		importancePerClassFilterGlobal[grad][nClass] = np.array(importancePerClassFilterGlobal[grad][nClass]).sum(0)
		if not(grad.startswith('stel') or grad.startswith('relul')):
			importancePerClassNeuronGlobal[grad][nClass] = np.array(importancePerClassNeuronGlobal[grad][nClass]).sum(0)

# Actual calculation of importance
for grad in list(synthLayers.keys()):
	for nClass in importancePerClassFilterGlobal[grad]:
		if not(grad.startswith('stel') or grad.startswith('relul')): # Then it comes from a conv layer
			importancePerClassFilterGlobal[grad][nClass] = importancePerClassFilterGlobal[grad][nClass] / lenImportanceAccum[grad][nClass]
			importancePerClassNeuronGlobal[grad][nClass] = importancePerClassNeuronGlobal[grad][nClass] / lenImportanceAccum[grad][nClass] # To store information about all neurons
			# Take the max score per filter
			importancePerClassFilterGlobal[grad][nClass] = importancePerClassFilterGlobal[grad][nClass].max(0)
		else: # Then it comes from a neural layer
			importancePerClassFilterGlobal[grad][nClass] = importancePerClassFilterGlobal[grad][nClass] / lenImportanceAccum[grad][nClass]

# Join the lists so each row is a class and each column a filter/neuron
for grad in importancePerClassFilterGlobal:
	importancePerClassFilterGlobal[grad] = np.row_stack(tuple(importancePerClassFilterGlobal[grad].values()))
	importancePerClassNeuronGlobal[grad] = np.row_stack(tuple(importancePerClassNeuronGlobal[grad].values()))

'''
Generate file with class-based criteria
'''
# Reload iterators
for grad in (list(synthLayers.keys()) + list(synthLayers.values())):
	if grad.startswith('relul') or grad.startswith('stel'):
		readerFromHooks[grad]['forward'] = pd.read_csv(f'{activationBaseFilename}/{grad}', index_col=False, header=None, dtype=np.float16, chunksize=numberRowsPerChunk)
	else:
		readerFromHooks[grad]['forward'] = []
		for file in os.scandir(f'{activationBaseFilename}/{grad}'):
			readerFromHooks[grad]['forward'].append(pd.read_csv(f'{file.path}', index_col=False, header=None, dtype=np.float16, chunksize=numberRowsPerChunk))
	print(f'Activations reader from {grad} loaded')

# Load readers for gradients
for grad in (list(synthLayers.keys()) + list(synthLayers.values())):
	if grad.startswith('relul') or grad.startswith('stel'):
		readerFromHooks[grad]['backward'] = pd.read_csv(f'{gradientBaseFilename}/{grad}', index_col=False, header=None, dtype=np.float16, chunksize=numberRowsPerChunk)
	else:
		readerFromHooks[grad]['backward'] = []
		for file in os.scandir(f'{gradientBaseFilename}/{grad}'):
			readerFromHooks[grad]['backward'].append(pd.read_csv(f'{file.path}', index_col=False, header=None, dtype=np.float16, chunksize=numberRowsPerChunk))
	print(f'Gradients reader from {grad} loaded')

header = True
while True:
	try:
		# Load activations
		for grad in (list(synthLayers.keys()) + list(synthLayers.values())):
			if grad.startswith('relul') or grad.startswith('stel'):
				model.dataFromHooks[grad]['forward'] = next(readerFromHooks[grad]['forward']).to_numpy()
			else:
				model.dataFromHooks[grad]['forward'] = []
				i = 0
				for file in os.scandir(f'{activationBaseFilename}/{grad}'):
					model.dataFromHooks[grad]['forward'].append(next(readerFromHooks[grad]['forward'][i]).to_numpy())
					i += 1
				model.dataFromHooks[grad]['forward'] = np.array(model.dataFromHooks[grad]['forward'])
				model.dataFromHooks[grad]['forward'] = np.moveaxis(model.dataFromHooks[grad]['forward'], 0, 1)
			print(f'Activations from {grad} loaded')
			print(model.dataFromHooks[grad]['forward'].shape)

		for grad in list(synthLayers.keys()):
			inputs = model.dataFromHooks[synthLayers[grad]]['forward']  # this must be 2D (sample, neurons)
			outputs = model.dataFromHooks[grad]['forward']  # this must be 2D (sample, neurons)

			inputColumnTags = [f'IN{i}' for i in range(inputs.shape[1])]
			inputs = pd.DataFrame(inputs, columns=inputColumnTags)

			# Iterate through the samples and check if the class is important
			importantClasses = (importancePerClassFilterGlobal[grad] > 0)  # (nClass, neurons)
			optimizedOutput = []
			for i in len(inputs):
				optimizedOutput.append(np.where(importantClasses[train_dataset.targets[i + targetsOffset]], outputs[i], -5))
			targetsOffset += i+1  # Should have the last sample in the importance class characterization

			# Generate file
			outputColumnTags = [f'OUT{i}' for i in range(outputs.shape[1])]
			outputsOptimized = pd.DataFrame(np.array(optimizedOutput), columns=outputColumnTags)

			optimizedDf = pd.concatenate([inputs, outputsOptimized], axis=1)
			mode = 'w' if header else 'a'
			optimizedDf.to_csv(f'{outputOptmizedFolder}/{grad}_classBased.csv', mode=mode, header=header, index=False)

		header = False
		print(f"Processed class-based optimization samples [{int((nChunk+1)*numberRowsPerChunk)}/50000]")
		nChunk += 1
	except StopIteration:  # No more chunks
		break
	