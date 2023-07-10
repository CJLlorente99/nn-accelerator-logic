import torch
from modelsCommon.auxTransformations import ToBlackAndWhite, ToSign
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize, RandomHorizontalFlip, RandomCrop
from torch.utils.data import DataLoader
from modules.vggSmall import VGGSmall
from modules.binaryVggVerySmall import binaryVGGVerySmall
from modules.binaryVggVerySmall2 import binaryVGGVerySmall2
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import os

modelName = 'binaryVGGVerySmall2_11110000_3'
# modelFilename = f'src\modelCreation\savedModels\{modelName}'
dataFolder = '/media/carlosl/CHAR/data'
resizeFactor = 3
relus = [1, 1, 1, 1, 0, 0, 0, 0]
# Check mps maybe if working in MacOS
device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
Importing CIFAR10 dataset
'''
print(f'DOWNLOAD DATASET\n')
train_dataset = datasets.CIFAR10(root=f'{dataFolder}', train=True, transform=Compose([
	RandomHorizontalFlip(),
	RandomCrop(32, 4),
	ToTensor(),
	Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
								 download=False)

'''
Load model
'''
model = binaryVGGVerySmall2(resizeFactor=resizeFactor, relus=relus)
# model.load_state_dict(torch.load(modelFilename))

"""
Process by chunks
"""
numberRowsPerChunk = 0.25 * 10 ** 3  # 250
gradientBaseFilename = f'{dataFolder}/gradients/{modelName}'
activationBaseFilename = f'{dataFolder}/activations/{modelName}'

# Create list for reader
readerFromHooks = {}
for i in model.helpHookList:
	readerFromHooks[i] = {'forward': None, 'backward': None}

# Load readers for activations
for grad in model.dataFromHooks:
	if grad.startswith('relul') or grad.startswith('stel'):
		readerFromHooks[grad]['forward'] = pd.read_csv(f'{activationBaseFilename}/{grad}', index_col=False, header=None, dtype=np.float16, chunksize=numberRowsPerChunk)
	else:
		readerFromHooks[grad]['forward'] = []
		for file in os.scandir(f'{activationBaseFilename}/{grad}'):
			readerFromHooks[grad]['forward'].append(pd.read_csv(f'{file.path}', index_col=False, header=None, dtype=np.float16, chunksize=numberRowsPerChunk))
	print(f'Activations reader from {grad} loaded')

# Load readers for gradients
for grad in model.dataFromHooks:
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
totalEntries = {}
# Initialize for 10 classes and layers
for grad in model.helpHookList:
	importancePerClassFilterGlobal[grad] = {}
	importancePerClassNeuronGlobal[grad] = {}
	lenImportanceAccum[grad] = {}
	entriesBelowThreshold[grad] = 0
	totalEntries[grad] = 0
	for i in range(10):
		importancePerClassFilterGlobal[grad][i] = []
		importancePerClassNeuronGlobal[grad][i] = []
		lenImportanceAccum[grad][i] = 0
while True:
	try:
		# Load activations
		for grad in model.dataFromHooks:
			if grad.startswith('relul') or grad.startswith('stel'):
				model.dataFromHooks[grad]['forward'] = next(readerFromHooks[grad]['forward']).to_numpy()
			else:
				model.dataFromHooks[grad]['forward'] = []
				i = 0
				for file in os.scandir(f'{activationBaseFilename}/{grad}'):
					model.dataFromHooks[grad]['forward'].append(next(readerFromHooks[grad]['forward'][i]).to_numpy())
					# print(f'Activations from {grad}_{i} loaded')
					# print(model.dataFromHooks[grad]['forward'][i].shape)
					i += 1
				model.dataFromHooks[grad]['forward'] = np.array(model.dataFromHooks[grad]['forward'])
				model.dataFromHooks[grad]['forward'] = np.moveaxis(model.dataFromHooks[grad]['forward'], 0, 1)
			print(f'Activations from {grad} loaded')
			print(model.dataFromHooks[grad]['forward'].shape)
		
		# Load gradients
		for grad in model.dataFromHooks:
			if grad.startswith('relul') or grad.startswith('stel'):
				model.dataFromHooks[grad]['backward'] = next(readerFromHooks[grad]['backward']).to_numpy()
			else:
				model.dataFromHooks[grad]['backward'] = []
				i = 0
				for file in os.scandir(f'{gradientBaseFilename}/{grad}'):
					model.dataFromHooks[grad]['backward'].append(next(readerFromHooks[grad]['backward'][i]).to_numpy())
					# print(f'Gradients from {grad}_{i} loaded')
					# print(model.dataFromHooks[grad]['backward'][i].shape)
					i += 1
				model.dataFromHooks[grad]['backward'] = np.array(model.dataFromHooks[grad]['backward'])
				model.dataFromHooks[grad]['backward'] = np.moveaxis(model.dataFromHooks[grad]['backward'], 0, 1)
			print(f'Gradients from {grad} loaded')
			print(model.dataFromHooks[grad]['backward'].shape)

		importances = model.computeImportance()

		'''
		Calculate importance scores and sum the entries below the threshold
		'''
		# threshold = 10e-50
		threshold = 10e-5  # acceptable when in presence of BN
		for i in range(len(importances)):
			importances[i] = (importances[i] > threshold)
			aux = importances[i].sum()
			if aux.ndim > 2:
				aux = aux.sum()
			print(f'Chunk {nChunk} layer {model.helpHookList[i]} has {aux} entries below threshold out of {importances[i].size}. Reduction {(1 - aux/importances[i].size)*100:.2f}%')
			entriesBelowThreshold[model.helpHookList[i]] += aux
			totalEntries[model.helpHookList[i]] += importances[i].size

		'''
		Calculate importance per class
		'''
		importancePerClassFilter = {}
		importancePerClassNeuron = {}
		# Initialize for 10 classes and layers
		for grad in model.helpHookList:
			importancePerClassFilter[grad] = {}
			for i in range(10):
				importancePerClassFilter[grad][i] = []

		# Iterate through the calculated importances and assign depending on the class
		imp = 0
		for grad in model.helpHookList:
			importance = importances[imp]
			for i in range(importance.shape[0]):
				importancePerClassFilter[grad][train_dataset.targets[i + targetsOffset]].append(importance[i])
			imp += 1
		targetsOffset += i+1  # Should have the last sample in the importance class characterization

			
		# Change from list to array each entry of importancePerClassFilter
		for grad in model.helpHookList:
			for nClass in importancePerClassFilter[grad]:
				importancePerClassFilter[grad][nClass] = np.array(importancePerClassFilter[grad][nClass])
		
		# Iterate through the importance scores and convert them into importance values
		for grad in model.helpHookList:
			for nClass in importancePerClassFilterGlobal[grad]:
				if importancePerClassFilter[grad][nClass].ndim > 2: # Then it comes from a conv layer
					importancePerClassFilterGlobal[grad][nClass].append(importancePerClassFilter[grad][nClass].sum(0))
					lenImportanceAccum[grad][nClass] += len(importancePerClassFilter[grad][nClass])
					importancePerClassNeuronGlobal[grad][nClass].append(importancePerClassFilter[grad][nClass].sum(0).flatten()) # To store information about all neurons
				else: # Then it comes from a neural layer
					importancePerClassFilterGlobal[grad][nClass].append(importancePerClassFilter[grad][nClass].sum(0))
					lenImportanceAccum[grad][nClass] += len(importancePerClassFilter[grad][nClass])
		print(f"Processed samples [{int((nChunk+1)*numberRowsPerChunk)}/50000]")
		nChunk += 1
		if nChunk == 10:
			break
	except StopIteration:  # No more chunks
		break

# List to array
for grad in model.helpHookList:
	for nClass in importancePerClassFilterGlobal[grad]:
		# print(np.array(importancePerClassFilterGlobal[grad][nClass]).shape)
		importancePerClassFilterGlobal[grad][nClass] = np.array(importancePerClassFilterGlobal[grad][nClass]).sum(0)
		# print(f'Filter list to array {grad}, class {nClass}')
		# print(importancePerClassFilterGlobal[grad][nClass].shape)
		if not(grad.startswith('stel') or grad.startswith('relul')):
			# print(np.array(importancePerClassNeuronGlobal[grad][nClass]).shape)
			importancePerClassNeuronGlobal[grad][nClass] = np.array(importancePerClassNeuronGlobal[grad][nClass]).sum(0)
			# print(f'Neuron list to array {grad}, class {nClass}')
			# print(importancePerClassNeuronGlobal[grad][nClass].shape)

# Actual calculation of importance
for grad in model.helpHookList:
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

# Print potential class-based and entry-based optimization
for grad in importancePerClassFilterGlobal:
	lenghts = np.array([lenImportanceAccum[grad][nClass] for nClass in lenImportanceAccum[grad]])
	if grad.startswith('relul') or grad.startswith('stel'):
		aux = (importancePerClassFilterGlobal[grad] > 0) * lenghts[:, np.newaxis]
		total = np.ones(importancePerClassFilterGlobal[grad].shape) * lenghts[:, np.newaxis]
		print(f'Total entries following the class-based optimization in {grad} are {aux.sum(dtype=np.int64)} out of all {total.sum(dtype=np.int64)}. Reduction {(1 - aux.sum().sum()/total.sum())*100:.2f}%')
	else:
		aux = (importancePerClassNeuronGlobal[grad] > 0) * lenghts[:, np.newaxis]
		total = np.ones(importancePerClassNeuronGlobal[grad].shape) * lenghts[:, np.newaxis]
		print(f'Total entries following the class-based optimization in {grad} are {aux.sum(dtype=np.int64)} out of all {total.sum(dtype=np.int64)}. Reduction {(1 - aux.sum()/total.sum())*100:.2f}%')
	print(f'Total entries following the entry-based optimization in {grad} are {entriesBelowThreshold[grad]} out of all {totalEntries[grad]}. Reduction {(1 - entriesBelowThreshold[grad]/totalEntries[grad])*100:.2f}%\n')

'''
Print results
'''
for grad in importancePerClassFilterGlobal:
	# Print aggregated importance
	aux = importancePerClassFilterGlobal[grad].sum(0)
	aux.sort()
 
	fig = go.Figure()
	fig.add_trace(go.Scatter(y=aux))
	fig.update_layout(title=f'{grad} total importance per filter ({len(aux)})')
	fig.show()
 
	# Print aggregated importance
	if not grad.startswith('relul') or grad.startswith('stel'):
		aux = importancePerClassNeuronGlobal[grad].sum(0)
		aux.sort()
	
		fig = go.Figure()
		fig.add_trace(go.Scatter(y=aux))
		fig.update_layout(title=f'{grad} total importance per neuron ({len(aux)})')
		fig.show()
	
	
	# Print classes that are important
	if grad.startswith('relul') or grad.startswith('stel'):
		aux = (importancePerClassFilterGlobal[grad] > 0).sum(0)
		aux.sort()
		
		fig = go.Figure()
		fig.add_trace(go.Scatter(y=aux))
		fig.update_layout(title=f'{grad} number of important classes per neuron ({len(aux)})')
		fig.show()
	else:
		aux = (importancePerClassNeuronGlobal[grad] > 0).sum(0)
		aux.sort()
		
		fig = go.Figure()
		fig.add_trace(go.Scatter(y=aux))
		fig.update_layout(title=f'{grad} number of important classes per neuron ({len(aux)})')
		fig.show()
	