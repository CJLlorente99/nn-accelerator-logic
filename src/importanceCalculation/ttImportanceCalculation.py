import random
import pandas as pd
import torch
from modelsCommon.auxTransformations import *
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize, RandomHorizontalFlip, RandomCrop, Resize
from torch.utils.data import DataLoader
from modules.binaryVggVerySmall import binaryVGGVerySmall
from modules.vggVerySmall import VGGVerySmall
from modules.vggSmall import VGGSmall
from ttUtilities.helpLayerNeuronGenerator import HelpGenerator
from ttUtilities.auxFunctions import integerToBinaryArray
import numpy as np
import os

modelName = f'binaryVGGVerySmall'
batch_size = 64
nClasses = 10

# Check mps maybe if working in MacOS
device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
Importing MNIST dataset
'''
print(f'DOWNLOAD DATASET\n')
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=Compose([
        RandomHorizontalFlip(),
        RandomCrop(32, 4),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        Resize(64, antialias=False)]),
    download=False)

'''
Create DataLoader
'''
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

model = binaryVGGVerySmall()
model.load_state_dict(torch.load(f'./src/modelCreation/savedModels/{modelName}'))

'''
Load activations
'''
model.loadActivations(f'data/activations\{modelName}')
model.loadGradients(f'data/gradients\{modelName}')
importances = model.computeImportance()

'''
Calculate importance scores
'''

print('APPLY THRESHOLD \n')
threshold = 10e-50
for i in range(len(importances)):
	importances[i] = (importances[i] > threshold)
 
importancePerClassFilter = {}
importancePerClassNeuron = {}
# Initialize for 10 classes and layers
for grad in model.helpHookList:
	importancePerClassFilter[grad] = {}
	importancePerClassNeuron[grad] = {}
	for i in range(10):
		importancePerClassFilter[grad][i] = []
		importancePerClassNeuron[grad][i] = {}
 
# Iterate through the calculated importances and assign depending on the class
print('ASSIGN TO CLASSES \n')
imp = 0
for grad in model.helpHookList:
	importance = importances[imp]
	if importance.ndim > 2:
		for i in range(importance.shape[1]):
			importancePerClassFilter[grad][train_dataset.targets[i]].append(importance[i, :, :])
	else:
		for i in range(importance.shape[0]):
			importancePerClassFilter[grad][train_dataset.targets[i]].append(importance[i, :])
	imp += 1
 
 # Change from list to array each entry of importancePerClassFilter
print('CLASSES LIST TO ARRAY \n')
for grad in model.helpHookList:
	for nClass in importancePerClassFilter[grad]:
		importancePerClassFilter[grad][nClass] = np.array(importancePerClassFilter[grad][nClass])
 
 # Iterate through the importance scores and convert them into importance values
print('CALCULATE IMPORTANCE VALUES \n')
for grad in model.helpHookList:
	for nClass in importancePerClassFilter[grad]:
		if importancePerClassFilter[grad][nClass].ndim > 2: # Then it comes from a conv layer
			importancePerClassFilter[grad][nClass] = importancePerClassFilter[grad][nClass].sum(0) / len(importancePerClassFilter[grad][nClass])
			importancePerClassNeuron[grad][nClass] = importancePerClassFilter[grad][nClass].flatten() # To store information about all neurons
			# Take the max score per filter
			importancePerClassFilter[grad][nClass] = importancePerClassFilter[grad][nClass].max(1)
		else: # Then it comes from a neural layer
			importancePerClassFilter[grad][nClass] = importancePerClassFilter[grad][nClass].sum(0) / len(importancePerClassFilter[grad][nClass])
	
# Join the lists so each row is a class and each column a filter/neuron
print('JOIN CLASS-IMPORTANCE \n')
for grad in importancePerClassFilter:
	importancePerClassFilter[grad] = np.row_stack(tuple(importancePerClassFilter[grad].values()))
	importancePerClassNeuron[grad] = np.row_stack(tuple(importancePerClassNeuron[grad].values()))

'''
Loop over the possibly optimized layers. Create and optimize the TT
'''

previousGrad = ''
iImportance = 0
for grad in model.helpHookList:
	if grad.startswith('ste') and previousGrad.startswith('ste'): # Input and output binary conv to conv  
		# If conv layer, change to (samples, filters)
		if not previousGrad.startswith('stel'):
		# TODO reduce the third dimension (samples, filters)
			inputToLayer = model.dataFromHooks[previousGrad]['forward'].reshape(-1, model.dataFromHooks[previousGrad]['forward'].shape[0])
		if not grad.startswith('stel'):
			outputToLayer = model.dataFromHooks[grad]['forward'].flatten()
			imp = importances[iImportance].flatten()
  
		columnTags = [f'F{i}' for i in range(inputToLayer.shape[1])]
		for iFilter in outputToLayer.shape[1]: # loop through the filters
			samples = []
			for iSample in inputToLayer.shape[0]: # loop through the samples
				# Standard way to reduce the truth table (looking which class is related to the entry and noting if it's important)
				if importancePerClassFilter[grad][train_dataset.targets[iSample]][iFilter] != 0: # Then add entry
					samples.append(iSample)
				# Alternative way to reduce the truth table (looking if the entry was impotant)
				# if imp[iSample, iFilter] != 0: # Then add entry
				# 	samples.append(iSample)
			df = pd.DataFrame(inputToLayer[samples, :], columns=columnTags)
			df[f'output{grad}_{iFilter}'] = outputToLayer[samples, iFilter]
			# Check folder exists
			if not os.path.exists(f'data/optimizedTT/{modelName}/{grad}'):
				os.makedirs(f'data/optimizedTT/{modelName}/{grad}')
			df.to_feather(f'data/optimizedTT/{modelName}/{grad}/F{iFilter}')
	previousGrad = grad	
	iImportance += 1
  
