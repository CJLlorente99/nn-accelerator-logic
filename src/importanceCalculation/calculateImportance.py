import torch
from modelsCommon.auxTransformations import ToBlackAndWhite, ToSign
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize, RandomHorizontalFlip, RandomCrop
from torch.utils.data import DataLoader
from modules.vggSmall import VGGSmall
import plotly.graph_objects as go
import numpy as np
import pandas as pd

modelFilename = f'src\modelCreation\savedModels\VGGSmall'
# Check mps maybe if working in MacOS
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
model = VGGSmall()
model.load_state_dict(torch.load(modelFilename))

model.loadActivations(f'data/activations/vggSmall')
model.loadGradients(f'data/gradients/vggSmall')
importances = model.computeImportance()

'''
Calculate importance scores
'''
threshold = 10e-50
for i in range(len(importances)):
	importances[i] = (importances[i] > threshold)

'''
Calculate importance per class
'''
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
imp = 0
for grad in model.helpHookList:
	importance = importances[imp]
	if importance.ndim > 2:
		for i in range(importance.shape[1]):
			importancePerClassFilter[grad][train_dataset.targets[i]].append(importance[:, i, :])
	else:
		for i in range(importance.shape[0]):
			importancePerClassFilter[grad][train_dataset.targets[i]].append(importance[i, :])
	imp += 1
 
 # Change from list to array each entry of importancePerClassFilter
for grad in model.helpHookList:
	for nClass in importancePerClassFilter[grad]:
		importancePerClassFilter[grad][nClass] = np.array(importancePerClassFilter[grad][nClass])
 
 # Iterate through the importance scores and convert them into importance values
for grad in model.helpHookList:
	for nClass in importancePerClassFilter[grad]:
		if importancePerClassFilter[grad][nClass].ndim > 2: # Then it comes from a conv layer
			importancePerClassFilter[grad][nClass] = importancePerClassFilter[grad][nClass].sum(0) / len(importancePerClassFilter[grad][nClass])
			importancePerClassNeuron[grad][nClass] = importancePerClassFilter[grad][nClass].flatten() # To store information about all neurons
			# Take the max score per filter
			importancePerClassFilter[grad][nClass] = importancePerClassFilter[grad][nClass].max(0)
		else: # Then it comes from a neural layer
			importancePerClassFilter[grad][nClass] = importancePerClassFilter[grad][nClass].sum(0) / len(importancePerClassFilter[grad][nClass])
	
# Join the lists so each row is a class and each column a filter/neuron
for grad in importancePerClassFilter:
	importancePerClassFilter[grad] = np.row_stack(tuple(importancePerClassFilter[grad].values()))
	importancePerClassNeuron[grad] = np.row_stack(tuple(importancePerClassNeuron[grad].values()))
	
'''
Print results
'''
for grad in importancePerClassFilter:
    # Print aggregated importance
	aux = importancePerClassFilter[grad].sum(0)
	aux.sort()
 
	fig = go.Figure()
	fig.add_trace(go.Scatter(y=aux))
	fig.update_layout(title=f'{grad} total importance per filter ({len(aux)})')
	fig.show()
 
	# Print aggregated importance
	if not grad.startswith('relul'):
		aux = importancePerClassNeuron[grad].sum(0)
		aux.sort()
	
		fig = go.Figure()
		fig.add_trace(go.Scatter(y=aux))
		fig.update_layout(title=f'{grad} total importance per neuron ({len(aux)})')
		fig.show()
	
	
	# Print classes that are important
	if grad.startswith('relul'):
		aux = (importancePerClassFilter[grad] > 0).sum(0)
		aux.sort()
		
		fig = go.Figure()
		fig.add_trace(go.Scatter(y=aux))
		fig.update_layout(title=f'{grad} number of important classes per neuron ({len(aux)})')
		fig.show()
	else:
		aux = (importancePerClassNeuron[grad] > 0).sum(0)
		aux.sort()
		
		fig = go.Figure()
		fig.add_trace(go.Scatter(y=aux))
		fig.update_layout(title=f'{grad} number of important classes per neuron ({len(aux)})')
		fig.show()
 
 	# Print importance per class
	# aux = pd.DataFrame(importancePerClassFilter[grad],
    #                 columns=[f'filter{i}' for i in range(importancePerClassFilter[grad].shape[1])])
	# fig = go.Figure()
	# for index, row in aux.iterrows():
	# 	fig.add_trace(go.Bar(name=f'class{index}', y=row, x=aux.columns))
	# fig.update_layout(title=f'{grad} importance per class', barmode='stack')
	# fig.show()
	pass

