import pandas as pd
import torch
from modelsCommon.auxTransformations import *
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize, RandomHorizontalFlip, RandomCrop
from torch.utils.data import DataLoader
from modules.binaryVGGSmall import VGGSmall
import plotly.graph_objects as go
import numpy as np

neuronPerLayer = 4096
modelFilename = f'src\modelCreation\savedModels\MNISTSignbinNN50Epoch4096NPLhingeCriterion'
batch_size = 64
perGradientSampling = 1
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
Create DataLoader
'''
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

sampleSize = int(perGradientSampling * len(train_dataset.data))  # sample size to be used for importance calculation

'''
Load model
'''
model = VGGSmall()
model.load_state_dict(torch.load(modelFilename))

'''
Load gradients previously calculated
'''
model.loadGradients('./data/gradients/vggSmall')
model.loadActivations('./data/gradients/vggSmall')

'''
Calculate importance values
'''
importances = model.computeImportance()

'''
Calculate importance scores
'''
threshold = 10e-50
for importance in importances:
	importance = (importance > threshold)

'''
Calculate importance per class
'''
importancePerClass = {}
# Initialize for 10 classes and layers
for grad in model.helpHookList:
	importancePerClass[grad] = {}
	for i in range(10):
		importancePerClass[grad][i] = None

# Iterate through the calculated importances and assign depending on the class
imp = 0
for grad in model.helpHookList:
	importance = importances[imp]
	for i in range(importance.shape[0]):
		importancePerClass[grad][train_dataset.targets[i]] = importance.sum(1)
	imp += 1
	
# Join the lists so each row is a class and each column a filter/neuron
for grad in importancePerClass:
	importancePerClass[grad] = np.row_stack(tuple(importancePerClass[grad].values()))
	
'''
Print results
'''
# Print aggregated importance
for grad in importancePerClass:
	aux = importancePerClass[grad].sum(1)
	
	fig = go.Figure()
	fig.add_trace(go.Bar(aux))
	fig.show()
	
# Print classes that are important
	aux = (importancePerClass[grad] > 0).sum(1)
	
	fig = go.Figure()
	fig.add_trace(go.Bar(aux))
	fig.show()
 
 # Print importance per class
	fig = go.Figure()
	fig.add_trace(go.Bar(importancePerClass[grad]))
	fig.show()
