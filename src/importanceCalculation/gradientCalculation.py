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
Calculate importance per class per neuron
'''

# Input samples and get gradients and values in each neuron
print(f'GET GRADIENTS AND ACTIVATION VALUES\n')

model.registerHooks()
model.eval()

for i in range(sampleSize):
	X, y = train_dataloader.dataset[i]
	model.zero_grad()
	pred = model(X)
	pred[0, y].backward()

	if (i+1) % 500 == 0:
		print(f"Get Gradients and Activation Values [{i+1:>5d}/{sampleSize:>5d}]")

model.listToArray()  # Hopefully improves memory usage

importances = model.computeImportance()
# model.saveActivations(f'data/activations/vggSmall')
# model.saveGradients(f'data/gradients/vggSmall')

'''
Calculate importance scores
'''
threshold = 10e-50
for i in range(len(importances)):
	importances[i] = (importances[i] > threshold)

'''
Calculate importance per class
'''
importancePerClass = {}
# Initialize for 10 classes and layers
for grad in model.helpHookList:
	importancePerClass[grad] = {}
	for i in range(10):
		importancePerClass[grad][i] = []

# Iterate through the calculated importances and assign depending on the class
imp = 0
for grad in model.helpHookList:
	importance = importances[imp]
	if importance.ndim > 2:
		for i in range(importance.shape[1]):
			importancePerClass[grad][train_dataset.targets[i]].append(importance[:, i, :])
	else:
		for i in range(importance.shape[0]):
			importancePerClass[grad][train_dataset.targets[i]].append(importance[i, :])
	imp += 1
 
 # Change from list to array each entry of importancePerClass
for grad in model.helpHookList:
	for nClass in importancePerClass[grad]:
		importancePerClass[grad][nClass] = np.array(importancePerClass[grad][nClass])
 
 # Iterate through the importance scores and convert them into importance values
for grad in model.helpHookList:
	for nClass in importancePerClass[grad]:
		if importancePerClass[grad][nClass].ndim > 2: # Then it comes from a conv layer
			importancePerClass[grad][nClass] = importancePerClass[grad][nClass].sum(0) / len(importancePerClass[grad][nClass])
			# Take the max score
			importancePerClass[grad][nClass] = importancePerClass[grad][nClass].max(0)
		else: # Then it comes from a neural layer
			importancePerClass[grad][nClass] = importancePerClass[grad][nClass].sum(0) / len(importancePerClass[grad][nClass])
	
# Join the lists so each row is a class and each column a filter/neuron
for grad in importancePerClass:
	importancePerClass[grad] = np.row_stack(tuple(importancePerClass[grad].values()))
	
'''
Print results
'''
for grad in importancePerClass:
    # Print aggregated importance
	aux = importancePerClass[grad].sum(0)
	aux.sort()	
 
	fig = go.Figure()
	fig.add_trace(go.Scatter(y=aux))
	fig.update_layout(title=f'{grad} total importance')
	fig.show()
	
	# Print classes that are important
	aux = (importancePerClass[grad] > 0).sum(0)
	aux.sort()
	
	fig = go.Figure()
	fig.add_trace(go.Scatter(y=aux))
	fig.update_layout(title=f'{grad} number of important classes')
	fig.show()
 
 	# Print importance per class
	aux = pd.DataFrame(importancePerClass[grad],
                    columns=[f'filter{i}' for i in range(importancePerClass[grad].shape[1])])
	fig = go.Figure()
	for index, row in aux.iterrows():
		fig.add_trace(go.Bar(name=f'class{index}', y=row, x=aux.columns))
	fig.update_layout(title=f'{grad} importance per class', barmode='stack')
	fig.show()
	pass


