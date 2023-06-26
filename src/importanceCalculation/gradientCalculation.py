import torch
from modelsCommon.auxTransformations import ToBlackAndWhite, ToSign
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize, RandomHorizontalFlip, RandomCrop, Resize
from torch.utils.data import DataLoader
from modules.vggSmall import VGGSmall
from modules.vggVerySmall import VGGVerySmall
from modules.binaryVggVerySmall import binaryVGGVerySmall
from modules.binaryVggVerySmall2 import binaryVGGVerySmall2
import plotly.graph_objects as go
import numpy as np
import pandas as pd

modelName = f'binaryVGGVerySmall2_11110000_3'
batch_size = 64
perGradientSampling = 1
resizeFactor = 3
model = binaryVGGVerySmall2(resizeFactor=resizeFactor, relus=[1, 1, 1, 1, 0, 0, 0, 0])
dataFolder = './data'
# Check mps maybe if working in MacOS
device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
Importing CIFAR10 dataset
'''
print(f'DOWNLOAD DATASET\n')
train_dataset = datasets.CIFAR10(root=dataFolder, train=True, transform=Compose([
	ToTensor(),
	Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
 	Resize(resizeFactor*32, antialias=False)]),
								 download=False)

'''
Create DataLoader
'''
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

sampleSize = int(perGradientSampling * len(train_dataset.data))  # sample size to be used for importance calculation

'''
Load model
'''
model.load_state_dict(torch.load(f'./src/modelCreation/savedModels/{modelName}'))

'''
Calculate importance per class per neuron
'''

# Input samples and get gradients and values in each neuron
print(f'GET GRADIENTS AND ACTIVATION VALUES\n')

model.registerHooks()
model.eval()

start = 30000
for i in range(start, sampleSize):
	X, y = train_dataloader.dataset[i]
	model.zero_grad()
	pred = model(X[None, :, :, :])
	pred[0, y].backward()

	if (i+1) % 10000 == 0:
		print(f"Appending new values [{i+1:>5d}/{sampleSize:>5d}]")
		model.listToArray()
		# model.saveActivations(f'{dataFolder}/activations/{modelName}')
		model.saveGradients(f'{dataFolder}/gradients/{modelName}')
		model.resetHookLists()

	if (i+1) % 500 == 0:
		print(f"Get Gradients and Activation Values [{i+1:>5d}/{sampleSize:>5d}]")

model.listToArray()
model.saveActivations(f'{dataFolder}/activations/{modelName}')
model.saveGradients(f'{dataFolder}/gradients/{modelName}')
model.resetHookLists()
