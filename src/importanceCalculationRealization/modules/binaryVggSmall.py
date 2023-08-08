import pandas as pd
from modelsCommon.steFunction import STEFunction
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from ttUtilities.auxFunctions import binaryArrayToSingleValue, integerToBinaryArray
import math
import os
from modelsCommon.customPruning import random_pruning_per_neuron

class binaryVGGSmall(nn.Module):
	def __init__(self, resizeFactor, relus, connectionsAfterPrune=0):
		super(binaryVGGSmall, self).__init__()
  
		self.helpHookList = []

		# Layer 0
		self.conv0 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
		self.bn0 = nn.BatchNorm2d(64)
		if relus[0]:
			self.relu0 = nn.ReLU()
		else:
			self.relu0 = STEFunction()
		self.maxpool0 = nn.MaxPool2d(kernel_size=2, stride=2)

		# Layer 1
		self.conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm2d(128)
		if relus[1]:
			self.relu1 = nn.ReLU()
		else:
			self.relu1 = STEFunction()
		self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

		# Layer 2.1
		self.conv21 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
		self.bn21 = nn.BatchNorm2d(256)
		if relus[2]:
			self.relu21 = nn.ReLU()
		else:
			self.relu21 = STEFunction()

		# Layer 2.2
		self.conv22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.bn22 = nn.BatchNorm2d(256)
		if relus[3]:
			self.relu22 = nn.ReLU()
		else:
			self.relu22 = STEFunction()
		self.maxpool22 = nn.MaxPool2d(kernel_size=2, stride=2)

		# Layer 3.1
		self.conv31 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
		self.bn31 = nn.BatchNorm2d(512)
		if relus[4]:
			self.relu31 = nn.ReLU()
		else:
			self.relu31 = STEFunction()

		# Layer 3.2
		self.conv32 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.bn32 = nn.BatchNorm2d(512)
		if relus[5]:
			self.relu32 = nn.ReLU()
		else:
			self.relu32 = STEFunction()
		self.maxpool32 = nn.MaxPool2d(kernel_size=2, stride=2)

		# Layer 4.1
		self.conv41 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.bn41 = nn.BatchNorm2d(512)
		if relus[6]:
			self.relu41 = nn.ReLU()
		else:
			self.relu41 = STEFunction()

		# Layer 4.2
		self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.bn42 = nn.BatchNorm2d(512)
		if relus[7]:
			self.relu42 = nn.ReLU()
		else:
			self.relu42 = STEFunction()
		self.maxpool42 = nn.MaxPool2d(kernel_size=2, stride=2)

		# Layer FC0
		self.dropoutl0 = nn.Dropout(0.5)
		self.l0 = nn.Linear(resizeFactor*resizeFactor*512, 4096)
		self.bnl0 = nn.BatchNorm1d(4096)
		if relus[8]:
			self.relul0 = nn.ReLU()
		else:
			self.relul0 = STEFunction()

		# Layer FC1
		self.dropoutl1 = nn.Dropout(0.5)
		self.l1 = nn.Linear(4096, 4096)
		self.bnl1 = nn.BatchNorm1d(4096)
		if relus[9]:
			self.relul1 = nn.ReLU()
		else:
			self.relul1 = STEFunction()

		# Layer FC2
		self.dropoutl2 = nn.Dropout(0.5)
		self.l2 = nn.Linear(4096, 1000)
		self.bnl2 = nn.BatchNorm1d(1000)
		if relus[10]:
			self.relul2 = nn.ReLU()
		else:
			self.relul2 = STEFunction()

		# Layer FC3
		self.l3 = nn.Linear(1000, 10)

		# Regular pruning
		if connectionsAfterPrune != 0:
			self.l0 = random_pruning_per_neuron(self.l0, name="weight", connectionsToPrune=resizeFactor*resizeFactor*512-connectionsAfterPrune)
			self.l1 = random_pruning_per_neuron(self.l1, name="weight", connectionsToPrune=4096-connectionsAfterPrune)
			self.l2 = random_pruning_per_neuron(self.l2, name="weight", connectionsToPrune=4096-connectionsAfterPrune)
  
		# Initialize
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				m.bias.data.zero_()

		# Lists for hook data
		self.gradientsSTEL0 = []
		self.gradientsSTEL1 = []
		self.gradientsSTEL2 = []

		self.valueSTE42 = []
		self.valueSTEL0 = []
		self.valueSTEL1 = []
		self.valueSTEL2 = []
    

	def forward(self, x):
		# Layer 0
		x = self.conv0(x)
		x = self.bn0(x)
		x = self.relu0(x)
		x = self.maxpool0(x)

		# Layer 1
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu1(x)
		x = self.maxpool1(x)
  
		# Layer 2.1
		x = self.conv21(x)
		x = self.bn21(x)
		x = self.relu21(x)
  
		# Layer 2.2
		x = self.conv22(x)
		x = self.bn22(x)
		x = self.relu22(x)
		x = self.maxpool22(x)
  
		# Layer 3.1
		x = self.conv31(x)
		x = self.bn31(x)
		x = self.relu31(x)

		# Layer 3.2
		x = self.conv32(x)
		x = self.bn32(x)
		x = self.relu32(x)
		x = self.maxpool32(x)
  
		# Layer 4.1
		x = self.conv41(x)
		x = self.bn41(x)
		x = self.relu41(x)
  
		# Layer 4.2
		x = self.conv42(x)
		x = self.bn42(x)
		x = self.relu42(x)
		x = self.maxpool42(x)
  
		x = x.reshape(x.size(0), -1)
  
		# Layer FC0
		x = self.dropoutl0(x)
		x = self.l0(x)
		x = self.bnl0(x)
		x = self.relul0(x)
  
		# Layer FC1
		x = self.dropoutl1(x)
		x = self.l1(x)
		x = self.bnl1(x)
		x = self.relul1(x)

		# Layer FC1
		x = self.dropoutl2(x)
		x = self.l2(x)
		x = self.bnl2(x)
		x = self.relul2(x)
  
		# Layer FC2
		x = self.l3(x)

		return x
	
	# Probably there exists a better way to do this
	def registerHooks(self, activations: bool = True, gradients: bool = True):
		# Forward hooks are needed to compute importance
		if activations:
			self.relu42.register_forward_hook(self.forward_hook_relu41)
			self.relul0.register_forward_hook(self.forward_hook_relul0)
			self.relul1.register_forward_hook(self.forward_hook_relul1)
			self.relul2.register_forward_hook(self.forward_hook_relul2)
  
		# Backward hooks are needed to compute importance
		if gradients:
			self.relul0.register_full_backward_hook(self.backward_hook_relul0)
			self.relul1.register_full_backward_hook(self.backward_hook_relul1)
			self.relul2.register_full_backward_hook(self.backward_hook_relul2)

	# Define all backward hooks  
	def backward_hook_relul0(self, module, grad_input, grad_output):
		self.gradientsSTEL0.append(np.float16(grad_output[0].cpu().detach().numpy()[0]))
  
	def backward_hook_relul1(self, module, grad_input, grad_output):
		self.gradientsSTEL1.append(np.float16(grad_output[0].cpu().detach().numpy()[0]))

	def backward_hook_relul2(self, module, grad_input, grad_output):
		self.gradientsSTEL2.append(np.float16(grad_output[0].cpu().detach().numpy()[0]))
  
	# Define all forward hooks  
	def forward_hook_relu41(self, module, val_input, val_output):
		aux = val_output.flatten(start_dim=-2, end_dim=-1).cpu().detach().numpy()[0]
		self.valueSTE42.append(np.float16(aux))
  
	def forward_hook_relul0(self, module, val_input, val_output):
		self.valueSTEL0.append(np.float16(val_output[0].cpu().detach().numpy()))
  
	def forward_hook_relul1(self, module, val_input, val_output):
		self.valueSTEL1.append(np.float16(val_output[0].cpu().detach().numpy()))

	def forward_hook_relul2(self, module, val_input, val_output):
		self.valueSTEL2.append(np.float16(val_output[0].cpu().detach().numpy()))
  
	# Change each hook list to an equivalent array
	def listToArray(self):
		self.valueSTE42 = np.array(self.valueSTE42)
		self.valueSTEL0 = np.array(self.valueSTEL0)
		self.valueSTEL1 = np.array(self.valueSTEL1)
		self.valueSTEL2 = np.array(self.valueSTEL2)

		self.gradientsSTEL0 = np.array(self.gradientsSTEL0)
		self.gradientsSTEL1 = np.array(self.gradientsSTEL1)
		self.gradientsSTEL2 = np.array(self.gradientsSTEL2)
   
	# Compute importance
	def computeImportance(self):
		importanceSTEL0 = np.abs(self.gradientsSTEL0)
		importanceSTEL1 = np.abs(self.gradientsSTEL1)
		importanceSTEL2 = np.abs(self.gradientsSTEL2)
	
		return [importanceSTEL0, importanceSTEL1, importanceSTEL2]	
    
    # Save activations
	def saveActivations(self, baseFilename):
		columnsInLayer42 = [f'N{i}' for i in range(len(self.valueSTE42[0]))]
		columnsInLayerL0 = [f'N{i}' for i in range(len(self.valueSTEL0[0]))]
		columnsInLayerL1 = [f'N{i}' for i in range(len(self.valueSTEL1[0]))]
		columnsInLayerL2 = [f'N{i}' for i in range(len(self.valueSTEL2[0]))]

		pd.DataFrame(
			self.valueSTE42, columns=columnsInLayer42).to_feather(
			f'{baseFilename}Input42')

		pd.DataFrame(
			self.valueSTEL0, columns=columnsInLayerL0).to_feather(
			f'{baseFilename}InputL0')

		pd.DataFrame(
			self.valueSTEL1, columns=columnsInLayerL1).to_feather(
			f'{baseFilename}InputL1')

		pd.DataFrame(
			self.valueSTEL2, columns=columnsInLayerL2).to_feather(
			f'{baseFilename}InputL2')
		
	# Load activations
	def loadActivations(self, baseFilename):
		self.valueSTE42 = pd.read_feather(f'{baseFilename}Input42').to_numpy()
		self.valueSTEL0 = pd.read_feather(f'{baseFilename}InputL0').to_numpy()
		self.valueSTEL1 = pd.read_feather(f'{baseFilename}InputL1').to_numpy()
		self.valueSTEL2 = pd.read_feather(f'{baseFilename}InputL2').to_numpy()
				
    # Save gradients
	def saveGradients(self, baseFilename):
		columnsInLayerL0 = [f'N{i}' for i in range(len(self.gradientsSTEL0[0]))]
		columnsInLayerL1 = [f'N{i}' for i in range(len(self.gradientsSTEL1[0]))]
		columnsInLayerL2 = [f'N{i}' for i in range(len(self.gradientsSTEL2[0]))]

		pd.DataFrame(
			self.gradientsSTEL0, columns=columnsInLayerL0).to_feather(
			f'{baseFilename}STEL0')

		pd.DataFrame(
			self.gradientsSTEL1, columns=columnsInLayerL1).to_feather(
			f'{baseFilename}STEL1')

		pd.DataFrame(
			self.gradientsSTEL2, columns=columnsInLayerL2).to_feather(
			f'{baseFilename}STEL2')
    
	# Load gradients
	def loadGradients(self, baseFilename):
		self.gradientsSTEL0 = pd.read_feather(f'{baseFilename}STEL0').to_numpy()
		self.gradientsSTEL1 = pd.read_feather(f'{baseFilename}STEL1').to_numpy()
		self.gradientsSTEL2 = pd.read_feather(f'{baseFilename}STEL2').to_numpy()
