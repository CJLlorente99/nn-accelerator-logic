import pandas as pd
from modelsCommon.steFunction import STEFunction
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from ttUtilities.auxFunctions import binaryArrayToSingleValue, integerToBinaryArray
import math
import os

class binaryVGGVerySmall2(nn.Module):
	def __init__(self, resizeFactor, relus):
		super(binaryVGGVerySmall2, self).__init__()
  
		self.helpHookList = []

		# Layer 0
		self.conv0 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
		self.bn0 = nn.BatchNorm2d(64)
		if relus[0]:
			self.relu0 = nn.ReLU()
		else:
			self.relu0 = STEFunction()
		self.maxpool0 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.helpHookList.append('relu0')

		# Layer 1
		self.conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm2d(128)
		if relus[1]:
			self.relu1 = nn.ReLU()
		else:
			self.relu1 = STEFunction()
		self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.helpHookList.append('ste1')
  
		# Layer 2.1
		self.conv21 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
		self.bn21 = nn.BatchNorm2d(256)
		if relus[2]:
			self.relu21 = nn.ReLU()
		else:
			self.relu21 = STEFunction()
		self.maxpool22 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.helpHookList.append('ste21')
  
		# Layer 3.1
		self.conv31 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
		self.bn31 = nn.BatchNorm2d(512)
		if relus[3]:
			self.relu31 = nn.ReLU()
		else:
			self.relu31 = STEFunction()
		self.maxpool32 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.helpHookList.append('ste31')
  
		# Layer 4.1
		self.conv41 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.bn41 = nn.BatchNorm2d(512)
		if relus[4]:
			self.relu41 = nn.ReLU()
		else:
			self.relu41 = STEFunction()
		self.maxpool42 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.helpHookList.append('ste41')
  
		# Layer FC0
		self.l0 = nn.Linear(resizeFactor*resizeFactor*512, 4096)
		self.bnl0 = nn.BatchNorm1d(4096)
		if relus[5]:
			self.relul0 = nn.ReLU()
		else:
			self.relul0 = STEFunction()
		self.helpHookList.append('stel0')
  
		# Layer FC1
		self.l1 = nn.Linear(4096, 4096)
		self.bnl1 = nn.BatchNorm1d(4096)
		if relus[6]:
			self.relul1 = nn.ReLU()
		else:
			self.relul1 = STEFunction()
		self.helpHookList.append('stel1')
  
		# Layer FC2
		self.l2 = nn.Linear(4096, 1000)
		self.bnl2 = nn.BatchNorm1d(1000)
		if relus[7]:
			self.relul2 = nn.ReLU()
		else:
			self.relul2 = STEFunction()
		self.helpHookList.append('stel2')

		# Layer FC3
		self.l3 = nn.Linear(1000, 10)
  
		# Initialize
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				m.bias.data.zero_()
    
		# Create list for hooks
		self.dataFromHooks = {}
		for i in self.helpHookList:
			self.dataFromHooks[i] = {'forward': [], 'backward': []}
   
		# Create list for activationMinimization (only Binary)
		self.activationSizeInfo = {}
		for i in self.helpHookList:
			self.activationSizeInfo[i] = 0
    

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
		x = self.maxpool22(x)
  
		# Layer 3.1
		x = self.conv31(x)
		x = self.bn31(x)
		x = self.relu31(x)
		x = self.maxpool32(x)
  
		# Layer 4.1
		x = self.conv41(x)
		x = self.bn41(x)
		x = self.relu41(x)
		x = self.maxpool42(x)
  
		x = x.reshape(x.size(0), -1)
  
		# Layer FC0
		x = self.l0(x)
		x = self.bnl0(x)
		x = self.relul0(x)
  
		# Layer FC1
		x = self.l1(x)
		x = self.bnl1(x)
		x = self.relul1(x)

		# Layer FC2
		x = self.l2(x)
		x = self.bnl2(x)
		x = self.relul2(x)
  
		# Layer FC3
		x = self.l3(x)

		return x

	# Probably there exists a better way to do this
	def registerHooks(self, activations: bool = True, gradients: bool = True):
		# Forward hooks are needed to compute importance
		if activations:
			self.relu0.register_forward_hook(self.forward_hook_relu0)
			self.relu1.register_forward_hook(self.forward_hook_relu1)
			self.relu21.register_forward_hook(self.forward_hook_relu21)
			self.relu31.register_forward_hook(self.forward_hook_relu31)
			self.relu41.register_forward_hook(self.forward_hook_relu41)
			self.relul0.register_forward_hook(self.forward_hook_relul0)
			self.relul1.register_forward_hook(self.forward_hook_relul1)
			self.relul2.register_forward_hook(self.forward_hook_relul2)
  
		# Backward hooks are needed to compute importance
		if gradients:
			self.relu0.register_full_backward_hook(self.backward_hook_relu0)
			self.relu1.register_full_backward_hook(self.backward_hook_relu1)
			self.relu21.register_full_backward_hook(self.backward_hook_relu21)
			self.relu31.register_full_backward_hook(self.backward_hook_relu31)
			self.relu41.register_full_backward_hook(self.backward_hook_relu41)
			self.relul0.register_full_backward_hook(self.backward_hook_relul0)
			self.relul1.register_full_backward_hook(self.backward_hook_relul1)
			self.relul2.register_full_backward_hook(self.backward_hook_relul2)

	# Define all backward hooks
	def backward_hook_relu0(self, module, grad_input, grad_output):
		aux = grad_output[0].flatten(start_dim=-2, end_dim=-1).cpu().detach().numpy()[0]
		self.dataFromHooks['relu0']['backward'].append(np.float16(aux))
  
	def backward_hook_relu1(self, module, grad_input, grad_output):
		aux = grad_output[0].flatten(start_dim=-2, end_dim=-1).cpu().detach().numpy()[0]
		self.dataFromHooks['ste1']['backward'].append(np.float16(aux))
  
	def backward_hook_relu21(self, module, grad_input, grad_output):
		aux = grad_output[0].flatten(start_dim=-2, end_dim=-1).cpu().detach().numpy()[0]
		self.dataFromHooks['ste21']['backward'].append(np.float16(aux))
  
	def backward_hook_relu31(self, module, grad_input, grad_output):
		aux = grad_output[0].flatten(start_dim=-2, end_dim=-1).cpu().detach().numpy()[0]
		self.dataFromHooks['ste31']['backward'].append(np.float16(aux))
  
	def backward_hook_relu41(self, module, grad_input, grad_output):
		aux = grad_output[0].flatten(start_dim=-2, end_dim=-1).cpu().detach().numpy()[0]
		self.dataFromHooks['ste41']['backward'].append(np.float16(aux))
  
	def backward_hook_relul0(self, module, grad_input, grad_output):
		self.dataFromHooks['stel0']['backward'].append(np.float16(grad_output[0].cpu().detach().numpy()[0]))
  
	def backward_hook_relul1(self, module, grad_input, grad_output):
		self.dataFromHooks['stel1']['backward'].append(np.float16(grad_output[0].cpu().detach().numpy()[0]))

	def backward_hook_relul2(self, module, grad_input, grad_output):
		self.dataFromHooks['stel2']['backward'].append(np.float16(grad_output[0].cpu().detach().numpy()[0]))
  
	# Define all forward hooks
	def forward_hook_relu0(self, module, val_input, val_output):
		aux = val_output.flatten(start_dim=-2, end_dim=-1).cpu().detach().numpy()[0]
		self.dataFromHooks['relu0']['forward'].append(np.float16(aux))
  
	def forward_hook_relu1(self, module, val_input, val_output):
		aux = val_output.flatten(start_dim=-2, end_dim=-1).cpu().detach().numpy()[0]
		self.dataFromHooks['ste1']['forward'].append(np.float16(aux))
  
	def forward_hook_relu21(self, module, val_input, val_output):
		aux = val_output.flatten(start_dim=-2, end_dim=-1).cpu().detach().numpy()[0]
		self.dataFromHooks['ste21']['forward'].append(np.float16(aux))
  
	def forward_hook_relu31(self, module, val_input, val_output):
		aux = val_output.flatten(start_dim=-2, end_dim=-1).cpu().detach().numpy()[0]
		self.dataFromHooks['ste31']['forward'].append(np.float16(aux))
  
	def forward_hook_relu41(self, module, val_input, val_output):
		aux = val_output.flatten(start_dim=-2, end_dim=-1).cpu().detach().numpy()[0]
		self.dataFromHooks['ste41']['forward'].append(np.float16(aux))
  
	def forward_hook_relul0(self, module, val_input, val_output):
		self.dataFromHooks['stel0']['forward'].append(np.float16(val_output[0].cpu().detach().numpy()))
  
	def forward_hook_relul1(self, module, val_input, val_output):
		self.dataFromHooks['stel1']['forward'].append(np.float16(val_output[0].cpu().detach().numpy()))

	def forward_hook_relul2(self, module, val_input, val_output):
		self.dataFromHooks['stel2']['forward'].append(np.float16(val_output[0].cpu().detach().numpy()))
  
	# Change each hook list to an equivalent array
	def listToArray(self):
		for grads in self.dataFromHooks:
			self.dataFromHooks[grads]['forward'] = np.array(self.dataFromHooks[grads]['forward'], dtype='float16')
			self.dataFromHooks[grads]['backward'] = np.array(self.dataFromHooks[grads]['backward'], dtype='float16')

	# Clean hooks list
	def resetHookLists(self):
		for grad in self.dataFromHooks:
			for direction in self.dataFromHooks[grad]:
				self.dataFromHooks[grad][direction] = []
   
	# Compute importance
	def computeImportance(self):
		importances = []

		for grad in self.dataFromHooks:
			aux = self.dataFromHooks[grad]['backward'].shape
			print(f'Shape backward {grad} is {aux}')
			aux = self.dataFromHooks[grad]['forward'].shape
			print(f'Shape forward {grad} is {aux}')
			importances.append(np.abs(np.multiply(self.dataFromHooks[grad]['backward'], self.dataFromHooks[grad]['forward'])))
	
		return importances
	
	# Compute importance layer by layer
	def computeImportanceLayer(self, layerName):
		aux = self.dataFromHooks[layerName]['backward'].shape
		print(f'Shape backward {layerName} is {aux}')
		aux = self.dataFromHooks[layerName]['forward'].shape
		print(f'Shape forward {layerName} is {aux}')
		return np.abs(np.multiply(self.dataFromHooks[layerName]['backward'], self.dataFromHooks[layerName]['forward']))
	

	# Load and return importances in the right order
	# TODO create save correctly
	def loadImportance(self, folderFilename : str):
		files = os.listdir(folderFilename)
		importances = []
		for grad in self.dataFromHooks:
			if f'{grad}_' in files:
				i = 0
				aux = []
				while f'{grad}_{i}' in files:
					aux.append(pd.read_feather(f'{folderFilename}/{grad}_{i}').to_numpy())
					i += 1
				importances.append(np.array(aux))
			else:
				aux.append(pd.read_feather(f'{folderFilename}/{grad}').to_numpy())

	# Save activations
	def saveActivations(self, baseFilename):
		# Check folder exists
		if not os.path.exists(f'{baseFilename}'):
			os.makedirs(f'{baseFilename}')
   
		for grad in self.dataFromHooks:
			if grad.startswith('relul') or grad.startswith('stel'): # then it is linear layer
				columnTags = [f'{i}' for i in range(self.dataFromHooks[grad]['forward'].shape[1])]
				pd.DataFrame(
						self.dataFromHooks[grad]['forward'], columns=columnTags).to_csv(
							f'{baseFilename}/{grad}', mode='a', index=False, header=False)
			else:
				if not os.path.exists(f'{baseFilename}/{grad}'):
					os.makedirs(f'{baseFilename}/{grad}')
     
				for iDepth in range(self.dataFromHooks[grad]['forward'].shape[1]):
					columnTags = [f'{i}' for i in range(self.dataFromHooks[grad]['forward'].shape[2])]
					pd.DataFrame(
						self.dataFromHooks[grad]['forward'][:, iDepth, :], columns=columnTags).to_csv(
							f'{baseFilename}/{grad}/{iDepth}', mode='a', index=False, header=False)
			print(f'Saved activations {grad}')
    
    # Load activations
	def loadActivations(self, baseFilename):
		for grad in self.dataFromHooks:
			if grad.startswith('relul'):
				self.dataFromHooks[grad]['forward'] = pd.read_feather(f'{baseFilename}/{grad}').to_numpy()
				print(self.dataFromHooks[grad]['forward'].shape)
			else:
				self.dataFromHooks[grad]['forward'] = []
				for file in os.scandir(f'{baseFilename}/{grad}'):
					self.dataFromHooks[grad]['forward'].append(pd.read_feather(f'{file.path}').to_numpy())
				self.dataFromHooks[grad]['forward'] = np.array(self.dataFromHooks[grad]['forward'])
				self.dataFromHooks[grad]['forward'] = np.moveaxis(self.dataFromHooks[grad]['forward'], 0, 1)
				print(self.dataFromHooks[grad]['forward'].shape)
			print(f'Activations from {grad} loaded')
				
    
    # Save gradients
	def saveGradients(self, baseFilename):
		# Check folder exists
		if not os.path.exists(f'{baseFilename}'):
			os.makedirs(f'{baseFilename}')
   
		for grad in self.dataFromHooks:
			if grad.startswith('relul') or grad.startswith('stel'):
				columnTags = [f'{i}' for i in range(self.dataFromHooks[grad]['backward'].shape[1])]
				pd.DataFrame(
					self.dataFromHooks[grad]['backward'], columns=columnTags).to_csv(
						f'{baseFilename}/{grad}', mode='a', index=False, header=False)
			else:
				if not os.path.exists(f'{baseFilename}/{grad}'):
					os.makedirs(f'{baseFilename}/{grad}')
     
				for iDepth in range(self.dataFromHooks[grad]['backward'].shape[1]):
					columnTags = [f'{i}' for i in range(self.dataFromHooks[grad]['backward'].shape[2])]
					pd.DataFrame(
						self.dataFromHooks[grad]['backward'][:, iDepth, :], columns=columnTags).to_csv(
							f'{baseFilename}/{grad}/{iDepth}', mode='a', index=False, header=False)
			print(f'Saved gradients {grad}')
    
	# Load gradients
	def loadGradients(self, baseFilename):
		for grad in self.dataFromHooks:
			if grad.startswith('relul'):
				self.dataFromHooks[grad]['backward'] = pd.read_feather(f'{baseFilename}/{grad}').to_numpy()
				print(self.dataFromHooks[grad]['backward'].shape)
			else:
				self.dataFromHooks[grad]['backward'] = []
				for file in os.scandir(f'{baseFilename}/{grad}'):
					self.dataFromHooks[grad]['backward'].append(pd.read_feather(f'{file.path}').to_numpy())
				self.dataFromHooks[grad]['backward'] = np.array(self.dataFromHooks[grad]['backward'])
				self.dataFromHooks[grad]['backward'] = np.moveaxis(self.dataFromHooks[grad]['backward'], 0, 1)
				print(self.dataFromHooks[grad]['backward'].shape)
			print(f'Gradients from {grad} loaded')
    
	# SqueezeActivationToUniqueValues (only binary)
	def individualActivationsToUniqueValue(self):
		for grad in self.dataFromHooks:
			aux = []
			i = 0
			for activation in self.dataFromHooks[grad]['forward']:
				aux.append(binaryArrayToSingleValue(activation))

				if (i + 1) % 5000 == 0:
					print(f"Activations to Unique Value (Input0) [{i + 1:>4d}/{len(self.input0):>4d}]")
				i += 1
    
			self.dataFromHooks[grad]['forward'] = np.array(aux)
			self.activationSizeInfo[grad] = int(self.dataFromHooks[grad]['forward'].shape[1] / 2)
