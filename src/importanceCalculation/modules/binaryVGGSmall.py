import pandas as pd
from modelsCommon.steFunction import STEFunction
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from ttUtilities.auxFunctions import binaryArrayToSingleValue, integerToBinaryArray
import math

class VGGSmall(nn.Module):
	def __init__(self):
		super(VGGSmall, self).__init__()

		self.helpHookList = []

		# Layer 0
		self.conv0 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
		self.ste0 = STEFunction()
		self.maxpool0 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.helpHookList.append('ste0')

		# Layer 1
		self.conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
		self.ste1 = STEFunction()
		self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.helpHookList.append('ste1')
  
		# Layer 2.1
		self.conv21 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
		self.ste21 = STEFunction()
		self.helpHookList.append('ste21')
  
		# Layer 2.2
		self.conv22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.ste22 = STEFunction()
		self.maxpool22 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.helpHookList.append('ste22')
  
		# Layer 3.1
		self.conv31 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
		self.ste31 = STEFunction()
		self.helpHookList.append('ste31')

		# Layer 3.2
		self.conv32 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.ste32 = STEFunction()
		self.maxpool32 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.helpHookList.append('ste32')
  
		# Layer 4.1
		self.conv41 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.ste41 = STEFunction()
		self.helpHookList.append('ste41')
  
		# Layer 4.2
		self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.ste42 = STEFunction()
		self.maxpool42 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.helpHookList.append('ste42')
  
		# Layer FC0
		self.dropoutl0 = nn.Dropout(0.5)
		self.l0 = nn.Linear(512, 512)
		self.stel0 = STEFunction()
		self.helpHookList.append('stel0')
  
		# Layer FC1
		self.dropoutl1 = nn.Dropout(0.5)
		self.l1 = nn.Linear(512, 512)
		self.stel1 = STEFunction()
		self.helpHookList.append('stel1')
  
		# Layer FC2
		self.l2 = nn.Linear(512, 10)
  
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
		x = self.ste0(x)
		x = self.maxpool0(x)

		# Layer 1
		x = self.conv1(x)
		x = self.ste1(x)
		x = self.maxpool1(x)
  
		# Layer 2.1
		x = self.conv21(x)
		x = self.ste21(x)
  
		# Layer 2.2
		x = self.conv22(x)
		x = self.ste22(x)
		x = self.maxpool22(x)
  
		# Layer 3.1
		x = self.conv31(x)
		x = self.ste31(x)

		# Layer 3.2
		x = self.conv32(x)
		x = self.ste32(x)
		x = self.maxpool32(x)
  
		# Layer 4.1
		x = self.conv41(x)
		x = self.ste41(x)
  
		# Layer 4.2
		x = self.conv42(x)
		x = self.ste42(x)
		x = self.maxpool42(x)
  
		x = x.reshape(x.size(0), -1)
  
		# Layer FC0
		x = self.dropoutl0(x)
		x = self.l0(x)
		x = self.stel0(x)
  
		# Layer FC1
		x = self.dropoutl1(x)
		x = self.l1(x)
		x = self.stel1(x)
  
		# Layer FC2
		x = self.l2(x)

		return x

	# Probably there exists a better way to do this
	def registerHooks(self):
		# Forward hooks are needed to compute importance
		self.ste0.register_forward_hook(self.forward_hook_ste0)
		self.ste1.register_forward_hook(self.forward_hook_ste1)
		self.ste21.register_forward_hook(self.forward_hook_ste21)
		self.ste22.register_forward_hook(self.forward_hook_ste22)
		self.ste31.register_forward_hook(self.forward_hook_ste31)
		self.ste32.register_forward_hook(self.forward_hook_ste32)
		self.ste41.register_forward_hook(self.forward_hook_ste41)
		self.ste42.register_forward_hook(self.forward_hook_ste42)
		self.stel0.register_forward_hook(self.forward_hook_stel0)
		self.stel1.register_forward_hook(self.forward_hook_stel1)
  
		# Backward hooks are needed to compute importance
		self.ste0.register_full_backward_hook(self.backward_hook_ste0)
		self.ste1.register_full_backward_hook(self.backward_hook_ste1)
		self.ste21.register_full_backward_hook(self.backward_hook_ste21)
		self.ste22.register_full_backward_hook(self.backward_hook_ste22)
		self.ste31.register_full_backward_hook(self.backward_hook_ste31)
		self.ste32.register_full_backward_hook(self.backward_hook_ste32)
		self.ste41.register_full_backward_hook(self.backward_hook_ste41)
		self.ste42.register_full_backward_hook(self.backward_hook_ste42)
		self.stel0.register_full_backward_hook(self.backward_hook_stel0)
		self.stel1.register_full_backward_hook(self.backward_hook_stel1)

	# Define all backward hooks
	def backward_hook_ste0(self, module, grad_input, grad_output):
		self.gradsFromHooks['ste0']['backward'].append(grad_output[0].cpu().detach().numpy()[0])
  
	def backward_hook_ste1(self, module, grad_input, grad_output):
		self.gradsFromHooks['ste1']['backward'].append(grad_output[0].cpu().detach().numpy()[0])
  
	def backward_hook_ste21(self, module, grad_input, grad_output):
		self.gradsFromHooks['ste21']['backward'].append(grad_output[0].cpu().detach().numpy()[0])
  
	def backward_hook_ste22(self, module, grad_input, grad_output):
		self.gradsFromHooks['ste22']['backward'].append(grad_output[0].cpu().detach().numpy()[0])
  
	def backward_hook_ste31(self, module, grad_input, grad_output):
		self.gradsFromHooks['ste31']['backward'].append(grad_output[0].cpu().detach().numpy()[0])
  
	def backward_hook_ste32(self, module, grad_input, grad_output):
		self.gradsFromHooks['ste32']['backward'].append(grad_output[0].cpu().detach().numpy()[0])
  
	def backward_hook_ste41(self, module, grad_input, grad_output):
		self.gradsFromHooks['ste41']['backward'].append(grad_output[0].cpu().detach().numpy()[0])
  
	def backward_hook_ste42(self, module, grad_input, grad_output):
		self.gradsFromHooks['ste42']['backward'].append(grad_output[0].cpu().detach().numpy()[0])
  
	def backward_hook_stel0(self, module, grad_input, grad_output):
		self.gradsFromHooks['stel0']['backward'].append(grad_output[0].cpu().detach().numpy()[0])
  
	def backward_hook_stel1(self, module, grad_input, grad_output):
		self.gradsFromHooks['stel1']['backward'].append(grad_output[0].cpu().detach().numpy()[0])
  
	# Define all forward hooks
	def backward_hook_ste0(self, module, val_input, val_output):
		self.activationsFromHooks['ste0']['forward'].append(val_output[0].cpu().detach().numpy()[0])
  
	def backward_hook_ste1(self, module, val_input, val_output):
		self.activationsFromHooks['ste1']['forward'].append(val_output[0].cpu().detach().numpy()[0])
  
	def backward_hook_ste21(self, module, val_input, val_output):
		self.activationsFromHooks['ste21']['forward'].append(val_output[0].cpu().detach().numpy()[0])
  
	def backward_hook_ste22(self, module, val_input, val_output):
		self.activationsFromHooks['ste22']['forward'].append(val_output[0].cpu().detach().numpy()[0])
  
	def backward_hook_ste31(self, module, val_input, val_output):
		self.activationsFromHooks['ste31']['forward'].append(val_output[0].cpu().detach().numpy()[0])
  
	def backward_hook_ste32(self, module, val_input, val_output):
		self.activationsFromHooks['ste32']['forward'].append(val_output[0].cpu().detach().numpy()[0])
  
	def backward_hook_ste41(self, module, val_input, val_output):
		self.activationsFromHooks['ste41']['forward'].append(val_output[0].cpu().detach().numpy()[0])
  
	def backward_hook_ste42(self, module, val_input, val_output):
		self.activationsFromHooks['ste42']['forward'].append(val_output[0].cpu().detach().numpy()[0])
  
	def backward_hook_stel0(self, module, val_input, val_output):
		self.activationsFromHooks['stel0']['forward'].append(val_output[0].cpu().detach().numpy()[0])
  
	def backward_hook_stel1(self, module, val_input, val_output):
		self.activationsFromHooks['stel1']['forward'].append(val_output[0].cpu().detach().numpy()[0])
  
	# Change each hook list to an equivalent array
	def listToArray(self):
		for grads in self.gradsFromHooks:
			self.gradsFromHooks[grads]['forward'] = np.array(self.gradsFromHooks[grads]['forward'])
			self.gradsFromHooks[grads]['backward'] = np.array(self.gradsFromHooks[grads]['backward'])
   
	# Compute importance
	def computeImportance(self):
		importances = []

		for grad in self.gradsFromHooks:
			importances.append(np.abs(np.multiply(self.gradsFromHooks[grad]['backward'], self.gradsFromHooks[grad]['forward'])))
   
		return importances

	# Save activations
	def saveActivations(self, baseFilename):
		for grad in self.dataFromHooks:
			pd.DataFrame(
				self.dataFromHooks[grad]['forward']).to_feather(
					f'{baseFilename}Forward{grad}')
    
    # Load activations
	def loadActivations(self, baseFilename):
		for grad in self.dataFromHooks:
			self.dataFromHooks[grad]['forward'] = pd.read_feather(f'{baseFilename}Forward{grad}')
    
    # Save gradients
	def saveGradients(self, baseFilename):
		for grad in self.dataFromHooks:
			pd.DataFrame(
				self.dataFromHooks[grad]['backward']).to_feather(
					f'{baseFilename}Backward{grad}')
    
	# Load gradients
	def loadGradients(self, baseFilename):
		for grad in self.dataFromHooks:
			self.dataFromHooks[grad]['backward'] = pd.read_feather(f'{baseFilename}Backward{grad}')
    
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