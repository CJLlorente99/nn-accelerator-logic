from modelsCommon.steFunction import STEFunction
from torch import nn
import torch.nn.functional as F
import numpy as np
from ttUtilities.auxFunctions import binaryArrayToSingleValue, integerToBinaryArray
import torch
import pandas as pd


class BinaryNeuralNetwork(nn.Module):
	def __init__(self, neuronPerLayer=100):
		super(BinaryNeuralNetwork, self).__init__()
		self.flatten = nn.Flatten()

		self.l0 = nn.Linear(28 * 28, neuronPerLayer)
		self.bn0 = nn.BatchNorm1d(neuronPerLayer)
		self.ste0 = STEFunction()

		self.l1 = nn.Linear(neuronPerLayer, neuronPerLayer)
		self.bn1 = nn.BatchNorm1d(neuronPerLayer)
		self.ste1 = STEFunction()

		self.l2 = nn.Linear(neuronPerLayer, neuronPerLayer)
		self.bn2 = nn.BatchNorm1d(neuronPerLayer)
		self.ste2 = STEFunction()

		self.l3 = nn.Linear(neuronPerLayer, neuronPerLayer)
		self.bn3 = nn.BatchNorm1d(neuronPerLayer)
		self.ste3 = STEFunction()

		self.l4 = nn.Linear(neuronPerLayer, 10)
		self.bn4 = nn.BatchNorm1d(10)

		self.iden = nn.Identity()

		# Lists for hook data
		self.gradientsSTE0 = []
		self.gradientsSTE1 = []
		self.gradientsSTE2 = []
		self.gradientsSTE3 = []
		self.gradientsIden = []

		self.valueSTE0 = []
		self.valueSTE1 = []
		self.valueSTE2 = []
		self.valueSTE3 = []
		self.valueIden = []

		self.input0 = []  # The rest are the same as the self.value...

		self.activationSize = 0
		self.activationSizeInput = 0
		self.activationSizeOutput = 0

		# Flag to enable importance calculation through forward (instead of estimation through gradient)
		self.legacyImportance = False
		self.neuronSwitchedOff = (0, 0)  # (layer, neuron)

	def forward(self, x):
		x = self.flatten(x)

		x = self.l0(x)
		x = self.bn0(x)
		x = self.ste0(x)

		x = self.l1(x)
		x = self.bn1(x)
		x = self.ste1(x)

		x = self.l2(x)
		x = self.bn2(x)
		x = self.ste2(x)

		x = self.l3(x)
		x = self.bn3(x)
		x = self.ste3(x)

		x = self.l4(x)
		x = self.bn4(x)

		x = self.iden(x)

		return F.log_softmax(x, dim=1)

	def registerHooks(self):
		self.l0.register_forward_hook(self.forward_hook_l0)
		self.ste0.register_forward_hook(self.forward_hook_ste0)
		self.ste1.register_forward_hook(self.forward_hook_ste1)
		self.ste2.register_forward_hook(self.forward_hook_ste2)
		self.ste3.register_forward_hook(self.forward_hook_ste3)
		self.iden.register_forward_hook(self.forward_hook_iden)

		if not self.legacyImportance:
			# Register hooks
			self.ste0.register_full_backward_hook(self.backward_hook_ste0)
			self.ste1.register_full_backward_hook(self.backward_hook_ste1)
			self.ste2.register_full_backward_hook(self.backward_hook_ste2)
			self.ste3.register_full_backward_hook(self.backward_hook_ste3)

	def backward_hook_ste0(self, module, grad_input, grad_output):
		self.gradientsSTE0.append(grad_output[0].cpu().detach().numpy()[0])

	def backward_hook_ste1(self, module, grad_input, grad_output):
		self.gradientsSTE1.append(grad_output[0].cpu().detach().numpy()[0])

	def backward_hook_ste2(self, module, grad_input, grad_output):
		self.gradientsSTE2.append(grad_output[0].cpu().detach().numpy()[0])

	def backward_hook_ste3(self, module, grad_input, grad_output):
		self.gradientsSTE3.append(grad_output[0].cpu().detach().numpy()[0])

	def forward_hook_l0(self, module, val_input, val_output):
		self.input0.append(val_input[0].cpu().detach().numpy())

	def forward_hook_ste0(self, module, val_input, val_output):
		if not self.legacyImportance:
			self.valueSTE0.append(val_output[0].cpu().detach().numpy())
		else:
			if self.neuronSwitchedOff[0] == 0:
				val_output[0][self.neuronSwitchedOff[1]] = 0

	def forward_hook_ste1(self, module, val_input, val_output):
		if not self.legacyImportance:
			self.valueSTE1.append(val_output[0].cpu().detach().numpy())
		else:
			if self.neuronSwitchedOff[0] == 1:
				val_output[0][self.neuronSwitchedOff[1]] = 0

	def forward_hook_ste2(self, module, val_input, val_output):
		if not self.legacyImportance:
			self.valueSTE2.append(val_output[0].cpu().detach().numpy())
		else:
			if self.neuronSwitchedOff[0] == 2:
				val_output[0][self.neuronSwitchedOff[1]] = 0

	def forward_hook_ste3(self, module, val_input, val_output):
		if not self.legacyImportance:
			self.valueSTE3.append(val_output[0].cpu().detach().numpy())
		else:
			if self.neuronSwitchedOff[0] == 3:
				val_output[0][self.neuronSwitchedOff[1]] = 0

	def forward_hook_iden(self, module, val_input, val_output):
		a = val_output.argmax(1)
		# TODO. Could be wrong
		self.valueIden.append(torch.zeros(val_output.shape).scatter(1, a.unsqueeze(1), 1.0).cpu().detach().numpy()[0])

	def listToArray(self, neuronPerLayer):
		self.input0 = np.array(self.input0).squeeze().reshape(len(self.input0), 28 * 28)

		self.gradientsSTE0 = np.array(self.gradientsSTE0).squeeze().reshape(len(self.gradientsSTE0), neuronPerLayer)
		self.gradientsSTE1 = np.array(self.gradientsSTE1).squeeze().reshape(len(self.gradientsSTE1), neuronPerLayer)
		self.gradientsSTE2 = np.array(self.gradientsSTE2).squeeze().reshape(len(self.gradientsSTE2), neuronPerLayer)
		self.gradientsSTE3 = np.array(self.gradientsSTE3).squeeze().reshape(len(self.gradientsSTE3), neuronPerLayer)

		self.valueSTE0 = np.array(self.valueSTE0).squeeze().reshape(len(self.valueSTE0), neuronPerLayer)
		self.valueSTE1 = np.array(self.valueSTE1).squeeze().reshape(len(self.valueSTE1), neuronPerLayer)
		self.valueSTE2 = np.array(self.valueSTE2).squeeze().reshape(len(self.valueSTE2), neuronPerLayer)
		self.valueSTE3 = np.array(self.valueSTE3).squeeze().reshape(len(self.valueSTE3), neuronPerLayer)

		# self.valueIden = np.array(self.valueIden).squeeze().reshape(len(self.valueIden), 10)

	def computeImportance(self, neuronPerLayer):
		# CAREFUL, as values are either +1 or -1, importance is equal to gradient
		importanceSTE0 = np.abs(np.multiply(self.valueSTE0, self.gradientsSTE0))
		print('Importance STE0 calculated')
		importanceSTE1 = np.abs(np.multiply(self.valueSTE1, self.gradientsSTE1))
		print('Importance STE1 calculated')
		importanceSTE2 = np.abs(np.multiply(self.valueSTE2, self.gradientsSTE2))
		print('Importance STE2 calculated')
		importanceSTE3 = np.abs(np.multiply(self.valueSTE3, self.gradientsSTE3))
		print('Importance STE3 calculated')

		return [importanceSTE0, importanceSTE1, importanceSTE2, importanceSTE3]

	def saveActivations(self, baseFilename):
		columnsInLayer0 = [f'{i}' for i in range(len(self.input0[0]))]
		columnsInLayer1 = [f'{i}' for i in range(len(self.valueSTE0[0]))]
		columnsInLayer2 = [f'{i}' for i in range(len(self.valueSTE1[0]))]
		columnsInLayer3 = [f'{i}' for i in range(len(self.valueSTE2[0]))]
		columnsInLayer4 = [f'{i}' for i in range(len(self.valueSTE3[0]))]

		# columnsOutLayer5 = [f'N{i}' for i in range(len(self.valueIden[0]))]

		pd.DataFrame(
			self.input0, columns=columnsInLayer0).to_feather(
			f'{baseFilename}Input0')

		pd.DataFrame(
			self.valueSTE0, columns=columnsInLayer1).to_feather(
			f'{baseFilename}Input1')

		pd.DataFrame(
			self.valueSTE1, columns=columnsInLayer2).to_feather(
			f'{baseFilename}Input2')

		pd.DataFrame(
			self.valueSTE2, columns=columnsInLayer3).to_feather(
			f'{baseFilename}Input3')

		pd.DataFrame(
			self.valueSTE3, columns=columnsInLayer4).to_feather(
			f'{baseFilename}Input4')

		# pd.DataFrame(
		# 	np.array(self.valueIden), columns=columnsOutLayer5).to_feather(
		# 	f'{baseFilename}Out5')

	def saveGradients(self, baseFilename: str, targets: list):
		columnsInLayer1 = [f'N{i}' for i in range(len(self.gradientsSTE0[0]))]
		columnsInLayer2 = [f'N{i}' for i in range(len(self.gradientsSTE1[0]))]
		columnsInLayer3 = [f'N{i}' for i in range(len(self.gradientsSTE2[0]))]
		columnsInLayer4 = [f'N{i}' for i in range(len(self.gradientsSTE3[0]))]

		aux = pd.DataFrame(self.gradientsSTE0, columns=columnsInLayer1)
		aux['target'] = targets
		aux.to_feather(f'{baseFilename}STE0')

		aux = pd.DataFrame(self.gradientsSTE1, columns=columnsInLayer2)
		aux['target'] = targets
		aux.to_feather(f'{baseFilename}STE1')

		aux = pd.DataFrame(self.gradientsSTE2, columns=columnsInLayer3)
		aux['target'] = targets
		aux.to_feather(f'{baseFilename}STE2')

		aux = pd.DataFrame(self.gradientsSTE3, columns=columnsInLayer4)
		aux['target'] = targets
		aux.to_feather(f'{baseFilename}STE3')

	def signToBinary(self):
		self.input0[self.input0 == -1] = 0
		self.valueSTE0[self.valueSTE0 == -1] = 0
		self.valueSTE1[self.valueSTE1 == -1] = 0
		self.valueSTE2[self.valueSTE2 == -1] = 0
		self.valueSTE3[self.valueSTE3 == -1] = 0

	def individualActivationsToUniqueValue(self):
		aux = []
		i = 0
		for activation in self.input0:
			aux.append(binaryArrayToSingleValue(activation))

			if (i + 1) % 5000 == 0:
				print(f"Activations to Unique Value (Input0) [{i + 1:>4d}/{len(self.input0):>4d}]")
			i += 1
		self.input0 = np.array(aux)
		self.activationSizeInput = int(self.input0.shape[1] / 2)

		aux = []
		i = 0
		for activation in self.valueSTE0:
			aux.append(binaryArrayToSingleValue(activation))

			if (i + 1) % 5000 == 0:
				print(f"Activations to Unique Value (STE0) [{i + 1:>4d}/{len(self.valueSTE0):>4d}]")
			i += 1
		self.valueSTE0 = np.array(aux)

		aux = []
		i = 0
		for activation in self.valueSTE1:
			aux.append(binaryArrayToSingleValue(activation))

			if (i + 1) % 5000 == 0:
				print(f"Activations to Unique Value (STE1) [{i + 1:>4d}/{len(self.valueSTE1):>4d}]")
			i += 1
		self.valueSTE1 = np.array(aux)

		aux = []
		i = 0
		for activation in self.valueSTE2:
			aux.append(binaryArrayToSingleValue(activation))

			if (i + 1) % 5000 == 0:
				print(f"Activations to Unique Value (STE2) [{i + 1:>4d}/{len(self.valueSTE2):>4d}]")
			i += 1
		self.valueSTE2 = np.array(aux)

		aux = []
		i = 0
		for activation in self.valueSTE3:
			aux.append(binaryArrayToSingleValue(activation))

			if (i + 1) % 5000 == 0:
				print(f"Activations to Unique Value (STE3) [{i + 1:>4d}/{len(self.valueSTE3):>4d}]")
			i += 1
		self.valueSTE3 = np.array(aux)

		# Activation size of the output layer will always be considered to be 10
		# aux = []
		# i = 0
		# for activation in self.valueIden:
		# 	aux.append(binaryArrayToSingleValue(activation))
		#
		# 	if (i + 1) % 250 == 0:
		# 		print(f"Activations to Unique Value (Iden) [{i + 1:>4d}/{len(self.valueIden):>4d}]")
		# 	i += 1
		# self.valueIden = np.array(aux)
		# self.activationSizeOutput = int(self.valueIden.shape[1] / 2)

		self.activationSize = int(self.valueSTE3.shape[1] / 2)  # It has the info of the lengths too
