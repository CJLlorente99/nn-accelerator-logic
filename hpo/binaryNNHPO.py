from models.steFunction import STEFunction
from hpo.quantizier import STEFunctionQuant
from torch import nn
import torch.nn.functional as F
import numpy as np
from ttUtilities.auxFunctions import binaryArrayToSingleValue, integerToBinaryArray


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

		# Lists for hook data
		self.gradientsSTE1 = []
		self.gradientsSTE2 = []
		self.gradientsSTE3 = []

		self.valueSTE1 = []
		self.valueSTE2 = []
		self.valueSTE3 = []

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

		return F.log_softmax(x, dim=1)

	def registerHooks(self):
		# Register hooks
		self.ste1.register_full_backward_hook(self.backward_hook_ste1)
		self.ste2.register_full_backward_hook(self.backward_hook_ste2)
		self.ste3.register_full_backward_hook(self.backward_hook_ste3)
		self.ste1.register_forward_hook(self.forward_hook_ste1)
		self.ste2.register_forward_hook(self.forward_hook_ste2)
		self.ste3.register_forward_hook(self.forward_hook_ste3)

	def backward_hook_ste1(self, module, grad_input, grad_output):
		self.gradientsSTE1.append(grad_input[0].cpu().detach().numpy()[0])

	def backward_hook_ste2(self, module, grad_input, grad_output):
		self.gradientsSTE2.append(grad_input[0].cpu().detach().numpy()[0])

	def backward_hook_ste3(self, module, grad_input, grad_output):
		self.gradientsSTE3.append(grad_input[0].cpu().detach().numpy()[0])

	def forward_hook_ste1(self, module, val_input, val_output):
		self.valueSTE1.append(val_output[0].cpu().detach().numpy())

	def forward_hook_ste2(self, module, val_input, val_output):
		self.valueSTE2.append(val_output[0].cpu().detach().numpy())

	def forward_hook_ste3(self, module, val_input, val_output):
		self.valueSTE3.append(val_output[0].cpu().detach().numpy())

	def computeImportance(self, neuronPerLayer):
		self.gradientsSTE1 = np.array(self.gradientsSTE1).squeeze().reshape(len(self.gradientsSTE1), neuronPerLayer)
		self.gradientsSTE2 = np.array(self.gradientsSTE2).squeeze().reshape(len(self.gradientsSTE2), neuronPerLayer)
		self.gradientsSTE3 = np.array(self.gradientsSTE3).squeeze().reshape(len(self.gradientsSTE3), neuronPerLayer)

		self.valueSTE1 = np.array(self.valueSTE1).squeeze().reshape(len(self.valueSTE1), neuronPerLayer)
		self.valueSTE2 = np.array(self.valueSTE2).squeeze().reshape(len(self.valueSTE2), neuronPerLayer)
		self.valueSTE3 = np.array(self.valueSTE3).squeeze().reshape(len(self.valueSTE3), neuronPerLayer)

		importanceSTE1 = abs(np.multiply(self.gradientsSTE1, self.valueSTE1))
		importanceSTE2 = abs(np.multiply(self.gradientsSTE2, self.valueSTE2))
		importanceSTE3 = abs(np.multiply(self.gradientsSTE3, self.valueSTE3))

		return importanceSTE1, importanceSTE2, importanceSTE3

	def individualActivationsToUniqueValue(self):
		aux = []
		for activation in self.valueSTE1:
			aux.append(binaryArrayToSingleValue(activation))
		self.valueSTE1 = np.array(aux)

		aux = []
		for activation in self.valueSTE2:
			aux.append(binaryArrayToSingleValue(activation))
		self.valueSTE2 = np.array(aux)

		aux = []
		for activation in self.valueSTE3:
			aux.append(binaryArrayToSingleValue(activation))
		self.valueSTE3 = np.array(aux)
