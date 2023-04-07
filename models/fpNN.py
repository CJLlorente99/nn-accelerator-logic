from torch import nn
import torch.nn.functional as F
import numpy as np


class FPNeuralNetwork(nn.Module):
	def __init__(self, neuronPerLayer):
		super(FPNeuralNetwork, self).__init__()
		self.flatten = nn.Flatten()

		self.l0 = nn.Linear(28 * 28, neuronPerLayer)
		self.bn0 = nn.BatchNorm1d(neuronPerLayer)
		self.relu0 = nn.ReLU()

		self.l1 = nn.Linear(neuronPerLayer, neuronPerLayer)
		self.bn1 = nn.BatchNorm1d(neuronPerLayer)
		self.relu1 = nn.ReLU()

		self.l2 = nn.Linear(neuronPerLayer, neuronPerLayer)
		self.bn2 = nn.BatchNorm1d(neuronPerLayer)
		self.relu2 = nn.ReLU()

		self.l3 = nn.Linear(neuronPerLayer, neuronPerLayer)
		self.bn3 = nn.BatchNorm1d(neuronPerLayer)
		self.relu3 = nn.ReLU()

		self.l4 = nn.Linear(neuronPerLayer, 10)
		self.bn4 = nn.BatchNorm1d(10)

		# Lists for hook data
		self.gradientsReLU1 = []
		self.gradientsReLU2 = []
		self.gradientsReLU3 = []

		self.valueReLU1 = []
		self.valueReLU2 = []
		self.valueReLU3 = []

		# Flag to enable importance calculation through forward (instead of estimation through gradient)
		self.legacyImportance = False
		self.neuronSwitchedOff = (0, 0)  # (layer, neuron)

	def forward(self, x):
		x = self.flatten(x)

		x = self.l0(x)
		x = self.bn0(x)
		x = self.relu0(x)

		x = self.l1(x)
		x = self.bn1(x)
		x = self.relu1(x)

		x = self.l2(x)
		x = self.bn2(x)
		x = self.relu2(x)

		x = self.l3(x)
		x = self.bn3(x)
		x = self.relu3(x)

		x = self.l4(x)
		x = self.bn4(x)

		return F.log_softmax(x, dim=1)

	def registerHooks(self):
		if not self.legacyImportance:
			# Register hooks
			self.relu1.register_full_backward_hook(self.backward_hook_relu1)
			self.relu2.register_full_backward_hook(self.backward_hook_relu2)
			self.relu3.register_full_backward_hook(self.backward_hook_relu3)

			self.relu1.register_forward_hook(self.forward_hook_relu1)
			self.relu2.register_forward_hook(self.forward_hook_relu2)
			self.relu3.register_forward_hook(self.forward_hook_relu3)
		else:
			self.relu1.register_forward_hook(self.forward_hook_relu1)
			self.relu2.register_forward_hook(self.forward_hook_relu2)
			self.relu3.register_forward_hook(self.forward_hook_relu3)

	def backward_hook_relu1(self, module, grad_input, grad_output):
		self.gradientsReLU1.append(grad_input[0].cpu().detach().numpy()[0])

	def backward_hook_relu2(self, module, grad_input, grad_output):
		self.gradientsReLU2.append(grad_input[0].cpu().detach().numpy()[0])

	def backward_hook_relu3(self, module, grad_input, grad_output):
		self.gradientsReLU3.append(grad_input[0].cpu().detach().numpy()[0])

	def forward_hook_relu1(self, module, val_input, val_output):
		if not self.legacyImportance:
			self.valueReLU1.append(val_output[0].cpu().detach().numpy())
		else:
			if self.neuronSwitchedOff[0] == 1:
				val_output[0][self.neuronSwitchedOff[1]] = 0

	def forward_hook_relu2(self, module, val_input, val_output):
		if not self.legacyImportance:
			self.valueReLU2.append(val_output[0].cpu().detach().numpy())
		else:
			if self.neuronSwitchedOff[0] == 2:
				val_output[0][self.neuronSwitchedOff[1]] = 0

	def forward_hook_relu3(self, module, val_input, val_output):
		if not self.legacyImportance:
			self.valueReLU3.append(val_output[0].cpu().detach().numpy())
		else:
			if self.neuronSwitchedOff[0] == 3:
				val_output[0][self.neuronSwitchedOff[1]] = 0

	def computeImportance(self, neuronPerLayer):
		self.gradientsReLU1 = np.array(self.gradientsReLU1)
		self.gradientsReLU2 = np.array(self.gradientsReLU2)
		self.gradientsReLU3 = np.array(self.gradientsReLU3)

		self.valueReLU1 = np.array(self.valueReLU1)
		self.valueReLU2 = np.array(self.valueReLU2)
		self.valueReLU3 = np.array(self.valueReLU3)

		importanceReLU1 = abs(np.multiply(self.gradientsReLU1, self.valueReLU1))
		importanceReLU2 = abs(np.multiply(self.gradientsReLU2, self.valueReLU2))
		importanceReLU3 = abs(np.multiply(self.gradientsReLU3, self.valueReLU3))

		return [importanceReLU1, importanceReLU2, importanceReLU3]
