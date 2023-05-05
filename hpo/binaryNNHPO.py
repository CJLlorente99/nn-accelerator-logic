import torch

from models.steFunction import STEFunction
from hpo.quantizier import STEFunctionQuant
from torch import nn
import torch.nn.functional as F
import numpy as np
from ttUtilities.auxFunctions import binaryArrayToSingleValue, integerToBinaryArray


class BinaryNeuralNetwork(nn.Module):
	def __init__(self, hiddenLayers, npl):
		super(BinaryNeuralNetwork, self).__init__()
		self.flatten = nn.Flatten()

		self.l0 = nn.Linear(28 * 28, npl)
		self.bn0 = nn.BatchNorm1d(npl)
		self.ste0 = STEFunction()

		self.l = []
		self.bn = []
		self.ste = []

		for i in range(hiddenLayers):
			self.l.append(nn.Linear(npl, npl))
			self.bn.append(nn.BatchNorm1d(npl))
			self.ste.append(STEFunction())

		self.l4 = nn.Linear(npl, 10)
		self.bn4 = nn.BatchNorm1d(10)

	def forward(self, x):
		x = self.flatten(x)

		x = self.l0(x)
		x = self.bn0(x)
		x = self.ste0(x)

		for i in range(len(self.l)):
			x = self.l[i](x)
			x = self.bn[i](x)
			x = self.ste[i](x)

		x = self.l4(x)
		x = self.bn4(x)

		return F.log_softmax(x, dim=1)
