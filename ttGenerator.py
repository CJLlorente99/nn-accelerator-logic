import math
from dataclasses import dataclass
from operator import itemgetter
import numpy as np
import torch
import torch.nn.modules as modules
import ttg
import pandas as pd

'''
This class should provide static methods so, when provided a binarized trained model, the TT per neuron can be retrieved
'''


@dataclass
class BinaryOutputNeuron:
	weight: torch.Tensor
	bias: torch.Tensor
	normWeight: torch.Tensor
	normBias: torch.Tensor
	normMean: torch.Tensor
	normVar: torch.Tensor
	nNeuron: int
	nLayer: int
	nClass: int

	def __post_init__(self):
		self.fanIn = len(self.weight)
		self.tt = pd.DataFrame()
		self.importancePerClass = {}
		self.importance = 0
		self.activations = pd.DataFrame()
		self.name = 'L' + str(self.nLayer) + 'N' + str(self.nNeuron)

	def _neuronAction(self, row):
		row = row[['n' + str(i) for i in range(self.fanIn)]].to_numpy(dtype=np.float32)
		# Multiply per weights
		z = torch.from_numpy(row).type(torch.FloatTensor) @ self.weight + self.bias
		# Batch normalization
		zNorm = (z - self.normMean) / (math.sqrt(self.normVar + 1e-5)) * self.normWeight + self.normBias
		# Activation layer
		a = (zNorm > 0).float().item()

		return a

	def _importanceToTT(self, row):
		row = row[['class' + str(i) for i in range(self.nClass)]]

		imp = 0
		for col in row.index:
			if row[col] != 0:
				imp += self.importancePerClass[col]

		return imp

	def giveImportance(self, importance: np.array, targets):
		dictImportance = {}
		nPerClass = {}
		for n in targets:
			dictImportance[n] = []
			self.importancePerClass['class' + str(n)] = 0
			nPerClass[n] = 0

		for i in range(len(importance)):
			nPerClass[targets[i]] += 1
			if importance[i] > 10e-50:
				dictImportance[targets[i]].append(importance[i])

		for n in dictImportance:
			self.importancePerClass['class' + str(n)] = len(dictImportance[n]) / nPerClass[n]
			self.importance += len(dictImportance[n]) / nPerClass[n]

	def createTT(self, activations: pd.DataFrame):
		self.tt = activations
		self.tt['output'] = self.tt.apply(self._neuronAction, axis=1)
		self.tt['importance'] = self.tt.apply(self._importanceToTT, axis=1)

		for col in self.tt:
			if col.startswith('class'):
				self.tt[col] = self.tt[col].apply(lambda column: float(column > 0))

	@staticmethod
	def getDfLayer(neurons, layer):
		res = pd.DataFrame()
		for neuron in neurons['layer'+str(layer)]:
			aux0 = pd.DataFrame()
			for i in neuron.importancePerClass:
				aux0['importanceClass' + str(i)] = [neuron.importancePerClass[i]]
			aux = pd.DataFrame({
				'name': neuron.name,
				'importance': neuron.importance
			}, index=[0])
			aux = pd.concat([aux, aux0], axis=1)

			res = pd.concat([res, aux], ignore_index=True)
		return res


@dataclass
class AccLayer:
	linear: dict
	norm: dict
	nNeurons: int


class TTGenerator:
	@staticmethod
	def getAccLayers(model: torch.nn.Module):
		# Count number of layers (by counting number of Linear)
		n = 0
		for layer in model.children():
			if isinstance(layer, modules.linear.Linear):
				n += 1

		layer = 0  # Layer (as PyTorch understands it)
		accLayer = 0  # Layer (as neuron layer, Linear+Norm+Activation)
		accLayers = [AccLayer({}, {}, 0) for _ in range(n)]

		for entry in model.state_dict():
			param = model.state_dict()[entry]
			if entry.startswith('l'):  # Linear
				if entry.endswith('weight'):
					accLayers[accLayer].linear['weight'] = param
					accLayers[accLayer].nNeurons = len(param)
				elif entry.endswith('bias'):
					accLayers[accLayer].linear['bias'] = param
					layer += 1

			if entry.startswith('bn'):  # batch normalization
				if entry.endswith('weight'):
					accLayers[accLayer].norm['weight'] = param
				elif entry.endswith('bias'):
					accLayers[accLayer].norm['bias'] = param
				elif entry.endswith('running_mean'):
					accLayers[accLayer].norm['running_mean'] = param
				elif entry.endswith('running_var'):
					accLayers[accLayer].norm['running_var'] = param
				elif entry.endswith('num_batches_tracked'):
					accLayers[accLayer].norm['num_batches_tracked'] = param
					layer += 2
					accLayer += 1

		return accLayers

	@staticmethod
	def getNeurons(accLayers: list):
		layer = 0
		neurons = {}
		for accLayer in accLayers:
			neuronsLayer = []
			if layer == 0:
				layer += 1
				continue

			for iNeuron in range(accLayer.nNeurons):
				neuron = BinaryOutputNeuron(
					weight=accLayer.linear['weight'][iNeuron, :],
					bias=accLayer.linear['bias'][iNeuron],
					normWeight=accLayer.norm['weight'][iNeuron],
					normBias=accLayer.norm['bias'][iNeuron],
					normMean=accLayer.norm['running_mean'][iNeuron],
					normVar=accLayer.norm['running_var'][iNeuron],
					nNeuron=iNeuron,
					nLayer=layer,
					nClass=10
				)
				neuronsLayer.append(neuron)
			neurons['layer' + str(layer)] = neuronsLayer
			layer += 1
		return neurons


