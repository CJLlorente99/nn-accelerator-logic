import math
from dataclasses import dataclass
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

	def __post_init__(self):
		self.fanIn = len(self.weight)
		self.tt = self._generateTT()

	def _generateTT(self):
		# Create tt
		inpTags = [str(i) for i in range(self.fanIn)]
		tt = ttg.Truths(inpTags).as_pandas()

		# For each entry, calculate output
		tt['out'] = tt.apply(self._neuronAction, axis=1)

		return tt

	def _neuronAction(self, row):
		# Multiply per weights
		z = row @ self.weight + self.bias
		# Batch normalization
		zNorm = (z - self.normMean) / (math.sqrt(self.normVar + 1e-5)) * self.normWeight + self.normBias
		# Activation layer
		a = (zNorm > 0).float().item()

		return a


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
		for layer in model.stack:
			if isinstance(layer, modules.linear.Linear):
				n += 1

		layer = 0  # Layer (as PyTorch understands it)
		accLayer = 0  # Layer (as neuron layer, Linear+Norm+Activation)
		accLayers = [AccLayer({}, {}, 0) for _ in range(n)]

		for entry in model.state_dict():
			param = model.state_dict()[entry]
			if isinstance(model.stack[layer], modules.linear.Linear):
				if entry.endswith('weight'):
					accLayers[accLayer].linear['weight'] = param
					accLayers[accLayer].nNeurons = len(param)
				elif entry.endswith('bias'):
					accLayers[accLayer].linear['bias'] = param
					layer += 1

			if isinstance(model.stack[layer], modules.batchnorm.BatchNorm1d):
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
		neurons = []
		for accLayer in accLayers:
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
					nLayer=layer
				)
				neurons.append(neuron)
			layer += 1
		return neurons


