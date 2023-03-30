import torch
import torch.nn.modules as modules
import pandas as pd
from ttUtilities.accLayer import AccLayer
from ttUtilities.binaryOutputNeuron import BinaryOutputNeuron

'''
This class should provide static methods so, when provided a binarized trained model, the TT per neuron can be retrieved
'''


class HelpGenerator:
	@staticmethod
	def getAccLayers(model: torch.nn.Module):
		# Count number of layers (by counting number of Linear)
		n = 0
		for layer in model.children():
			if isinstance(layer, modules.linear.Linear):
				n += 1

		layer = 0  # Layer (as PyTorch understands it)
		accLayer = 0  # Layer (as neuron layer, Linear+Norm+Activation)
		accLayers = [AccLayer(f'Layer {i}',{}, {}, 0, [], pd.DataFrame()) for i in range(n)]

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
		layer = 1
		for accLayer in accLayers:
			neuronsLayer = []

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
					nClass=10,
					accLayer=accLayer
				)
				layer += 1
				neuronsLayer.append(neuron)

			accLayer.neurons = neuronsLayer
