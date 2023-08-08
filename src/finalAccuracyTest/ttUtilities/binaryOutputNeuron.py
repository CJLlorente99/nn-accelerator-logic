import math
from dataclasses import dataclass
import numpy as np
import torch
import pandas as pd
from ttUtilities.accLayer import AccLayer
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import random
from ttUtilities.auxFunctions import integerToBinaryArray


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
	accLayer: AccLayer

	def __post_init__(self):
		self.fanIn = len(self.weight)
		self.importancePerClass = {}
		self.importance = 0
		self.name = 'L' + str(self.nLayer) + 'N' + str(self.nNeuron)
		self.sopForm = ''

	def giveImportance(self, importance: np.array, targets, threshold=10e-4):
		"""
		Method that provides the neuron with the importance values
		:param importance:
		:param targets:
		"""
		dictImportance = {}
		nPerClass = {}
		for n in targets:
			dictImportance[n] = []
			self.importancePerClass['class' + str(n)] = 0
			nPerClass[n] = 0

		for i in range(len(importance)):
			nPerClass[targets[i]] += 1
			if importance[i] > threshold:
				dictImportance[targets[i]].append(importance[i])

		for n in dictImportance:
			self.importancePerClass['class' + str(n)] = len(dictImportance[n]) / nPerClass[n] * 100
			self.importance += len(dictImportance[n]) / nPerClass[n] * 100

	def createTT(self, activation):
		"""
		Method that adds new columns to the layer truth table corresponding to the importance and output of the neuron
		"""
		if not activation:
			self.accLayer.tt['output' + self.name] = self.accLayer.tt.apply(self._neuronAction, axis=1)
		else:
			self.accLayer.tt['output' + self.name] = activation
		self.accLayer.tt['importance' + self.name] = self.accLayer.tt.apply(self._importanceToTT, axis=1)

		for col in self.accLayer.tt:
			if col.startswith('class'):
				# Depending on approach
				# self.accLayer.tt[col] = self.accLayer.tt[col].apply(lambda column: np.uint8(column > 0))
				self.accLayer.tt['importance' + col + self.name] = np.float32(self.accLayer.tt[col] * self.importancePerClass[col])

	def returnTTImportanceStats(self):
		"""
		Method that returns some metrics related to the importance of the neuron
		:return:
		"""
		# Return distribution of number of importances per entry
		numImportances = self.accLayer.tt.apply(self._countImportances, axis=1)
		return self.name[1], self.name, numImportances.mean(), len(numImportances)

	def showImportancePerEntry(self):
		"""
		Method that plots the importance per entry of the TT
		"""
		fig = go.Figure()

		tt = self.accLayer.tt
		tt.sort_values(['importance' + self.name], inplace=True)
		fig.add_trace(go.Scatter(name='totalImportance', x=tt.index, y=tt['importance' + self.name]))

		fig.update_xaxes(type='category')
		fig.update_layout(
			barmode='stack',
			title=f'Neuron L{self.nLayer}N{self.nNeuron}',
			hovermode="x unified"
		)
		fig.show()

	def showImportancePerClassPerEntry(self):
		"""
		Method that plots the importance per entry per class of the TT
		"""
		fig = make_subplots(rows=3, cols=4)

		tt = self.accLayer.tt
		tt.sort_index(inplace=True)
		i = 0
		for col in tt:
			if col.startswith('importanceclass') and col.endswith(self.name):
				fig.add_trace(go.Scatter(name=col, x=tt.index, y=tt[col]),
							  row=(i // 4) + 1, col=(i % 4) + 1)
				i += 1

		fig.update_xaxes(type='category')
		fig.update_layout(
			barmode='stack',
			title=f'Neuron L{self.nLayer}N{self.nNeuron}',
			hovermode="x unified"
		)
		fig.show()

	def _neuronAction(self, row):
		"""
		Private method that performs the mimics the calculation of the neuron represented
		:param row:
		:return:
		"""
		tags = [col for col in row.index if col.startswith('activation')]
		lengthsTags = [col for col in row.index if col.startswith('lengthActivation')]
		row = np.array(integerToBinaryArray(row[tags].values, row[lengthsTags].values))
		# Pad zeros at the beginning for length consistency
		row = np.pad(row.squeeze(), (len(self.weight) - len(row), 0))
		# Multiply per weights
		z = torch.from_numpy(row).type(torch.FloatTensor) @ self.weight + self.bias
		# Batch normalization
		zNorm = (z - self.normMean) / (math.sqrt(self.normVar + 1e-5)) * self.normWeight + self.normBias
		# Activation layer
		a = (zNorm > 0).float().item()

		return np.uint8(a)

	def _importanceToTT(self, row):
		"""
		Private method that sums the importance of every class into a total importance
		:param row:
		:return:
		"""
		row = row[['class' + str(i) for i in range(self.nClass)]]

		imp = 0
		for col in row.index:
			if row[col] != 0:
				imp += row[col] * self.importancePerClass[col]

		return np.float32(imp)

	def _countImportances(self, row):
		"""
		Private method that counts the number of times a neuron is important for every class
		:param row:
		:return:
		"""
		row = row[['importanceclass' + str(i) + self.name for i in range(self.nClass)]]
		return (row > 0).sum()
