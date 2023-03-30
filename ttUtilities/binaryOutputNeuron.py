import math
from dataclasses import dataclass
import numpy as np
import torch
import pandas as pd
from ttUtilities.accLayer import AccLayer
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import random


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

	def _neuronAction(self, row):
		row = row[['n' + str(i) for i in range(self.fanIn)]].to_numpy(dtype=np.float32)
		# Multiply per weights
		z = torch.from_numpy(row).type(torch.FloatTensor) @ self.weight + self.bias
		# Batch normalization
		zNorm = (z - self.normMean) / (math.sqrt(self.normVar + 1e-5)) * self.normWeight + self.normBias
		# Activation layer
		a = (zNorm > 0).float().item()

		return np.uint8(a)

	def _importanceToTT(self, row):
		row = row[['class' + str(i) for i in range(self.nClass)]]

		imp = 0
		for col in row.index:
			if row[col] != 0:
				imp += self.importancePerClass[col]

		return np.float32(imp)

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

	def createTT(self):
		self.accLayer.tt['output' + self.name] = self.accLayer.tt.apply(self._neuronAction, axis=1)
		self.accLayer.tt['importance' + self.name] = self.accLayer.tt.apply(self._importanceToTT, axis=1)

		for col in self.accLayer.tt:
			if col.startswith('class'):
				# Depending on approach
				# self.accLayer.tt[col] = self.accLayer.tt[col].apply(lambda column: np.uint8(column > 0))
				self.accLayer.tt['importance' + col + self.name] = np.float32(self.accLayer.tt[col] * self.importancePerClass[col])

	def _countImportances(self, row):
		row = row[['importanceclass' + str(i) + self.name for i in range(self.nClass)]]
		return (row > 0).sum()

	def returnTTImportanceStats(self):
		# Return distribution of number of importances per entry
		numImportances = self.accLayer.tt.apply(self._countImportances, axis=1)
		return self.name[1], self.name, numImportances.mean(), len(numImportances)

	def reduceTTByImportance(self, importanceThreshold):
		self.tt = self.tt.query('importance >= @importanceThreshold')

	def ttPCA(self):
		inputs = torch.Tensor(self.tt[['n' + str(i) for i in range(self.fanIn)]])
		U, S, V = torch.pca_lowrank(inputs)

	def showImportancePerEntry(self):
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
