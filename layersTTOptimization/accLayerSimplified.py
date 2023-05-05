from dataclasses import dataclass
import pandas as pd
import numpy as np
from ttUtilities.auxFunctions import binaryArrayToSingleValue
import plotly.graph_objects as go


@dataclass
class AccLayer:
	name: str
	nNeurons: int
	tt: pd.DataFrame

	def __post_init__(self):
		self.neurons = {}
		self.ttClassTags = [col for col in self.tt if col.startswith('class')]
		self.ttClassTags.sort()

	def reduceTTThreshold(self, threshold: float = 0):
		# Let's use matrices to speed things up
		# First, an ordered matrix of classes per entry
		ttOrderedClasses = self.tt[self.ttClassTags].to_numpy().reshape((len(self.tt), 1, len(self.ttClassTags)))
		ttOrderedClasses = np.broadcast_to(ttOrderedClasses, (len(self.tt), self.nNeurons, len(self.ttClassTags)))
		ttOrderedClasses = np.transpose(ttOrderedClasses, (1, 0, 2))

		# Second, ordered list of importance per neuron
		aux = pd.DataFrame()
		importanceTags = list(next(iter(self.neurons.values())).importancePerClass.keys())
		importanceTags.sort()
		for neuron in self.neurons.values():
			aux = pd.concat([aux, pd.DataFrame(neuron.importancePerClass, index=[0])], ignore_index=True)

		aux = aux[importanceTags].to_numpy().reshape((self.nNeurons, 1, len(self.ttClassTags)))
		aux = np.broadcast_to(aux, (self.nNeurons, len(self.tt), len(self.ttClassTags)))

		# (TTentries, neurons, Classes)
		result = np.multiply(ttOrderedClasses, aux)
		result = np.transpose(result, (1, 0, 2))
		result = np.sum(result, axis=2)  # We care about the sum of all importance belonging to one entry
		result = result > threshold

		metrics = np.sum(result, axis=0)

		dataDiscrimination = []
		for i in range(result.shape[0]):
			dataDiscrimination.append(binaryArrayToSingleValue(result[i, :], 30))

		discriminationTags = ['discriminator' + str(i) for i in range(int(len(dataDiscrimination[0]) / 2))]
		discriminationTags = discriminationTags + ['lengthDiscriminator' + str(i) for i in range(int(len(dataDiscrimination[0]) / 2))]

		self.tt[discriminationTags] = dataDiscrimination

		i = 0
		for neuron in self.neurons.values():
			neuron.activeTTEntries = metrics[i]
			i += 1

	def saveTT(self, filename):
		"""
		Method that saves the TT into a feather file
		:param filename:
		"""
		outputTags = [col for col in self.tt if col.startswith('output')]
		activationTags = [col for col in self.tt if col.startswith('activation')]
		lengthActivationTags = [col for col in self.tt if col.startswith('lengthActivation')]
		discriminatorTags = [col for col in self.tt if col.startswith('discriminator')]
		lengthDiscriminatorTags = [col for col in self.tt if col.startswith('lengthDiscriminator')]

		self.tt[outputTags + activationTags + lengthActivationTags + discriminatorTags + lengthDiscriminatorTags].to_feather(filename)

	def plotGainsPerNeuron(self):
		aux = pd.DataFrame(columns=['name', 'entryDiscrimination'])
		for neuron in self.neurons.values():
			data = {'name': neuron.name, 'entryDiscrimination': len(self.tt) - neuron.activeTTEntries}
			aux = pd.concat([aux, pd.DataFrame(data, index=[0])], ignore_index=True)

		aux = aux.set_index(['name'])

		fig = go.Figure()

		fig.add_trace(go.Bar(x=aux.index, y=aux['entryDiscrimination']))

		fig.update_xaxes(type='category', tickmode='linear')
		fig.update_layout(title='Number of discounted entries per neuron')

		fig.show()
