from dataclasses import dataclass
import pandas as pd
import plotly.graph_objects as go


@dataclass
class AccLayer:
	name: str
	linear: dict
	norm: dict
	nNeurons: int
	neurons: list
	tt: pd.DataFrame

	def fillTT(self):
		"""
		Method that calls each neuron in the layer recurrently and creates the TT per each
		"""
		i = 0
		n = len(self.neurons)
		for neuron in self.neurons:
			neuron.createTT()

			if (i + 1) % 1 == 0:
				print(f"{self.name} Neuron TT [{i + 1:>4d}/{n:>4d}]")
			i += 1

	def plotImportancePerNeuron(self, name):
		"""
		Method that plots the importance per neuron
		:param name:
		"""
		# Get neuron name and importance
		aux = pd.DataFrame()
		for neuron in self.neurons:
			aux = pd.concat([aux, pd.DataFrame({'name': neuron.name, 'importance': neuron.importance}, index=[0])],
							ignore_index=True)

		aux.sort_values(['importance'], inplace=True)

		fig = go.Figure()

		fig.add_trace(go.Bar(name='totalImportance', x=aux['name'], y=aux['importance']))

		fig.update_layout(title=name + ' ' + self.name)

		fig.show()

	def plotImportancePerClass(self, name):
		"""
		Method that plots the importance per neuron per layer
		:param name:
		"""
		# Get neuron name and importance per class
		aux = pd.DataFrame()
		for neuron in self.neurons:
			d = neuron.importancePerClass
			d['name'] = neuron.name
			d['importance'] = neuron.importance
			aux = pd.concat([aux, pd.DataFrame(d, index=[0])], ignore_index=True)

		aux.sort_values(['importance'], inplace=True)

		fig = go.Figure()

		for col in aux:
			if col not in ['name', 'importance']:
				fig.add_trace(go.Bar(name=col, x=aux['name'], y=aux[col]))

		fig.update_layout(title=name + ' ' + self.name, barmode='stack', hovermode="x unified")
		fig.show()

	def saveTT(self, filename):
		"""
		Method that saves the TT into a feather file
		:param filename:
		"""
		outputTags = [col for col in self.tt if col.startswith('output')]
		activationTags = [col for col in self.tt if col.startswith('activation')]
		lengthActivationTags = [col for col in self.tt if col.startswith('lengthActivation')]

		self.tt[outputTags + activationTags + lengthActivationTags].to_feather(filename)
