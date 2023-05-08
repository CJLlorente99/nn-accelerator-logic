from dataclasses import dataclass
import pandas as pd
import plotly.graph_objects as go
import numpy as np

@dataclass
class AccLayer:
	name: str
	linear: dict
	norm: dict
	nNeurons: int
	neurons: list
	tt: pd.DataFrame

	def fillTT(self, activations: pd.DataFrame):
		"""
		Method that calls each neuron in the layer recurrently and creates the TT per each
		"""
		i = 0
		n = len(self.neurons)
		for neuron in self.neurons:
			if activations != None:
				neuron.createTT(activations[neuron.name])
			else:
				neuron.createTT(None)

			if (i + 1) % 1 == 0:
				print(f"{self.name} Neuron TT [{i + 1:>4d}/{n:>4d}]")
			i += 1

	def plotImportancePerNeuron(self, name, ordered=False):
		"""
		Method that plots the importance per neuron
		:param name:
		"""
		# Get neuron name and importance
		aux = pd.DataFrame()
		for neuron in self.neurons:
			aux = pd.concat([aux, pd.DataFrame({'name': neuron.name, 'importance': neuron.importance}, index=[0])],
							ignore_index=True)

		if ordered:
			aux.sort_values(['importance'], inplace=True)

		fig = go.Figure()

		fig.add_trace(go.Scatter(name='totalImportance', x=aux['name'], y=aux['importance']))

		fig.update_layout(title=name + ' ' + self.name)

		fig.show()

	def plotImportancePerClass(self, name, ordered=False):
		"""
		Method that plots the importance per neuron per layer
		:param name:
		"""
		# Get neuron name and importance per class
		aux = pd.DataFrame()
		for neuron in self.neurons:
			d = neuron.importancePerClass.copy()
			d['name'] = neuron.name
			d['importance'] = neuron.importance
			aux = pd.concat([aux, pd.DataFrame(d, index=[0])], ignore_index=True)

		for col in aux:
			if col not in ['name', 'importance']:
				if ordered:
					aux.sort_values(col, inplace=True)

				fig = go.Figure()

				fig.add_trace(go.Scatter(name=col, x=aux['name'], y=aux[col]))

				fig.update_layout(title=name + ' ' + col, barmode='stack', hovermode="x unified")
				fig.show()

	def plotNumImportantClasses(self, name, ordered=False):
		"""
		Method that plots the importance per neuron per layer
		:param name:
		"""
		# Get neuron name and importance per class
		aux = pd.DataFrame()
		for neuron in self.neurons:
			d = neuron.importancePerClass.copy()
			d['numImportant'] = (np.array(list(neuron.importancePerClass.values())) > 0).sum()
			d['name'] = neuron.name
			aux = pd.concat([aux, pd.DataFrame(d, index=[0])], ignore_index=True)

		if ordered:
			aux.sort_values(['numImportant'], inplace=True)

		fig = go.Figure()

		fig.add_trace(go.Scatter(name='Number of Important Classes', x=aux['name'], y=aux['numImportant']))

		# fig.update_xaxes(type='category', tickmode='linear')
		fig.update_layout(title='Number of Important Classes' + ' ' + self.name, barmode='stack', hovermode="x unified")
		fig.show()

	def saveTT(self, filename):
		"""
		Method that saves the TT into a feather file
		:param filename:
		"""
		outputTags = [col for col in self.tt if col.startswith('output')]
		activationTags = [col for col in self.tt if col.startswith('activation')]
		lengthActivationTags = [col for col in self.tt if col.startswith('lengthActivation')]
		classTags = [col for col in self.tt if col.startswith('class')]

		self.tt[outputTags + activationTags + lengthActivationTags + classTags].to_feather(filename)

	def saveImportance(self, filename):
		"""
		Method that saves the importance per neuron into a feather file
		:param filename:
		"""
		# Get neuron name and importance per class
		aux = pd.DataFrame()
		for neuron in self.neurons:
			d = neuron.importancePerClass
			d['name'] = neuron.name
			d['importance'] = neuron.importance
			aux = pd.concat([aux, pd.DataFrame(d, index=[0])], ignore_index=True)

		aux.to_feather(filename)

	def fillImportanceDf(self, df: pd.DataFrame):
		i = 0
		for n in self.neurons:
			imp = df.query('name == @n.name')
			n.importance = imp['importance']
			n.importancePerClass = imp.drop(['importance', 'name'], axis=1).iloc[0].to_dict()

			if (i + 1) % 250 == 0:
				print(f"{self.name} Fill Importance From Df [{i + 1:>4d}/{len(self.neurons):>4d}]")
			i += 1
       
		