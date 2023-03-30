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
		i = 0
		n = len(self.neurons)
		for neuron in self.neurons:
			neuron.createTT()

			i += 1
			if (i + 1) % 25 == 0:
				print(f"{self.name} Neuron TT [{i + 1:>4d}/{n:>4d}]")

	def plotImportancePerNeuron(self):
		# Get neuron name and importance
		aux = pd.DataFrame()
		for neuron in self.neurons:
			aux = pd.concat([aux, pd.DataFrame({'name': neuron.name, 'importance': neuron.importance}, index=[0])],
							ignore_index=True)

		aux.sort_values(['importance'], inplace=True)

		fig = go.Figure()

		fig.add_trace(go.Bar(name='totalImportance', x=aux['name'], y=aux['importance']))

		fig.show()

	def plotImportancePerClass(self):
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

		fig.update_layout(barmode='stack', hovermode="x unified")
		fig.show()


