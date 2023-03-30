'''
This file contains the methods needed to realize the NN into a HW compatible realization
'''
import pandas as pd
import ttg

tt = pd.DataFrame()


def initializeGlobalTT(nNeurons: int):
	global tt
	tt = ttg.Truths([f'n {n}' for n in range(nNeurons)]).as_pandas()
	resetOutputTT()


def resetOutputTT():
	global tt
	tt['output'] = 0


def assignActivations(activations: pd.DataFrame):
	# Probably a more efficient way of doing it exists
	global tt
	data = activations.drop(['output'], axis=1)
	for index, row in data.iterrows():
		tt[tt[data.columns] == row]['output'] = activations['output']


def assignOutputBasedOnDistance():
	# 1NN (maybe try with different distance definitions)


def optimizeNeuron(tt: pd.DataFrame, neuronNames: list, outputName: str):
	tt = tt[neuronNames + list(outputName)]

	pass
