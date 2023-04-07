'''
This file contains the methods needed to realize the NN into a HW compatible realization
'''
import pandas as pd
import ttg
import itertools
from sklearn.neighbors import NearestNeighbors
from sympy.logic import SOPform
from sympy import symbols
from ttUtilities.auxFunctions import integerToBinaryArray, binaryArrayToSingleValue


class DNFRealization:
	def __init__(self, nNeurons):
		self.tt = pd.DataFrame()
		self.nNeurons = nNeurons
		self.outputTags = []
		self.activationTags = []

	def loadTT(self, filename):
		"""
		Method that loads a TT from a feather file
		:param filename:
		"""
		self.tt = pd.read_feather(filename)
		self.outputTags = [col for col in self.tt if col.startswith('output')]
		self.activationTags = [col for col in self.tt if col.startswith('activation')]

	def assignOutputBasedOnDistance(self, df: pd.DataFrame):
		# 1NN (maybe try with different distance definitions)
		# P=2 euclidean, P=1 manhattan_distance
		setOnOff = df.drop(['output'], axis=1).apply(self._toBinaryArray, axis=1)
		outputsOnOff = df['output']
		nbrs = NearestNeighbors(n_neighbors=1, metric='minkowski', p=2).fit(setOnOff.to_numpy())

		# Find nearest neighbor of each entry in DC-set and add it to tt
		# Leverage generator so whole DC-set is not created
		gen = itertools.product([0, 1], repeat=self.nNeurons)
		while True:
			try:
				entry = next(gen)
				if entry not in setOnOff:
					aux = pd.DataFrame([binaryArrayToSingleValue(entry) + outputsOnOff[nbrs.kneighbors(entry).squeeze()]],
									   columns=df.columns, index=[0])
					df = pd.concat([df, aux], ignore_index=True)
			except StopIteration:
				break

		return df

	def generateSoP(self, df: pd.DataFrame):
		# Get only ON-set
		df.drop(self.tt[self.tt['output'] != 1].index, inplace=True).drop(['output'], axis=1)

		# Generate symbols
		symb = []
		for n in range(self.nNeurons):
			symb.append(symbols(f'n{n}'))

		# Generate SoP
		return SOPform(symb, df.to_numpy())

	def realizeNeurons(self):
		realizations = []
		for neuron in self.outputTags:
			df = self.tt[self.activationTags + neuron].copy()
			df.rename(columns={neuron: 'output'}, inplace=True)
			df = self.assignOutputBasedOnDistance(df)
			realizations.append(self.generateSoP(df))
			del df  # caring about memory
		return realizations

	def _toBinaryArray(self, row):
		row = row[self.activationTags]
		return integerToBinaryArray(row)
