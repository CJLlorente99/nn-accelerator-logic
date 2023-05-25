'''
This file contains the methods needed to realize the NN into a HW compatible realization
'''
import pandas as pd
import itertools
from sklearn.neighbors import NearestNeighbors
from sympy.logic import SOPform
from sympy import symbols
from ttUtilities.auxFunctions import integerToBinaryArray, binaryArrayToSingleValue
import numpy as np


class DNFRealization:
	def __init__(self, nNeurons):
		self.tt = pd.DataFrame()
		self.nNeurons = nNeurons
		self.outputTags = []
		self.activationTags = []
		self.lengthActivationTags = []
		self.discriminationTags = []
		self.lengthDiscriminationTags = []
		self.tag = [f'N{i}' for i in range(self.nNeurons)]
		self.discrimData = None
		self.setOnOff = None

	def loadTT(self, filename: str):
		"""
		Method that loads a TT from a feather file
		:param filename:
		"""
		self.tt = pd.read_feather(filename)
		self.outputTags = [col for col in self.tt if col.startswith('output')]
		self.activationTags = [col for col in self.tt if col.startswith('activation')]
		self.lengthActivationTags = [col for col in self.tt if col.startswith('lengthActivation')]
		self.discriminationTags = [col for col in self.tt if col.startswith('discriminator')]
		self.lengthDiscriminationTags = [col for col in self.tt if col.startswith('lengthDiscriminator')]

		print(f'Unrolling to binary array')
		setOnOff = self.tt.drop(self.outputTags + self.discriminationTags + self.lengthDiscriminationTags, axis=1).apply(self._toBinaryArrayActivations, axis=1)
		self.setOnOff = np.array([np.array(i) for i in setOnOff])
		self.setOnOff = pd.DataFrame(self.setOnOff, columns=self.tag)
		print(f'Unrolled to binary array')

		if self.discriminationTags:
			print(f'Unrolling discriminator')
			discrimData = self.tt.drop(self.outputTags + self.activationTags + self.lengthActivationTags, axis=1).apply(self._toBinaryArrayDiscriminator, axis=1)
			self.discrimData = np.array([np.array(i) for i in discrimData])
			print(f'Unrolled discriminator')

		self.tt = self.tt[self.outputTags]

	def assignOutputBasedOnDistance(self, df: pd.DataFrame):
		"""
		This method receives a df with the ON-set and OFF-set and calculated the values of the DC-set based on the
		nearest distance
		:param df:
		:return:
		"""
		print(f'Unrolling to binary array')
		setOnOff = df.drop(['output'], axis=1).apply(self._toBinaryArrayActivations, axis=1)
		setOnOff = np.array([np.array(i) for i in setOnOff])
		outputsOnOff = df['output']

		# 1NN (maybe try with different distance definitions)
		# P=2 euclidean, P=1 manhattan_distance
		print(f'Fitting Nearest Neighbor')
		nbrs = NearestNeighbors(n_neighbors=1, metric='minkowski', p=2).fit(setOnOff)
		setOnOff = pd.DataFrame(setOnOff, columns=self.tag)

		# Find nearest neighbor of each entry in DC-set and add it to tt
		# Leverage generator so whole DC-set is not created
		gen = itertools.product([0, 1], repeat=self.nNeurons)
		i = len(setOnOff)
		while True:
			try:
				entry = next(gen)
				y = pd.DataFrame(np.array(entry).reshape((1, self.nNeurons)), columns=self.tag, index=[0])
				if not (setOnOff == y.values).sum(axis=1).sum() == self.nNeurons:
					# TODO. Check it is correctly done
					aux = pd.DataFrame([binaryArrayToSingleValue(entry) + [outputsOnOff[nbrs.kneighbors(y.values)[1].squeeze().tolist()]]],
									   columns=df.columns, index=[0])
					df = pd.concat([df, aux], ignore_index=True)
				if (i + 1) % 500 == 0:
					print(f"AssignOutputsBasedOnDistance [{i + 1:>5d}/{2**self.nNeurons:>5d}]")
				i += 1
			except StopIteration:
				break

		return df

	def generateSoP(self, df: pd.DataFrame):
		"""
		Method that generated the SoP expression making use of the SymPy library
		:param df:
		:return:
		"""
		# Get only ON-set
		df = df.drop(df[df['output'] != 1].index).drop(['output'], axis=1)

		# Generate symbols
		symb = []
		for n in range(self.nNeurons):
			symb.append(symbols(f'n{n}'))

		# Generate SoP
		return SOPform(symb, df.to_numpy().tolist())

	def createBinaryOutput(self, df: pd.DataFrame, filename: str):
		"""
		This method receives a df with the ON-set and OFF-set and calculated the values of the DC-set based on the
		nearest distance
		:param df:
		:param filename:
		:return:
		"""
		dcSymbol = '~'
		dcCounter = 0

		gen = itertools.product([0, 1], repeat=self.nNeurons)
		i = 0
		with open(f'{filename}.binout', 'w') as f:
			while True:
				try:
					entry = next(gen)
					entry = -(np.array(entry) - 1)
					if not ((entry == df[self.tag]).all(1)).any():
						dcCounter += 1
					else:
						f.write(dcSymbol * dcCounter)
						dcCounter = 0
						f.write(df[df[self.tag] == entry]['output'])  # TODO. Probably bugged

					if (i + 1) % 500 == 0:
						print(f"binaryOutput [{i + 1:>5d}/{2 ** self.nNeurons:>5d}]")
					i += 1
				except StopIteration:
					break

	def createBinaryOutputRepresentation(self, baseFilename: str):
		i = 0
		for neuron in self.outputTags:
			df = self.tt[self.activationTags + self.lengthActivationTags + [neuron]].copy()
			df.rename(columns={neuron: 'output'}, inplace=True)

			print(f'Unrolling to binary array')
			setOnOff = df.drop(['output'], axis=1).apply(self._toBinaryArrayActivations, axis=1)
			setOnOff = np.array([np.array(i) for i in setOnOff])
			outputsOnOff = df['output']
			df = pd.concat([pd.DataFrame(setOnOff, columns=self.tag), outputsOnOff], axis=1)
			df = df.astype('int')

			self.createBinaryOutput(df, f'{baseFilename}/{neuron}')
			del df  # caring about memory
			print(f"Realize Espresso neurons [{i + 1:>5d}/{len(self.outputTags):>5d}]")
			i += 1

	def generateEspressoInput(self, df: pd.DataFrame, filename: str):
		"""
		Method that parses an ON-set and OFF-set defined TT into a PLA file that can be processed by Espresso SW
		:param df:
		:param filename:
		"""
		with open(f'{filename}.pla', 'w') as f:
			# Write header of PLA file
			f.write(f'.i {self.nNeurons}\n')  # Number of input neurons
			f.write(f'.o {1}\n')  # Number of output per neuron (just 1)
			tags = [f'N{i}' for i in range(self.nNeurons)]
			tags = ' '.join(tags)
			f.write(f'.ilb {tags}\n')  # Names of the input variables
			f.write(f'.ob output\n')  # Name of the output variable
			f.write(f'.type fr\n')  # .pla contains ON-Set and OFF-Set
			for index, row in df.iterrows():
				text = ''.join(row[self.tag].to_string(header=False, index=False).split('\n'))
				f.write(f'{text} {row.output}\n')
			f.write(f'.e')

	def generateABCInput(self, df: pd.DataFrame, filename: str, neuron: str):
		"""
		Method that parses an ON-set and OFF-set defined TT into a file that can be processed by ABC SW
		:param df:
		:param filename;
		"""
		# ABC only assumes fd type in pla file. This means, the data in pla represents the ON-Set (1) and the
		# DC-Set (-). As DC-Set is not supported, PLA file should only contain (1)

		with open(f'{filename}.pla', 'w') as f:
			# Write header of PLA file
			f.write(f'.i {self.nNeurons}\n')  # Number of input neurons
			f.write(f'.o {1}\n')  # Number of output per neuron (just 1)
			# tags = [f'N{i}' for i in range(self.nNeurons)]
			# tags = ' '.join(tags)
			# f.write(f'.ilb {tags}\n')  # Names of the input variables
			# f.write(f'.ob {neuron}\n')  # Name of the output variable
			f.write(f'.p {len(df)}\n')
			for index, row in df.iterrows():
				text = ''.join(row[self.tag].to_string(header=False, index=False).split('\n'))
				output = row[neuron]
				f.write(f'{text} {output}\n')
			f.write(f'.e')

	def realizeNeurons(self):
		realizations = []
		i = 0
		for neuron in self.outputTags:
			df = self.tt[self.activationTags + self.lengthActivationTags + [neuron]].copy()
			df.rename(columns={neuron: 'output'}, inplace=True)

			print(f'Unrolling to binary array')
			setOnOff = df.drop(['output'], axis=1).apply(self._toBinaryArrayActivations, axis=1)
			setOnOff = np.array([np.array(i) for i in setOnOff])
			outputsOnOff = df['output']
			df = pd.concat([pd.DataFrame(setOnOff, columns=self.tag), outputsOnOff], axis=1)

			# df = self.assignOutputBasedOnDistance(df)
			realizations.append(self.generateSoP(df))
			del df  # caring about memory
			print(f"Realize neurons [{i + 1:>5d}/{len(self.outputTags):>5d}]")
			i += 1
		return realizations

	def createPLAFileEspresso(self, baseFilename: str, discriminated: bool = False):
		i = 0
		for neuron in self.outputTags:
			df = self.tt[neuron].copy()
			df = pd.concat([df, self.setOnOff], axis=1)
			# df.rename(columns={neuron: 'output'}, inplace=True)

			if discriminated:
				print(f'Applying discriminator')
				discrimData = self.discrimData[:, i]
				df['drop'] = discrimData
				df = df[df['drop'] == 1]

			df = df.astype('int')

			self.generateEspressoInput(df, f'{baseFilename}{neuron}')
			del df  # caring about memory
			print(f"Realize Espresso neurons [{i + 1:>5d}/{len(self.outputTags):>5d}]")
			i += 1

	def createPLAFileABC(self, baseFilename: str, discriminated: bool = False):
		i = 0
		for neuron in self.outputTags:
			df = self.tt[neuron].copy()
			df = pd.concat([df, self.setOnOff], axis=1)
			# df.rename(columns={neuron: 'output'}, inplace=True)

			if discriminated:
				print(f'Applying discriminator')
				discrimData = self.discrimData[:, i]
				df['drop'] = discrimData
				df = df[df['drop'] == 1]

			# Take out entries out of the ON-Set
			# Select only ON-Set
			# df = df[df[neuron] == 1]

			df = df.astype('int')

			self.generateABCInput(df, f'{baseFilename}{neuron}', neuron)
			del df  # caring about memory
			print(f"Realize ABC neurons [{i + 1:>5d}/{len(self.outputTags):>5d}]")
			i += 1

	def _toBinaryArrayActivations(self, row):
		"""
		Private method
		:param row:
		:return:
		"""
		return integerToBinaryArray(row[self.activationTags], row[self.lengthActivationTags])

	def _toBinaryArrayDiscriminator(self, row):
		"""
		Private method
		:param row:
		:return:
		"""
		return integerToBinaryArray(row[self.discriminationTags], row[self.lengthDiscriminationTags])
