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
   
	def generateAIG(self, df: pd.DataFrame, filename: str, neuron: str):
		"""
		Method that parses an ON-set  defined TT into a AIG file
		:param df:
		:param filename;
		"""
		pass
   
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
   
	def createAIGFile(self, baseFilename: str, discriminated: bool = False):
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
			df = df[df[neuron] == 1]

			df = df.astype('int')

			self.generateAIG(df, f'{baseFilename}{neuron}', neuron)
			del df  # caring about memory
			print(f"Realize AIG neurons [{i + 1:>5d}/{len(self.outputTags):>5d}]")
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
