'''
This file contains the methods needed to realize the NN into a HW compatible realization
'''
import pandas as pd
import itertools
import numpy as np
import os


class DNFRealization:
	def __init__(self, ttFolderName):
		self.ttFolderName = ttFolderName

	def generateEspressoInput(self, df: pd.DataFrame, filename: str):
		"""
		Method that parses an ON-set and OFF-set defined TT into a PLA file that can be processed by Espresso SW
		:param df:
		:param filename:
		"""
		inTags = [col for col in df.columns if col.startswith('IN')]
		outTag = [col for col in df.columns if col.startswith('OUT')]

		with open(f'{filename}.pla', 'w') as f:
			# Write header of PLA file
			f.write(f'.i {len(inTags)}\n')  # Number of input neurons
			f.write(f'.o {1}\n')  # Number of output per neuron (just 1)
			tags = ' '.join(inTags)
			f.write(f'.ilb {tags}\n')  # Names of the input variables
			f.write(f'.ob {outTag[0]}\n')  # Name of the output variable
			f.write(f'.type fr\n')  # .pla contains ON-Set and OFF-Set
			for index, row in df.iterrows():
				text = ''.join(row[inTags].to_string(header=False, index=False).split('\n'))
				outputText = row[outTag[0]]
				f.write(f'{text} {outputText}\n')
			f.write(f'.e')
		print(f'ESPRESSO PLA {filename} completed')

	def generateABCInput(self, df: pd.DataFrame, filename: str):
		"""
		Method that parses an ON-set and OFF-set defined TT into a file that can be processed by ABC SW
		:param df:
		:param filename;
		"""
		# ABC only assumes fd type in pla file. This means, the data in pla represents the ON-Set (1) and the
		# DC-Set (-). As DC-Set is not supported, PLA file should only contain (1)
		inTags = [col for col in df.columns if col.startswith('IN')]
		outTag = [col for col in df.columns if col.startswith('OUT')]

		with open(f'{filename}.pla', 'w') as f:
			# Write header of PLA file
			f.write(f'.i {len(inTags)}\n')  # Number of input neurons
			f.write(f'.o {1}\n')  # Number of output per neuron (just 1)
			# tags = [f'N{i}' for i in range(self.nNeurons)]
			# tags = ' '.join(tags)
			# f.write(f'.ilb {tags}\n')  # Names of the input variables
			# f.write(f'.ob {neuron}\n')  # Name of the output variable
			f.write(f'.p {len(df)}\n')
			for index, row in df.iterrows():
				text = ''.join(row[inTags].to_string(header=False, index=False).split('\n'))
				outputText = row[outTag[0]]
				f.write(f'{text} {outputText}\n')
			f.write(f'.e')
		print(f'ABC PLA {filename} completed')
   
   
	def createPLAFileEspresso(self, baseFilename: str):
		# Care about the output folder
		dirs = os.scandir(self.ttFolderName)

		for dir in dirs:
			if not os.path.isdir(f'{baseFilename}/{dir.name}'):
				os.makedirs(f'{baseFilename}/{dir.name}')

		# Loop over the dirs and their files
		dirs = os.scandir(self.ttFolderName)
		for dir in dirs:
			files = os.scandir(f'{self.ttFolderName}/{dir.name}')
			for file in files:
				df = pd.read_csv(f'{self.ttFolderName}/{dir.name}/{file.name}', index_col=False)
				df = df.astype('int')
				df.drop_duplicates(inplace=True)
				self.generateEspressoInput(df, f'{baseFilename}/{dir.name}/{file.name}')

	def createPLAFileABC(self, baseFilename: str):
		# Care about the output folder
		dirs = os.scandir(self.ttFolderName)

		for dir in dirs:
			if not os.path.isdir(f'{baseFilename}/{dir.name}'):
				os.makedirs(f'{baseFilename}/{dir.name}')

		# Loop over the dirs and their files
		dirs = os.scandir(self.ttFolderName)
		for dir in dirs:
			files = os.scandir(f'{self.ttFolderName}/{dir.name}')
			for file in files:
				df = pd.read_csv(f'{self.ttFolderName}/{dir.name}/{file.name}', index_col=False)
				df = df.astype('int')
				df.drop_duplicates(inplace=True)
				self.generateABCInput(df, f'{baseFilename}/{dir.name}/{file.name}')
