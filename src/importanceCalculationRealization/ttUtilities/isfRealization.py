
'''
This file contains the methods needed to realize the NN into a HW compatible realization
'''
import pandas as pd
import numpy as np
import os

def generateEspressoInput(df: pd.DataFrame, filename: str):
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
			text = text.replace('2', '-')
			outputText = row[outTag[0]]
			f.write(f'{text} {outputText}\n')
		f.write(f'.e')
	print(f'ESPRESSO PLA {filename} completed')

def generateABCInput(df: pd.DataFrame, filename: str):
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
			text = text.replace('2', '-')
			outputText = row[outTag[0]]
			f.write(f'{text} {outputText}\n')
		f.write(f'.e')
	print(f'ABC PLA {filename} completed')


def createPLAFileEspresso(df: pd.DataFrame, outputFilename: str, conflictMode=-1):
			
	# Deal with conflict of orthogonality between On and Off set
	if conflictMode != -1:
		# Get rows with DC (2)
		rowsWithDC = df[(df == 2).any(axis=1)]
		# Get conflictive ones
		for idx, row in rowsWithDC.iterrows():
			auxDf = df.copy().drop(df.columns[-1], axis=1)
			auxDf = auxDf.drop(row[row == 2].index, axis=1)
			dup = (auxDf == row.drop(row.index[row == 2]).drop(row.index[-1])).all(axis=1)
			# 1) Keep without DC
			if dup.sum() != 0: # There is a coincidence
				if conflictMode == 0:
					aux = dup.copy()
					aux[(df != row).any(axis=1)] = False # All but the row will be false
					df.drop(df.index[aux], axis=0, inplace=True)
				# 2) Keep only DC
				elif conflictMode == 1:
					aux = dup.copy()
					aux[(df == row).all(axis=1)] = False # All but the row will be true
					df.drop(df.index[aux], axis=0, inplace=True)
					# ! When 2 or more DC are present
				# 3) Remove all
				elif conflictMode == 2:
					df.drop(df.index[dup], axis=0, inplace=True)
					# ! Sometimes incurs in empty pla (everything has a DC)
	generateEspressoInput(df, outputFilename)

def createPLAFileABC(df: pd.DataFrame, outputFilename: str):
	
	generateABCInput(df, outputFilename)

def pruneAndDrop(df: pd.DataFrame, dfPruned: pd.DataFrame, neuronName: str):
	df = df.astype('int')
	aux = df.drop(df.columns[dfPruned[neuronName].values], axis=1)
	aux.drop_duplicates(inplace=True)
	return aux