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
				text = text.replace('2', '-')
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
				text = text.replace('2', '-')
				outputText = row[outTag[0]]
				f.write(f'{text} {outputText}\n')
			f.write(f'.e')
		print(f'ABC PLA {filename} completed')
   
   
	def createPLAFileEspresso(self, baseFilename: str, pruned=False, prunedBaseFilename=None, conflictMode=-1, bin=False):
		# Care about the output folder
		dirs = os.scandir(self.ttFolderName)

		for dir in dirs:
			if not os.path.isdir(f'{baseFilename}/{dir.name}'):
					os.makedirs(f'{baseFilename}/{dir.name}')
			if conflictMode != -1:
					if not os.path.isdir(f'{baseFilename}/{dir.name}_{conflictMode}'):
						os.makedirs(f'{baseFilename}/{dir.name}_{conflictMode}')

		# Loop over the dirs and their files
		dirs = os.scandir(self.ttFolderName)
		i = 1
		for dir in dirs:
			if pruned:
				dfPruned = pd.read_csv(f'{prunedBaseFilename}{i}.csv')
			files = os.scandir(f'{self.ttFolderName}/{dir.name}')
			for file in files:
				df = pd.read_feather(f'{self.ttFolderName}/{dir.name}/{file.name}')
				if bin:
					dfInt = df.apply(binaryStrToInt, axis=1)
					columns = [f'N{i:04d}' for i in range(len(dfInt[0]))]
					dfInt = pd.DataFrame(np.stack(dfInt.to_numpy(), axis=0), columns=columns)
					df = pd.concat([dfInt, df[df.columns[-1]]], axis=1)
				df = df.astype('int')
				df.drop_duplicates(inplace=True)
				if pruned:
					df.drop(df.columns[dfPruned[file.name].values], axis=1, inplace=True)
					df.drop_duplicates(inplace=True)
				# Deal with conflict of orthogonality between On and Off set
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
				self.generateEspressoInput(df, f'{baseFilename}/{dir.name}/{file.name}')
			i += 1

	def createPLAFileABC(self, baseFilename: str, pruned=False, prunedBaseFilename=None, bin=False):
		# Care about the output folder
		dirs = os.scandir(self.ttFolderName)

		for dir in dirs:
			if not os.path.isdir(f'{baseFilename}/{dir.name}'):
				os.makedirs(f'{baseFilename}/{dir.name}')

		# Loop over the dirs and their files
		dirs = os.scandir(self.ttFolderName)
		i = 1
		for dir in dirs:
			if pruned:
				dfPruned = pd.read_csv(f'{prunedBaseFilename}{i}.csv')
			files = os.scandir(f'{self.ttFolderName}/{dir.name}')
			for file in files:
				df = pd.read_feather(f'{self.ttFolderName}/{dir.name}/{file.name}')
				if bin:
					dfInt = df.apply(binaryStrToInt, axis=1)
					columns = [f'N{i:04d}' for i in range(len(dfInt[0]))]
					dfInt = pd.DataFrame(np.stack(dfInt.to_numpy(), axis=0), columns=columns)
					df = pd.concat([dfInt, df[df.columns[-1]]], axis=1)
				df = df.astype('int')
				df.drop_duplicates(inplace=True)
				if pruned:
					df.drop(df.columns[dfPruned[file.name].values], axis=1, inplace=True)
					df.drop_duplicates(inplace=True)
				self.generateABCInput(df, f'{baseFilename}/{dir.name}/{file.name}')
			i += 1

# Function to translate binary into integer array
def binaryStrToInt(row):
    npl = 4096
    row = bin(int(row['int'], 0))[2:]
    row = row.zfill(npl)
    return np.array(list(map(int, row)))
