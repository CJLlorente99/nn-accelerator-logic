import pandas as pd
from binaryNeuronSimplified import BinaryOutputNeuron
from accLayerSimplified import AccLayer

neuronsPerLayer = 300

layer0ImportanceFilename = f'../data/layersImportance/2023_04_21_layer0ImportanceGradientBinary20epochs{neuronsPerLayer}nnpl'
layer1ImportanceFilename = f'../data/layersImportance/2023_04_21_layer1ImportanceGradientBinary20epochs{neuronsPerLayer}nnpl'

layer0TTFilename = f'../data/layersTT/2023_04_15_layer0_binaryNN20Epoch{neuronsPerLayer}NPLBlackAndWhite'
layer1TTFilename = f'../data/layersTT/2023_04_15_layer1_binaryNN20Epoch{neuronsPerLayer}NPLBlackAndWhite'

# Load importance and TT
layer0ImportanceDf = pd.read_feather(layer0ImportanceFilename).set_index(['name'])
layer1ImportanceDf = pd.read_feather(layer1ImportanceFilename).set_index(['name'])

classImportanceTags = []
for col in layer0ImportanceDf.columns:
	if col.startswith('class'):
		classImportanceTags.append(col)

layer0TTDf = pd.read_feather(layer0TTFilename)
layer1TTDf = pd.read_feather(layer1TTFilename)

tags0 = []
for col in layer0TTDf.columns:
	if col.startswith('activation'):
		tags0.append(col)
	elif col.startswith('lengthActivation'):
		tags0.append(col)
	elif col.startswith('class'):
		tags0.append(col)
	elif col.startswith('output'):
		tags0.append(col)

tags1 = []
for col in layer1TTDf.columns:
	if col.startswith('activation'):
		tags1.append(col)
	elif col.startswith('lengthActivation'):
		tags1.append(col)
	elif col.startswith('class'):
		tags1.append(col)
	elif col.startswith('output'):
		tags1.append(col)

# Recreate the acc layers
accLayer0 = AccLayer('layer0', len(layer0ImportanceDf), layer0TTDf[tags0])
accLayer1 = AccLayer('layer1', len(layer1ImportanceDf), layer1TTDf[tags1])

# Recreate the neurons
# Layer 0
neurons = {}
for index, row in layer0ImportanceDf.iterrows():
	neurons[index] = BinaryOutputNeuron(index, accLayer0, row[classImportanceTags].to_dict(), row['importance'])

accLayer0.neurons = neurons

# Layer 1
neurons = {}
for index, row in layer1ImportanceDf.iterrows():
	neurons[index] = BinaryOutputNeuron(index, accLayer1, row[classImportanceTags].to_dict(), row['importance'])

accLayer1.neurons = neurons

# Add new columns to the acc layer indicating which TT entries are to be keep per neuron
accLayer0.reduceTTThreshold()
accLayer1.reduceTTThreshold()

# Save optimized TT
accLayer0.saveTT(f'../data/layersTTOptimized/layer0_binary20epoch{neuronsPerLayer}NPLBlackAndWhite')
accLayer1.saveTT(f'../data/layersTTOptimized/layer1_binary20epoch{neuronsPerLayer}NPLBlackAndWhite')

# Print same info on gains
accLayer0.plotGainsPerNeuron()
accLayer1.plotGainsPerNeuron()

