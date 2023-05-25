import pandas as pd
import torch
from modelsCommon.auxTransformations import *
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader
from modules.binaryEnergyEfficiencyAllGradients import BinaryNeuralNetwork
from ttUtilities.helpLayerNeuronGenerator import HelpGenerator
import copy

neuronPerLayer = 100
modelFilename = f'src\modelCreation\savedModels\MNISTSignbinNN100Epoch100NPLnllCriterion'
batch_size = 64
perGradientSampling = 1
# Check mps maybe if working in MacOS
device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
Importing MNIST dataset
'''
print(f'IMPORT DATASET\n')

training_data = datasets.MNIST(
    root='C:/Users/carlo/OneDrive/Documentos/Universidad/MUIT/Segundo/TFM/Code/data',
    train=True,
    download=False,
    transform=Compose([
            ToTensor(),
            ToBlackAndWhite(),
            ToSign()
        ])
)

sampleSize = int(perGradientSampling * len(training_data.data))  # sample size to be used for importance calculation

'''
Create DataLoader
'''
train_dataloader = DataLoader(training_data, batch_size=batch_size)

model = BinaryNeuralNetwork(neuronPerLayer)
model.load_state_dict(torch.load(modelFilename))

'''
Load gradients previously calculated
'''

model.gradientsSTE0 = pd.read_feather('data/gradients/gradientsSignBin100epochs100nplSTE0').drop(['target'], axis=1).to_numpy()
model.gradientsSTE1 = pd.read_feather('data/gradients/gradientsSignBin100epochs100nplSTE1').drop(['target'], axis=1).to_numpy()
model.gradientsSTE2 = pd.read_feather('data/gradients/gradientsSignBin100epochs100nplSTE2').drop(['target'], axis=1).to_numpy()
model.gradientsSTE3 = pd.read_feather('data/gradients/gradientsSignBin100epochs100nplSTE3').drop(['target'], axis=1).to_numpy()

model.gradientsL0 = pd.read_feather('data/gradients/gradientsSignBin100epochs100nplL0').drop(['target'], axis=1).to_numpy()
model.gradientsL1 = pd.read_feather('data/gradients/gradientsSignBin100epochs100nplL1').drop(['target'], axis=1).to_numpy()
model.gradientsL2 = pd.read_feather('data/gradients/gradientsSignBin100epochs100nplL2').drop(['target'], axis=1).to_numpy()
model.gradientsL3 = pd.read_feather('data/gradients/gradientsSignBin100epochs100nplL3').drop(['target'], axis=1).to_numpy()

model.valueL0 = pd.read_feather('data/activations/activationsSignBin100epochs100nplValueL0').to_numpy()
model.valueL1 = pd.read_feather('data/activations/activationsSignBin100epochs100nplValueL1').to_numpy()
model.valueL2 = pd.read_feather('data/activations/activationsSignBin100epochs100nplValueL2').to_numpy()
model.valueL3 = pd.read_feather('data/activations/activationsSignBin100epochs100nplValueL3').to_numpy()

'''
Generate AccLayers and Neuron objects
'''

accLayers = HelpGenerator.getAccLayers(model)
HelpGenerator.getNeurons(accLayers)

# Create copies
accLayersBN = copy.deepcopy(accLayers)

print(f'Number of layers is {len(accLayers)}')
i = 0
for layer in accLayers:
    print(f'Layer {i} has {layer.nNeurons} neurons')
    i += 1

'''
Calculate importance per class per neuron
'''

# Compute importance

importance = model.computeImportance(neuronPerLayer)

# Give each neuron its importance values
for j in range(len(importance)//2):
    for i in range(len(accLayers[j].neurons)):
        accLayers[j].neurons[i].giveImportance(importance[j][:, i], training_data.targets.tolist(), 10e-2)

        if (i + 1) % 50 == 0:
            print(f"Give Importance Layer [{j + 1:>1d}/{len(importance):>1d}] Neuron [{i + 1:>4d}/{neuronPerLayer:>4d}]")
            
for j in range(len(importance)//2):
    for i in range(len(accLayersBN[j].neurons)):
        accLayersBN[j].neurons[i].giveImportance(importance[j + len(importance)//2][:, i], training_data.targets.tolist(), 10e-2)

        if (i + 1) % 50 == 0:
            print(f"Give Importance Layer [{j + len(importance)//2 + 1:>1d}/{len(importance):>1d}] Neuron [{i + 1:>4d}/{neuronPerLayer:>4d}]")

# Plot importance of neurons per layer

for i in range(len(accLayers)):
    accLayers[i].plotImportancePerNeuron(f'Layer {i}', False)
    accLayersBN[i].plotImportancePerNeuron(f'Layer {i}', False)
    # accLayers[i].plotImportancePerClass(f'Layer {i}', True)
    accLayers[i].plotNumImportantClasses(f'Layer {i}', False)
    accLayersBN[i].plotNumImportantClasses(f'Layer {i}', False)
    # accLayers[i].saveImportance(f'data/layersImportance/layer{i}Importance1e3GradientBinarySignBNN50epochs{neuronPerLayer}npl')
    print(f'Creating plots and saving layer [{i + 1:>1d}/{len(accLayers):>1d}]')
