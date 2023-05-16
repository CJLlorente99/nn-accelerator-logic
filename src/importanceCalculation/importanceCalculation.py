import pandas as pd
import torch
from modelsCommon.auxTransformations import *
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader
from modules.binaryEnergyEfficiency import BinaryNeuralNetwork
from ttUtilities.helpLayerNeuronGenerator import HelpGenerator

neuronPerLayer = 1024
modelFilename = f'src\modelCreation\savedModels\MNISTSignbinNN100Epoch1024NPLnllCriterion'
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

model.gradientsSTE0 = pd.read_feather('data/gradients/gradientsSignBin100epochs1024nplSTE0').drop(['target'], axis=1).to_numpy()
model.gradientsSTE1 = pd.read_feather('data/gradients/gradientsSignBin100epochs1024nplSTE1').drop(['target'], axis=1).to_numpy()
model.gradientsSTE2 = pd.read_feather('data/gradients/gradientsSignBin100epochs1024nplSTE2').drop(['target'], axis=1).to_numpy()
model.gradientsSTE3 = pd.read_feather('data/gradients/gradientsSignBin100epochs1024nplSTE3').drop(['target'], axis=1).to_numpy()

'''
Generate AccLayers and Neuron objects
'''

accLayers = HelpGenerator.getAccLayers(model)
HelpGenerator.getNeurons(accLayers)

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
for j in range(len(importance)):
    for i in range(len(accLayers[j].neurons)):
        accLayers[j].neurons[i].giveImportance(importance[j][:, i], training_data.targets.tolist(), 10e-2)

        if (i + 1) % 50 == 0:
            print(f"Give Importance Layer [{j + 1:>1d}/{len(importance):>1d}] Neuron [{i + 1:>4d}/{neuronPerLayer:>4d}]")

# Plot importance of neurons per layer

for i in range(len(accLayers)):
    accLayers[i].plotImportancePerNeuron(f'Layer {i}', True)
    # accLayers[i].plotImportancePerClass(f'Layer {i}', True)
    accLayers[i].plotNumImportantClasses(f'Layer {i}', True)
    # accLayers[i].saveImportance(f'data/layersImportance/layer{i}Importance1e3GradientBinarySignBNN50epochs{neuronPerLayer}npl')
    print(f'Creating plots and saving layer [{i + 1:>1d}/{len(accLayers):>1d}]')
