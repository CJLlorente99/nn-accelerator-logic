import torch
from modelsCommon.auxFunctionsEnergyEfficiency import trainAndTest, test
from modelsCommon.auxTransformations import *
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader
from modules.binaryEnergyEfficiency import BinaryNeuralNetwork
from modules.fpNN import FPNeuralNetwork
import torch.nn as nn

batch_size = 100
neuronPerLayer = 100
mod = True  # Change accordingly in modelFilename too
modelFilename = f'src\modelCreation\savedModels\eeb_100ep_100npl'
criterionName = 'nll'
# criterionName = 'cel'
precision = 'bin'
# precision = 'full'


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

test_data = datasets.MNIST(
    root='C:/Users/carlo/OneDrive/Documentos/Universidad/MUIT/Segundo/TFM/Code/data',
    train=False,
    download=False,
    transform=Compose([
        ToTensor(),
        ToBlackAndWhite(),
        ToSign()
    ])
)

'''
Create DataLoader
'''
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

'''
Instantiate NN models
'''
print(f'MODEL INSTANTIATION\n')

if precision == 'full':
    model = FPNeuralNetwork(neuronPerLayer).to(device)
elif precision == 'bin':
    model = BinaryNeuralNetwork(neuronPerLayer).to(device)
model.load_state_dict(torch.load(modelFilename))

'''
Test
'''
print(f'TEST\n')

if criterionName == 'nll':
    criterion = nn.NLLLoss()
elif criterionName == 'cel':
    criterion = nn.CrossEntropyLoss()

test(test_dataloader, train_dataloader, model, criterion)

