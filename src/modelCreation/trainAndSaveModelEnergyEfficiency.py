import torch
from modelsCommon.auxFunctionsEnergyEfficiency import trainAndTest
from modelsCommon.auxTransformations import *
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader
from modules.binaryEnergyEfficiency import BinaryNeuralNetwork
from modules.binaryEnergyEfficiencyNoBN import BinaryNeuralNetworkNoBN
from modules.fpNN import FPNeuralNetwork
import torch.optim as optim
import torch.nn as nn

batch_size = 64
neuronPerLayer = 100
epochs = 10
criterionName = 'nll'
# criterionName = 'cel'
# precision = 'bin'
precision = 'binNoBN'
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
elif precision == 'binNoBN':
    model = BinaryNeuralNetworkNoBN(neuronPerLayer).to(device)

'''
Train and test
'''
print(f'TRAINING\n')

opt = optim.Adamax(model.parameters(), lr=3e-3, weight_decay=1e-6)

if criterionName == 'nll':
    criterion = nn.NLLLoss()
elif criterionName == 'cel':
    criterion = nn.CrossEntropyLoss()

trainAndTest(epochs, train_dataloader, test_dataloader, model, opt, criterion)

'''
Save
'''

torch.save(model.state_dict(), f'./src/modelCreation/savedModels/MNISTSign{precision}NN{epochs}Epoch{neuronPerLayer}NPL{criterionName}Criterion')
