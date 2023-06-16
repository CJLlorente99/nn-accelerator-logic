import torch
from modelsCommon.auxFunctionsEnergyEfficiency import trainAndTest
from modelsCommon.auxTransformations import *
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader
from modules.binaryEnergyEfficiency import BinaryNeuralNetwork
from modules.binaryEnergyEfficiencyNoBN import BinaryNeuralNetworkNoBN
import torch.optim as optim
import torch.nn as nn

batch_size = 64
neuronPerLayer = 100
epochs = 50

# Check mps maybe if working in MacOS
device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
Importing MNIST dataset
'''
print(f'IMPORT DATASET\n')

training_data = datasets.MNIST(
    root='/home/carlosl/Dokumente/nn-accelerator-logic/data',
    train=True,
    download=False,
    transform=Compose([
            ToTensor(),
            ToBlackAndWhite(),
            ToSign()
        ])
)

test_data = datasets.MNIST(
    root='/home/carlosl/Dokumente/nn-accelerator-logic/data',
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

model = BinaryNeuralNetwork(neuronPerLayer).to(device)

'''
Train and test
'''
print(f'TRAINING\n')

opt = optim.Adamax(model.parameters(), lr=3e-3, weight_decay=1e-6)

criterion = nn.NLLLoss()

trainAndTest(epochs, train_dataloader, test_dataloader, model, opt, criterion)

'''
Save
'''

torch.save(model.state_dict(), f'/home/carlosl/Dokumente/nn-accelerator-logic/src/modelCreation/savedModels/MNIST_bin_100NPL')
