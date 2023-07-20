import torch
from modelsCommon.auxFunctionsEnergyEfficiency import trainAndTest, testReturn
from modelsCommon.auxTransformations import *
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader
from modules.binaryEnergyEfficiency import BinaryNeuralNetwork
from modules.binaryEnergyEfficiencyNoBN import BinaryNeuralNetworkNoBN
import torch.optim as optim
import torch.nn as nn
import pandas as pd

batch_size = 64
neuronPerLayer = 100
epochs = 100
pruned = True

# Check mps maybe if working in MacOS
device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
Importing MNIST dataset
'''
print(f'IMPORT DATASET\n')

training_data = datasets.MNIST(
    root='/srv/data/image_dataset/MNIST',
    train=True,
    download=False,
    transform=Compose([
            ToTensor(),
            ToBlackAndWhite(),
            ToSign()
        ])
)

test_data = datasets.MNIST(
    root='/srv/data/image_dataset/MNIST',
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
print(f'SAVING\n')

if pruned:
    prunedConnections = model.pruningSparsification(neuronPerLayer - 30)
    for layer in prunedConnections:
        columnTags = [f'N{i}' for i in range(prunedConnections[layer].shape[0])]
        df = pd.DataFrame(prunedConnections[layer].T, columns=columnTags)
        df.to_csv(f'savedModels/eeb_pruned_{epochs}ep_{neuronPerLayer}npl_prunnedInfo{layer}.csv', index=False)

metrics = '\n'.join(testReturn(test_dataloader, train_dataloader, model, criterion))
print(metrics)
with open(f'savedModels/eeb_pruned_{epochs}ep_{neuronPerLayer}npl.txt', 'w') as f:
    f.write(metrics)
    f.write('\n')
    f.write(f'epochs {epochs}\n')
    f.write(f'batch {batch_size}\n')
    f.close()

torch.save(model.state_dict(), f'savedModels/eeb_pruned_{epochs}ep_{neuronPerLayer}npl')
