import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from modelsCommon.auxTransformations import *

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
            ToBlackAndWhite()
        ])
)

test_data = datasets.MNIST(
    root='C:/Users/carlo/OneDrive/Documentos/Universidad/MUIT/Segundo/TFM/Code/data',
    train=False,
    download=False,
    transform=Compose([
        ToTensor(),
        ToBlackAndWhite()
    ])
)

'''
Create DataLoader
'''
train_dataloader = DataLoader(training_data, batch_size=1)
test_dataloader = DataLoader(test_data, batch_size=1)

'''
Create input files
'''
trainInputFile = 'data/inputs/trainInput'
testInputFile = 'data/inputs/testInput'

with open(trainInputFile, 'w') as f:
    for i in range(len(training_data.data)):
        X, y = train_dataloader.dataset[i]
        f.write(''.join(map(str, X.flatten().type(torch.int).tolist())))
        f.write('\n')

        if (i+1) % 500 == 0:
            print(f"Write Inputs [{i+1:>5d}/{len(training_data.data):>5d}]")
    f.close()
            
with open(testInputFile, 'w') as f:
    for i in range(len(test_data.data)):
        X, y = test_dataloader.dataset[i]
        f.write(''.join(map(str, X.flatten().type(torch.int).tolist())))
        f.write('\n')

        if (i+1) % 500 == 0:
            print(f"Write Inputs [{i+1:>5d}/{len(test_data.data):>5d}]")
    f.close()
