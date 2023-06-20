import sys

import torch
from modelsCommon.auxFunc import trainAndTest, testReturn
from modelsCommon.auxTransformations import *
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize, RandomHorizontalFlip, RandomCrop, Resize
from torch.utils.data import DataLoader
from modules.binaryVggVerySmall import binaryVGGVerySmall
from modules.binaryVggVerySmall2 import binaryVGGVerySmall2
from modules.vggVerySmall import VGGVerySmall
from modules.vggSmall import VGGSmall
from modules.binaryVggSmall import BinaryVGGSmall
import torch.optim as optim
import torch.nn as nn
import sys

# ! Change the name of the saved module
batch_size = 200
epochs = 20

resizeFactor = 2
relus = [1, 1, 1, 1, 0, 0, 0, 0]  # 0 = Sign, 1 = relu
relusStr = ''.join(map(str, relus))
modelName = 'binaryVGGVerySmall2'

# Check mps maybe if working in MacOS
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if modelName == 'binaryVGGVerySmall':
    model = binaryVGGVerySmall(resizeFactor, relus).to(device)
elif modelName == 'binaryVGGVerySmall2':
    model = binaryVGGVerySmall2(resizeFactor, relus).to(device)

'''
Importing CIFAR10 dataset
'''
print(f'DOWNLOAD DATASET\n')
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=Compose([
        RandomHorizontalFlip(),
        RandomCrop(32, 4),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        Resize(resizeFactor*32, antialias=False)]),
    download=False)

test_dataset = datasets.CIFAR10(root='./data', train=False, transform=Compose([
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        Resize(resizeFactor*32, antialias=False)]),
    download=False)

'''
Create DataLoader
'''
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

'''
Instantiate NN models
'''
print(f'MODEL INSTANTIATION\n')


'''
Train and test
'''
print(f'TRAINING\n')

opt = optim.SGD(model.parameters(), lr=0.02, weight_decay=0.0005, momentum=0.9)
criterion = nn.CrossEntropyLoss()

trainAndTest(epochs, train_dataloader, test_dataloader, model, opt, criterion)

'''
Save
'''

metrics = '\n'.join(testReturn(test_dataloader, train_dataloader, model, criterion))
print(metrics)
with open(f'./src/modelCreation/savedModels/{modelName}_{relusStr}_{resizeFactor}.txt', 'w') as f:
    f.write(metrics)
    f.close()

torch.save(model.state_dict(), f'./src/modelCreation/savedModels/{modelName}_{relusStr}_{resizeFactor}')