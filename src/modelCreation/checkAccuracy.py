import torch
from modelsCommon.auxFunc import test
from modelsCommon.auxTransformations import *
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize, RandomHorizontalFlip, RandomCrop, Resize
from torch.utils.data import DataLoader
from modules.binaryVggVerySmall import binaryVGGVerySmall
from modules.binaryVggVerySmall2 import binaryVGGVerySmall2
from modules.vggVerySmall import VGGVerySmall
import torch.optim as optim
import torch.nn as nn

batch_size = 128
modelFilename = f'src\modelCreation\savedModels/binaryVGGVerySmall'
relus = [1, 0, 0, 0, 0, 0, 1]
resizeFactor = 2
model = binaryVGGVerySmall(resizeFactor, relus)

# Check mps maybe if working in MacOS
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

model.load_state_dict(torch.load(modelFilename))

'''
Train and test
'''
print(f'TESTING\n')

criterion = nn.CrossEntropyLoss()

test(test_dataloader, train_dataloader, model, criterion)
