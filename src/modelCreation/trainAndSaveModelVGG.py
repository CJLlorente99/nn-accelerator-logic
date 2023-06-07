import torch
from modelsCommon.auxFunc import trainAndTest
from modelsCommon.auxTransformations import *
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize, RandomHorizontalFlip, RandomCrop
from torch.utils.data import DataLoader
from modules.vggSmall import VGGSmall
import torch.optim as optim
import torch.nn as nn

batch_size = 128
epochs = 25

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
    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
                                 download=False)

test_dataset = datasets.CIFAR10(root='./data', train=False, transform=Compose([
    ToTensor(),
    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
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

model = VGGSmall().to(device)

'''
Train and test
'''
print(f'TRAINING\n')

opt = optim.SGD(model.parameters(), lr=0.05, weight_decay=0.0005, momentum=0.9)
criterion = nn.CrossEntropyLoss()

trainAndTest(epochs, train_dataloader, test_dataloader, model, opt, criterion)

'''
Save
'''

torch.save(model.state_dict(), f'./src/modelCreation/savedModels/VGGSmall')
