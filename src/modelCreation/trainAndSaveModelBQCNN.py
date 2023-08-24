import torch
from modelsCommon.auxFunc import trainAndTest, testReturn
from modelsCommon.auxTransformations import *
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize, RandomHorizontalFlip, RandomCrop, Resize
from torch.utils.data import DataLoader
from modules.binaryBQCNN import BQCNN
import torch.optim as optim
import torch.nn as nn
import sys
import pandas as pd
import numpy as np

batch_size = 128
epochs = 100

def main(modelName, resizeFactor, inputsAfterBTPruning):

    # Check mps maybe if working in MacOS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    model = BQCNN().to(device)
    
    '''
    Importing CIFAR10 dataset
    '''
    print(f'DOWNLOAD DATASET\n')
    train_dataset = datasets.CIFAR10(root='/srv/data/image_dataset/CIFAR10/', train=True, transform=Compose([
            RandomHorizontalFlip(),
            RandomCrop(32, 4),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            Resize(resizeFactor*32, antialias=False)]),
        download=False)
    
    test_dataset = datasets.CIFAR10(root='/srv/data/image_dataset/CIFAR10/', train=False, transform=Compose([
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
    Train and test
    '''
    print(f'TRAINING\n')
    
    opt = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0005, momentum=0.7)
    criterion = nn.CrossEntropyLoss()
    
    trainAndTest(epochs, train_dataloader, test_dataloader, model, opt, criterion)
    
    '''
    Save
    '''
    # First layer
    weightMask = list(model.l0.named_buffers())[0][1]
    idxs = []
    for iNeuron in range(len(weightMask)):
        weights = weightMask[iNeuron].detach().cpu().numpy()
        idxs.append(np.where(weights == 0)[0])
    columnTags = [f'N{i:04d}' for i in range(len(weightMask))]
    df = pd.DataFrame(np.array(idxs).T, columns=columnTags)
    df.to_csv(f'savedModels/{modelName}_prunedBT{inputsAfterBTPruning}_{resizeFactor}_prunedInfo{0}.csv', index=False)

    # Second layer
    weightMask = list(model.l1.named_buffers())[0][1]
    idxs = []
    for iNeuron in range(len(weightMask)):
        weights = weightMask[iNeuron].detach().cpu().numpy()
        idxs.append(np.where(weights == 0)[0])
    columnTags = [f'N{i:04d}' for i in range(len(weightMask))]
    df = pd.DataFrame(np.array(idxs).T, columns=columnTags)
    df.to_csv(f'savedModels/{modelName}_prunedBT{inputsAfterBTPruning}_{resizeFactor}_prunedInfo{1}.csv', index=False)

    # Third layer
    weightMask = list(model.l2.named_buffers())[0][1]
    idxs = []
    for iNeuron in range(len(weightMask)):
        weights = weightMask[iNeuron].detach().cpu().numpy()
        idxs.append(np.where(weights == 0)[0])
    columnTags = [f'N{i:04d}' for i in range(len(weightMask))]
    df = pd.DataFrame(np.array(idxs).T, columns=columnTags)
    df.to_csv(f'savedModels/{modelName}_prunedBT{inputsAfterBTPruning}_{resizeFactor}_prunedInfo{2}.csv', index=False)

    metrics = '\n'.join(testReturn(test_dataloader, train_dataloader, model, criterion))
    print(metrics)
    with open(f'savedModels/{modelName}_prunedBT{inputsAfterBTPruning}_{resizeFactor}.txt', 'w') as f:
        f.write(metrics)
        f.write('\n')
        f.write(f'epochs {epochs}\n')
        f.write(f'batch {batch_size}\n')
        f.close()
    
    torch.save(model.state_dict(), f'savedModels/{modelName}_prunedBT{inputsAfterBTPruning}_{resizeFactor}')
    
    
if __name__ == '__main__':
    main('binaryFullBVGGVerySmall', 4, 6)

    
