import torch
from modelsCommon.auxFunc import trainAndTest, testReturn
from modelsCommon.auxTransformations import *
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize, RandomHorizontalFlip, RandomCrop, Resize
from torch.utils.data import DataLoader
from modules.binaryVggVerySmall2 import binaryVGGVerySmall2
from modules.binaryVggSmall import binaryVGGSmall
from modules.vggSmall import VGGSmall
import torch.optim as optim
import torch.nn as nn
import sys
import pandas as pd
import numpy as np

batch_size = 128
epochs = 200

def main(modelName, relus, resizeFactor, irregularPrune, inputsRegularToPrune, inputsAfterIrregularPrune):
    # Regular False, Irregular True
    # If regular prune
    if irregularPrune:
        inputsRegularToPrune = -1
    # If irregular prune
    if not irregularPrune:
        inputsAfterIrregularPrune = -1

    relusStr = ''.join(map(str, relus))

    # Check mps maybe if working in MacOS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    if modelName == 'binaryVGGVerySmall2':
        model = binaryVGGVerySmall2(resizeFactor, relus, inputsRegularToPrune).to(device)
    elif modelName == 'binaryVGGSmall':
        model = binaryVGGSmall(resizeFactor, relus, inputsRegularToPrune).to(device)
    
    print(model.named_modules)
    
    '''
    Importing CIFAR10 dataset
    '''
    print(f'DOWNLOAD DATASET\n')
    train_dataset = datasets.CIFAR10(root='/srv/data/image_dataset/CIFAR10', train=True, transform=Compose([
            RandomHorizontalFlip(),
            RandomCrop(32, 4),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            Resize(resizeFactor*32, antialias=False)]),
        download=False)
    
    test_dataset = datasets.CIFAR10(root='/srv/data/image_dataset/CIFAR10', train=False, transform=Compose([
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

    print(f'SAVING\n')

    if irregularPrune:
        prunedConnections = model.pruningSparsification(inputsAfterIrregularPrune)
        for layer in prunedConnections:
            columnTags = [f'N{i}' for i in range(prunedConnections[layer].shape[0])]
            df = pd.DataFrame(prunedConnections[layer].T, columns=columnTags)
            df.to_csv(f'savedModels/{modelName}_prunedIrregular_{relusStr}_{resizeFactor}_prunnedInfo{layer}.csv', index=False)
    
        metrics = '\n'.join(testReturn(test_dataloader, train_dataloader, model, criterion))
        print(metrics)
        with open(f'savedModels/{modelName}_prunedIrregular_{relusStr}_{resizeFactor}.txt', 'w') as f:
            f.write(metrics)
            f.write('\n')
            f.write(f'epochs {epochs}\n')
            f.write(f'batch {batch_size}\n')
            f.close()
        
        torch.save(model.state_dict(), f'savedModels/{modelName}_prunedIrregular_{relusStr}_{resizeFactor}')

    else:
        # First layer
        weightMask = list(model.l0.named_buffers())[0][1]
        idxs = []
        for iNeuron in range(len(weightMask)):
            weights = weightMask[iNeuron].detach().cpu().numpy()
            idxs.append(np.where(weights == 0)[0])
        columnTags = [f'N{i}' for i in range(len(weightMask))]
        df = pd.DataFrame(np.array(idxs).T, columns=columnTags)
        df.to_csv(f'savedModels/{modelName}_prunedRegular_{relusStr}_{resizeFactor}_prunnedInfo{0}.csv', index=False)

        # Second layer
        weightMask = list(model.l1.named_buffers())[0][1]
        idxs = []
        for iNeuron in range(len(weightMask)):
            weights = weightMask[iNeuron].detach().cpu().numpy()
            idxs.append(np.where(weights == 0)[0])
        columnTags = [f'N{i}' for i in range(len(weightMask))]
        df = pd.DataFrame(np.array(idxs).T, columns=columnTags)
        df.to_csv(f'savedModels/{modelName}_prunedRegular_{relusStr}_{resizeFactor}_prunnedInfo{1}.csv', index=False)

        # Third layer
        weightMask = list(model.l2.named_buffers())[0][1]
        idxs = []
        for iNeuron in range(len(weightMask)):
            weights = weightMask[iNeuron].detach().cpu().numpy()
            idxs.append(np.where(weights == 0)[0])
        columnTags = [f'N{i}' for i in range(len(weightMask))]
        df = pd.DataFrame(np.array(idxs).T, columns=columnTags)
        df.to_csv(f'savedModels/{modelName}_prunedRegular_{relusStr}_{resizeFactor}_prunnedInfo{2}.csv', index=False)

        metrics = '\n'.join(testReturn(test_dataloader, train_dataloader, model, criterion))
        print(metrics)
        with open(f'savedModels/{modelName}_prunedRegular_{relusStr}_{resizeFactor}.txt', 'w') as f:
            f.write(metrics)
            f.write('\n')
            f.write(f'epochs {epochs}\n')
            f.write(f'batch {batch_size}\n')
            f.close()
        
        torch.save(model.state_dict(), f'savedModels/{modelName}_prunedRegular_{relusStr}_{resizeFactor}')
    
    
if __name__ == '__main__':
    main('binaryVGGVerySmall2', [1, 1, 1, 1, 0, 0, 0, 0], 4, True, 30, 30)
    main('binaryVGGVerySmall2', [1, 1, 1, 1, 0, 0, 0, 0], 4, False, 30, 30)     
    main('binaryVGGSmall', [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], 4, True, 30, 30)
    main('binaryVGGSmall', [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], 4, False, 30, 30)
    main('binaryVGGVerySmall2', [1, 1, 1, 1, 0, 0, 0, 0], 7, True, 30, 30)
    main('binaryVGGVerySmall2', [1, 1, 1, 1, 0, 0, 0, 0], 7, False, 30, 30)    
    main('binaryVGGSmall', [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], 7, True, 30, 30)
    main('binaryVGGSmall', [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], 7, False, 30, 30)
    
