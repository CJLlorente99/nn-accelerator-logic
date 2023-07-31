import torch
from modelsCommon.auxFunctionsEnergyEfficiency import trainAndTest, testReturn
from modelsCommon.auxTransformations import *
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader
from modules.binaryEnergyEfficiency import BinaryNeuralNetwork
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np

def main(atPruning, inputsAfterBeforeTrainingPrune, inputsAfterAfterTrainingPrune):
    print(f'atPruning: {atPruning}, inputsAfterBeforeTrainingPrune: {inputsAfterBeforeTrainingPrune}, inputsAfterAfterTrainingPrune: {inputsAfterAfterTrainingPrune}')
    batch_size = 64
    neuronPerLayer = 100
    epochs = 100
    # atPruning = False # Before Training False, After Training True
    # If regular prune
    # inputsAfterBeforeTrainingPrune = 30
    if atPruning:
        inputsAfterBeforeTrainingPrune = neuronPerLayer
    # If irregular prune
    # inputsAfterAfterTrainingPrune = 30
    if not atPruning:
        inputsAfterAfterTrainingPrune = neuronPerLayer

    # Check mps maybe if working in MacOS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    '''
    Importing MNIST dataset
    '''
    print(f'IMPORT DATASET\n')

    training_data = datasets.MNIST(
        # root='/srv/data/image_dataset/MNIST',
        root='data',
        train=True,
        download=False,
        transform=Compose([
                ToTensor(),
                ToBlackAndWhite(),
                ToSign()
            ])
    )

    test_data = datasets.MNIST(
        # root='/srv/data/image_dataset/MNIST',
        root = 'data',
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

    model = BinaryNeuralNetwork(neuronPerLayer, neuronPerLayer - inputsAfterBeforeTrainingPrune).to(device)

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

    if atPruning:
        prunedConnections = model.pruningSparsification(neuronPerLayer - inputsAfterAfterTrainingPrune)
        for layer in prunedConnections:
            columnTags = [f'N{i:04}' for i in range(prunedConnections[layer].shape[0])]
            df = pd.DataFrame(prunedConnections[layer].T, columns=columnTags)
            df.to_csv(f'savedModels/eeb_prunedAT{inputsAfterAfterTrainingPrune}_{epochs}ep_{neuronPerLayer}npl_prunedInfol{layer}.csv', index=False)

        metrics = '\n'.join(testReturn(test_dataloader, train_dataloader, model, criterion))
        print(metrics)
        with open(f'savedModels/eeb_prunedAT{inputsAfterAfterTrainingPrune}_{epochs}ep_{neuronPerLayer}npl.txt', 'w') as f:
            f.write(metrics)
            f.write('\n')
            f.write(f'epochs {epochs}\n')
            f.write(f'batch {batch_size}\n')
            f.close()

        torch.save(model.state_dict(), f'savedModels/eeb_prunedAT{inputsAfterAfterTrainingPrune}_{epochs}ep_{neuronPerLayer}npl')
    else:
        # First layer
        weightMask = list(model.l1.named_buffers())[0][1]
        idxs = []
        for iNeuron in range(len(weightMask)):
            weights = weightMask[iNeuron].detach().cpu().numpy()
            idxs.append(np.where(weights == 0)[0])
        columnTags = [f'N{i:04}' for i in range(len(weightMask))]
        df = pd.DataFrame(np.array(idxs).T, columns=columnTags)
        df.to_csv(f'savedModels/eeb_prunedBT{inputsAfterBeforeTrainingPrune}_{epochs}ep_{neuronPerLayer}npl_prunedInfol{1}.csv', index=False)

        # Second layer
        weightMask = list(model.l2.named_buffers())[0][1]
        idxs = []
        for iNeuron in range(len(weightMask)):
            weights = weightMask[iNeuron].detach().cpu().numpy()
            idxs.append(np.where(weights == 0)[0])
        columnTags = [f'N{i:04}' for i in range(len(weightMask))]
        df = pd.DataFrame(np.array(idxs).T, columns=columnTags)
        df.to_csv(f'savedModels/eeb_prunedBT{inputsAfterBeforeTrainingPrune}_{epochs}ep_{neuronPerLayer}npl_prunedInfol{2}.csv', index=False)

        # Third layer
        weightMask = list(model.l3.named_buffers())[0][1]
        idxs = []
        for iNeuron in range(len(weightMask)):
            weights = weightMask[iNeuron].detach().cpu().numpy()
            idxs.append(np.where(weights == 0)[0])
        columnTags = [f'N{i:04}' for i in range(len(weightMask))]
        df = pd.DataFrame(np.array(idxs).T, columns=columnTags)
        df.to_csv(f'savedModels/eeb_prunedBT{inputsAfterBeforeTrainingPrune}_{epochs}ep_{neuronPerLayer}npl_prunedInfol{3}.csv', index=False)

        metrics = '\n'.join(testReturn(test_dataloader, train_dataloader, model, criterion))
        print(metrics)
        with open(f'savedModels/eeb_prunedBT{inputsAfterBeforeTrainingPrune}_{epochs}ep_{neuronPerLayer}npl.txt', 'w') as f:
            f.write(metrics)
            f.write('\n')
            f.write(f'epochs {epochs}\n')
            f.write(f'batch {batch_size}\n')
            f.close()

        torch.save(model.state_dict(), f'savedModels/eeb_prunedBT{inputsAfterBeforeTrainingPrune}_{epochs}ep_{neuronPerLayer}npl')

if __name__ == '__main__':
    main(False, 6, 6)
    main(True, 6, 6)
    main(False, 8, 8)
    main(True, 8, 8)
    main(False, 10, 10)
    main(True, 10, 10)
    main(False, 12, 12)
    main(True, 12, 12)
