import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader
from modules.binaryEnergyEfficiency import BinaryNeuralNetwork
import torch.nn as nn
import pandas as pd
import numpy as np
from modelsCommon.auxTransformations import *
import torch.nn.functional as F

batch_size = 1
neuronPerLayer = 100

# Check mps maybe if working in MacOS
device = 'cuda' if torch.cuda.is_available() else 'cpu'

modelFilenames = ['eeb/eeb_prunedBT6_100ep_100npl',
                  'eeb/eeb_prunedBT8_100ep_100npl',
                  'eeb/eeb_prunedBT10_100ep_100npl',
                  'eeb/eeb_prunedBT12_100ep_100npl']
                  
subfolderPLA = ['ABC',
                'ESPRESSO',
                'ESPRESSOOptimizedPerClass_0',
                'ESPRESSOOptimizedPerClass_1',
                'ESPRESSOOptimizedPerClass_2',
                'ESPRESSOOptimizedPerEntry_0',
                'ESPRESSOOptimizedPerEntry_1',
                'ESPRESSOOptimizedPerEntry_2']

# Check mps maybe if working in MacOS
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

'''
Load the simulated inputs to the last layer (provided by minimized network)
'''

# Check on training data

def check(simulatedFilenameL0, simulatedFilenameL1, simulatedFilenameL2, simulatedFilenameL3, inputFilename, data, dataloader, key):
    f_simL0 = np.genfromtxt(simulatedFilenameL0, delimiter=1)
    columns = [f'N{i:04d}' for i in range(f_simL0.shape[1])]
    f_simL0 = pd.DataFrame(f_simL0, columns=columns)
    print(f'Loaded {simulatedFilenameL0}. Size {f_simL0.shape}')

    f_simL1 = np.genfromtxt(simulatedFilenameL1, delimiter=1)
    columns = [f'N{i:04d}' for i in range(f_simL1.shape[1])]
    f_simL1 = pd.DataFrame(f_simL1, columns=columns)
    print(f'Loaded {simulatedFilenameL1}. Size {f_simL1.shape}')

    f_simL2 = np.genfromtxt(simulatedFilenameL2, delimiter=1)
    columns = [f'N{i:04d}' for i in range(f_simL2.shape[1])]
    f_simL2 = pd.DataFrame(f_simL2, columns=columns)
    print(f'Loaded {simulatedFilenameL2}. Size {f_simL2.shape}')

    f_simL3 = np.genfromtxt(simulatedFilenameL3, delimiter=1)
    columns = [f'N{i:04d}' for i in range(f_simL3.shape[1])]
    f_simL3 = pd.DataFrame(f_simL3, columns=columns)
    print(f'Loaded {simulatedFilenameL3}. Size {f_simL3.shape}')

    f_input = np.genfromtxt(inputFilename, delimiter=1)
    columns = [f'N{i:04d}' for i in range(f_input.shape[1])]
    f_input = pd.DataFrame(f_input, columns=columns)
    print(f'Loaded {inputFilename}. Size {f_input.shape}')

    numLines[key] = len(f_simL1)

    for X, y in dataloader:
        y_simulatedL0 = f_simL0.iloc[count[key]]
        y_simulatedL1 = f_simL1.iloc[count[key]]
        y_simulatedL2 = f_simL2.iloc[count[key]]
        y_simulatedL3 = f_simL3.iloc[count[key]]
        x_sim = f_input.iloc[count[key]]

        line_simulatedL0 = np.array(y_simulatedL0, dtype=np.double)
        line_simulatedL0[line_simulatedL0 == 0] = -1
        
        line_simulatedL1 = np.array(y_simulatedL1, dtype=np.double)
        line_simulatedL1[line_simulatedL1 == 0] = -1

        line_simulatedL2 = np.array(y_simulatedL2, dtype=np.double)
        line_simulatedL2[line_simulatedL2 == 0] = -1

        line_simulatedL3 = np.array(y_simulatedL3, dtype=np.double)
        line_simulatedL3[line_simulatedL3 == 0] = -1

        line_sim = np.array(x_sim, dtype=np.double)
        line_sim[line_sim == 0] = -1
        
        inputSample = torch.tensor(line_sim[None, :]).type(torch.FloatTensor)
        X = torch.flatten(X, start_dim=1).to(device)
        predL0 = model.forwardOneLayer(X , 0)
        predL1 = model.forwardOneLayer(predL0, 1)           
        predL2 = model.forwardOneLayer(predL1, 2)           
        predL3 = model.forwardOneLayer(predL2, 3)
        pred = F.log_softmax(model.forwardOneLayer(predL3.type(torch.FloatTensor).to(device), 4) , dim=1)

        if (X.cpu().detach().numpy()[0] == inputSample.cpu().detach().numpy()[0]).all():
            correctInput[key] += 1

        # L0 Checks
        if (predL0.cpu().detach().numpy()[0] == line_simulatedL0).all():
            correctL0[key] += 1
        
        x = torch.tensor(line_simulatedL0[None, :]).type(torch.FloatTensor).to(device)
        x = model.forwardOneLayer(x, 2)
        x = model.forwardOneLayer(x, 3)
        x = model.forwardOneLayer(x, 4)
        if (x.argmax(1) == data.targets[count[key]].item()).type(torch.float).sum().item():
            correctAllModelFromL0[key] += 1

        # L1 Checks
        if (predL1.cpu().detach().numpy()[0] == line_simulatedL1).all():
            correctL1[key] += 1
        
        x = torch.tensor(line_simulatedL1[None, :]).type(torch.FloatTensor).to(device)
        x = model.forwardOneLayer(x, 2)
        x = model.forwardOneLayer(x, 3)
        x = model.forwardOneLayer(x, 4)
        if (x.argmax(1) == data.targets[count[key]].item()).type(torch.float).sum().item():
            correctAllModelFromL1[key] += 1

        # L2 Checks
        if (predL2.cpu().detach().numpy()[0] == line_simulatedL2).all():
            correctL2[key] += 1
        
        x = torch.tensor(line_simulatedL2[None, :]).type(torch.FloatTensor).to(device)
        x = model.forwardOneLayer(x, 3)
        x = model.forwardOneLayer(x, 4)
        if (x.argmax(1) == data.targets[count[key]].item()).type(torch.float).sum().item():
            correctAllModelFromL2[key] += 1

        # L3 Checks
        if (predL3.cpu().detach().numpy()[0] == line_simulatedL3).all():
            correctL3[key] += 1

        x = torch.tensor(line_simulatedL3[None, :]).type(torch.FloatTensor).to(device)
        x = model.forwardOneLayer(x, 4)
        if (x.argmax(1) == data.targets[count[key]].item()).type(torch.float).sum().item():
            correctAllModelFromL3[key] += 1

        # Original Accuracy
        if (pred.argmax(1) == data.targets[count[key]].item()).type(torch.float).sum().item():
            correctAllModel[key] += 1
        
        count[key] += 1
        if (count[key]+1) % 5000 == 0:
            print(f"Check inputs {key} [{count[key]+1:>5d}/{numLines[key]:>5d}]")   

if __name__ == "__main__":

    for modelFilename in modelFilenames:
        '''
        Instantiate NN models
        '''
        print(f'MODEL {modelFilename} INSTANTIATION\n')

        model = BinaryNeuralNetwork(neuronPerLayer, 1).to(device)
        model.load_state_dict(torch.load(f'data/savedModels/{modelFilename}', map_location=device))
        model.eval()

        with open(f'data/inputSimulated/{modelFilename}/results.txt', 'w') as f:
            for subFolder in subfolderPLA:

                print(f'Testing subfolder {subFolder}\n')
                f.write(f'====================================================\n')
                f.write(f'{subFolder}\n')
                f.write(f'====================================================\n')
                
                # try:
                # Train
                # Import dataset
                print(f'IMPORT DATASET\n')
                data = datasets.MNIST(
                    root='/srv/data/image_dataset/MNIST',
                    train=True,
                    download=False,
                    transform=Compose([
                            ToTensor(),
                            ToBlackAndWhite(),
                            ToSign()
                        ])
                    )

                dataloader = DataLoader(data, batch_size=batch_size)

                simulatedFilenameL0 = f'data/inputSimulated/{modelFilename}/{subFolder}/trainlayer1'
                simulatedFilenameL1 = f'data/inputSimulated/{modelFilename}/{subFolder}/trainlayer2'
                simulatedFilenameL2 = f'data/inputSimulated/{modelFilename}/{subFolder}/trainlayer3'
                simulatedFilenameL3 = f'data/inputSimulated/{modelFilename}/{subFolder}/trainlayer4'
                inputFilename = f'data/inputs/trainInput'
                key = 'train'

                correctInput = {'train': 0, 'test': 0}
                correctL0 = {'train': 0, 'test': 0}
                correctAllModelFromL0 = {'train': 0, 'test': 0}
                correctL1 = {'train': 0, 'test': 0}
                correctAllModelFromL1 = {'train': 0, 'test': 0}
                correctL2 = {'train': 0, 'test': 0}
                correctAllModelFromL2 = {'train': 0, 'test': 0}
                correctL3 = {'train': 0, 'test': 0}
                correctAllModelFromL3 = {'train': 0, 'test': 0}
                correctAllModel = {'train': 0, 'test': 0}
                count = {'train': 0, 'test': 0}
                numLines = {'train': 0, 'test': 0}

                check(simulatedFilenameL0, simulatedFilenameL1, simulatedFilenameL2, simulatedFilenameL3, inputFilename, data, dataloader, key)

                # Print Results
                print('\n================================================================================================')
                print('TRAIN RESULTS\n')
                print(f'Correct Input {correctInput[key]}/{numLines[key]} {correctInput[key] / numLines[key]*100}%')
                print(f'Correct L0 {correctL0[key]}/{numLines[key]} {correctL0[key] / numLines[key]*100}%')
                print(f"Train Error from L0: \n Accuracy: {(100 * correctAllModelFromL0[key] / numLines[key]):>0.2f}%\n")
                print(f'Correct L1 {correctL1[key]}/{numLines[key]} {correctL1[key] / numLines[key]*100}%')
                print(f"Train Error from L1: \n Accuracy: {(100 * correctAllModelFromL1[key] / numLines[key]):>0.2f}%\n")
                print(f'Correct L2 {correctL2[key]}/{numLines[key]} {correctL2[key] / numLines[key]*100}%')
                print(f"Train Error from L2: \n Accuracy: {(100 * correctAllModelFromL2[key] / numLines[key]):>0.2f}%\n")
                print(f'Correct L3 {correctL3[key]}/{numLines[key]} {correctL3[key] / numLines[key]*100}%')
                print(f"Train Error from L3: \n Accuracy: {(100 * correctAllModelFromL3[key] / numLines[key]):>0.2f}%\n")
                print(f"Train Error Original: \n Accuracy: {(100 * correctAllModel[key] / numLines[key]):>0.2f}%\n")
                print('================================================================================================\n')

                text = []
                text.append('\n================================================================================================')
                text.append('TRAIN RESULTS\n')
                text.append(f'Correct Input {correctInput[key]}/{numLines[key]} {correctInput[key] / numLines[key]*100}%')
                text.append(f'Correct L0 {correctL0[key]}/{numLines[key]} {correctL0[key] / numLines[key]*100}%')
                text.append(f"Train Error from L0: \n Accuracy: {(100 * correctAllModelFromL0[key] / numLines[key]):>0.2f}%\n")
                text.append(f'Correct L1 {correctL1[key]}/{numLines[key]} {correctL1[key] / numLines[key]*100}%')
                text.append(f"Train Error from L1: \n Accuracy: {(100 * correctAllModelFromL1[key] / numLines[key]):>0.2f}%\n")
                text.append(f'Correct L2 {correctL2[key]}/{numLines[key]} {correctL2[key] / numLines[key]*100}%')
                text.append(f"Train Error from L2: \n Accuracy: {(100 * correctAllModelFromL2[key] / numLines[key]):>0.2f}%\n")
                text.append(f'Correct L3 {correctL3[key]}/{numLines[key]} {correctL3[key] / numLines[key]*100}%')
                text.append(f"Train Error from L3: \n Accuracy: {(100 * correctAllModelFromL3[key] / numLines[key]):>0.2f}%\n")
                text.append(f"Train Error Original: \n Accuracy: {(100 * correctAllModel[key] / numLines[key]):>0.2f}%\n")
                text.append('================================================================================================\n')
                f.write('\n'.join(text))

                # Test
                # Import dataset
                print(f'IMPORT DATASET\n')
                data = datasets.MNIST(
                    root='/srv/data/image_dataset/MNIST',
                    train=False,
                    download=False,
                    transform=Compose([
                            ToTensor(),
                            ToBlackAndWhite(),
                            ToSign()
                        ])
                    )

                dataloader = DataLoader(data, batch_size=batch_size)

                simulatedFilenameL0 = f'data/inputSimulated/{modelFilename}/{subFolder}/testlayer1'
                simulatedFilenameL1 = f'data/inputSimulated/{modelFilename}/{subFolder}/testlayer2'
                simulatedFilenameL2 = f'data/inputSimulated/{modelFilename}/{subFolder}/testlayer3'
                simulatedFilenameL3 = f'data/inputSimulated/{modelFilename}/{subFolder}/testlayer4'
                inputFilename = f'data/inputs/testInput'
                key = 'test'

                check(simulatedFilenameL0, simulatedFilenameL1, simulatedFilenameL2, simulatedFilenameL3, inputFilename, data, dataloader, key)

                # Print Results
                print('\n================================================================================================')
                print('TEST RESULTS\n')
                print(f'Correct Input {correctInput[key]}/{numLines[key]} {correctInput[key] / numLines[key]*100}%')
                print(f'Correct L0 {correctL0[key]}/{numLines[key]} {correctL0[key] / numLines[key]*100}%')
                print(f"Train Error from L0: \n Accuracy: {(100 * correctAllModelFromL0[key] / numLines[key]):>0.2f}%\n")
                print(f'Correct L1 {correctL1[key]}/{numLines[key]} {correctL1[key] / numLines[key]*100}%')
                print(f"Train Error from L1: \n Accuracy: {(100 * correctAllModelFromL1[key] / numLines[key]):>0.2f}%\n")
                print(f'Correct L2 {correctL2[key]}/{numLines[key]} {correctL2[key] / numLines[key]*100}%')
                print(f"Train Error from L2: \n Accuracy: {(100 * correctAllModelFromL2[key] / numLines[key]):>0.2f}%\n")
                print(f'Correct L3 {correctL3[key]}/{numLines[key]} {correctL3[key] / numLines[key]*100}%')
                print(f"Train Error from L3: \n Accuracy: {(100 * correctAllModelFromL3[key] / numLines[key]):>0.2f}%\n")
                print(f"Train Error Original: \n Accuracy: {(100 * correctAllModel[key] / numLines[key]):>0.2f}%\n")
                print('================================================================================================\n')

                text = []
                text.append('\n================================================================================================')
                text.append('TEST RESULTS\n')
                text.append(f'Correct Input {correctInput[key]}/{numLines[key]} {correctInput[key] / numLines[key]*100}%')
                text.append(f'Correct L0 {correctL0[key]}/{numLines[key]} {correctL0[key] / numLines[key]*100}%')
                text.append(f"Train Error from L0: \n Accuracy: {(100 * correctAllModelFromL0[key] / numLines[key]):>0.2f}%\n")
                text.append(f'Correct L1 {correctL1[key]}/{numLines[key]} {correctL1[key] / numLines[key]*100}%')
                text.append(f"Train Error from L1: \n Accuracy: {(100 * correctAllModelFromL1[key] / numLines[key]):>0.2f}%\n")
                text.append(f'Correct L2 {correctL2[key]}/{numLines[key]} {correctL2[key] / numLines[key]*100}%')
                text.append(f"Train Error from L2: \n Accuracy: {(100 * correctAllModelFromL2[key] / numLines[key]):>0.2f}%\n")
                text.append(f'Correct L3 {correctL3[key]}/{numLines[key]} {correctL3[key] / numLines[key]*100}%')
                text.append(f"Train Error from L3: \n Accuracy: {(100 * correctAllModelFromL3[key] / numLines[key]):>0.2f}%\n")
                text.append(f"Train Error Original: \n Accuracy: {(100 * correctAllModel[key] / numLines[key]):>0.2f}%\n")
                text.append('================================================================================================\n')
                f.write('\n'.join(text))

                # Both
                print('\n================================================================================================')
                print('TOTAL RESULTS\n')
                print(f'Correct Input {sum(correctInput.values())}/{sum(numLines.values())} {sum(correctInput.values())/sum(numLines.values())*100}%')
                print(f'Correct L0 {sum(correctL0.values())}/{sum(numLines.values())} {sum(correctL0.values())/sum(numLines.values())*100}%')
                print(f"Train Error from L0: \n Accuracy: {(100 * sum(correctAllModelFromL0.values()) / sum(numLines.values())):>0.2f}%\n")
                print(f'Correct L1 {sum(correctL1.values())}/{sum(numLines.values())} {sum(correctL1.values())/sum(numLines.values())*100}%')
                print(f"Train Error from L1: \n Accuracy: {(100 * sum(correctAllModelFromL1.values()) / sum(numLines.values())):>0.2f}%\n")
                print(f'Correct L2 {sum(correctL2.values())}/{sum(numLines.values())} {sum(correctL2.values())/sum(numLines.values())*100}%')
                print(f"Train Error from L2: \n Accuracy: {(100 * sum(correctAllModelFromL2.values()) / sum(numLines.values())):>0.2f}%\n")
                print(f'Correct L3 {sum(correctL3.values())}/{sum(numLines.values())} {sum(correctL3.values())/sum(numLines.values())*100}%')
                print(f"Train Error from L3: \n Accuracy: {(100 * sum(correctAllModelFromL3.values()) / sum(numLines.values())):>0.2f}%\n")
                print(f"Train Error Original: \n Accuracy: {(100 * sum(correctAllModel.values()) / sum(numLines.values())):>0.2f}%\n")
                print('================================================================================================\n')
                
                text = []
                text.append('\n================================================================================================')
                text.append('TOTAL RESULTS\n')
                text.append(f'Correct Input {sum(correctInput.values())}/{sum(numLines.values())} {sum(correctInput.values())/sum(numLines.values())*100}%')
                text.append(f'Correct L0 {sum(correctL0.values())}/{sum(numLines.values())} {sum(correctL0.values())/sum(numLines.values())*100}%')
                text.append(f"Train Error from L0: \n Accuracy: {(100 * sum(correctAllModelFromL0.values()) / sum(numLines.values())):>0.2f}%\n")
                text.append(f'Correct L1 {sum(correctL1.values())}/{sum(numLines.values())} {sum(correctL1.values())/sum(numLines.values())*100}%')
                text.append(f"Train Error from L1: \n Accuracy: {(100 * sum(correctAllModelFromL1.values()) / sum(numLines.values())):>0.2f}%\n")
                text.append(f'Correct L2 {sum(correctL2.values())}/{sum(numLines.values())} {sum(correctL2.values())/sum(numLines.values())*100}%')
                text.append(f"Train Error from L2: \n Accuracy: {(100 * sum(correctAllModelFromL2.values()) / sum(numLines.values())):>0.2f}%\n")
                text.append(f'Correct L3 {sum(correctL3.values())}/{sum(numLines.values())} {sum(correctL3.values())/sum(numLines.values())*100}%')
                text.append(f"Train Error from L3: \n Accuracy: {(100 * sum(correctAllModelFromL3.values()) / sum(numLines.values())):>0.2f}%\n")
                text.append(f"Train Error Original: \n Accuracy: {(100 * sum(correctAllModel.values()) / sum(numLines.values())):>0.2f}%\n")
                text.append('================================================================================================\n')
                f.write('\n'.join(text))
                # except:
                #     print('FAILURE\n')
                #     f.write('FAILURE\n')
        f.close()