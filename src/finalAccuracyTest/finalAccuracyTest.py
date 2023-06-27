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
modelFilename = f'src\modelCreation\savedModels\eeb_100ep_100npl'
simulatedFilenameL0 = f'data/L0'
simulatedFilenameL1 = f'data/L1'
simulatedFilenameL2 = f'data/L2'
simulatedFilenameL3 = f'data/L3'
inputFilename = f'data/inputs/trainInput'
lastLayerInputsTestFilename = f'example'


# Check mps maybe if working in MacOS
device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
Importing MNIST dataset
'''
print(f'IMPORT DATASET\n')

training_data = datasets.MNIST(
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
    root='data',
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
model.load_state_dict(torch.load(modelFilename))

'''
Load the simulated inputs to the last layer (provided by minimized network)
'''

correctInput = 0
correctL0 = 0
correctL1 = 0
correctL2 = 0
correctL3 = 0
correctAllModel = 0
count = 0
model.eval()
with open(simulatedFilenameL0) as f_simL0:
    with open(simulatedFilenameL1) as f_simL1:
        with open(simulatedFilenameL2) as f_simL2:
            with open(simulatedFilenameL3) as f_simL3:
                with open(inputFilename) as f_input:
                    numLines = len(f_simL0.readlines())
                    f_simL0.seek(0)
                    for X, y in train_dataloader:
                        y_simulatedL0 = list(f_simL0.readline())
                        y_simulatedL1 = list(f_simL1.readline())
                        y_simulatedL2 = list(f_simL2.readline())
                        y_simulatedL3 = list(f_simL3.readline())
                        x_sim = list(f_input.readline())

                        if len(y_simulatedL0):
                            y_simulatedL0.pop()  # Cause last value is a \n
                            y_simulatedL1.pop()  # Cause last value is a \n
                            y_simulatedL2.pop()  # Cause last value is a \n
                            y_simulatedL3.pop()  # Cause last value is a \n
                            x_sim.pop()

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
                            X = torch.flatten(X, start_dim=1)
                            predL0 = model.forwardOneLayer(X , 0)
                            predL1 = model.forwardOneLayer(predL0, 1)           
                            predL2 = model.forwardOneLayer(predL1, 2)           
                            predL3 = model.forwardOneLayer(predL2, 3)
                            pred = F.log_softmax(model.forwardOneLayer(predL2.type(torch.FloatTensor), 4) , dim=1)

                            if (X.cpu().detach().numpy()[0] == inputSample.cpu().detach().numpy()[0]).all():
                                print('Input Equal')
                                correctInput += 1
                            
                            if (predL0.cpu().detach().numpy()[0] == line_simulatedL0).all():
                                print('L0 Equal')
                                correctL0 += 1

                            if (predL1.cpu().detach().numpy()[0] == line_simulatedL1).all():
                                print('L1 Equal')
                                correctL1 += 1

                            if (predL2.cpu().detach().numpy()[0] == line_simulatedL2).all():
                                print('L2 Equal')
                                correctL2 += 1

                            if (predL3.cpu().detach().numpy()[0] == line_simulatedL3).all():
                                print('L3 Equal')
                                correctL3 += 1

                            if (pred.argmax(1) == training_data.targets[count].item()).type(torch.float).sum().item():
                                print('Result Equal')
                                correctAllModel += 1
                            
                            count += 1
                            if (count+1) % 5000 == 0:
                                print(f"Load inputs [{count+1:>5d}/{numLines:>5d}]")
                        else:
                            break    
                        
# dfInputsLastLayerTest = pd.DataFrame(columns=columnTags)
        
# with open(lastLayerInputsTestFilename) as f:
#         while True:
#             line = np.array(list(f.readline()), dtype=np.float)
#             if not line: break
            
#             dfInputsLastLayerTest = pd.concat(dfInputsLastLayerTest,
#                                           pd.DataFrame(line, columns=columnTags))

"""
Print results
"""

print(f'Correct Input {correctInput}/{numLines} {correctInput/numLines*100}%')
print(f'Correct L0 {correctL0}/{numLines} {correctL0/numLines*100}%')
print(f'Correct L1 {correctL1}/{numLines} {correctL1/numLines*100}%')
print(f'Correct L2 {correctL2}/{numLines} {correctL2/numLines*100}%')
print(f'Correct L3 {correctL3}/{numLines} {correctL3/numLines*100}%')
print(f"Train Error: \n Accuracy: {(100 * correctAllModel / numLines):>0.2f}%\n")
    
# print(f'TEST Test\n')

# correct = 0
# size = len(test_dataloader.dataset)
# count = 0
# for index, row in dfInputsLastLayerTest.iterrows():
#     with torch.no_grad():
#         pred = model.forwardLastLayer(row)
#         correct += (pred.argmax(1) == test_dataloader.target[count]).type(torch.float).sum().item()
#         count += 1

# totalCorrect += correct
# print(f"Test Error: \n Accuracy: {(100 * correct / size):>0.2f}%\n")
