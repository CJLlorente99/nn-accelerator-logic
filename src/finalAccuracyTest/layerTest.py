import pandas as pd
import numpy as np
from ttUtilities.auxFunctions import integerToBinaryArray, binaryArrayToSingleValue
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from modules.binaryEnergyEfficiency import BinaryNeuralNetwork
import torch.nn as nn

inputFilename = f'data/inputs/trainInput'
simulatedFilenameL0 = f'data/L0'
simulatedFilenameL1 = f'data/L1'
simulatedFilenameL2 = f'data/L2'
simulatedFilenameL3 = f'data/L3'
neuronPerLayer = 100
modelFilename = f'src\modelCreation\savedModels\eeb_100ep_100npl'

# Check mps maybe if working in MacOS
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = BinaryNeuralNetwork(neuronPerLayer).to(device)
model.load_state_dict(torch.load(modelFilename))
model.eval()

correctL0 = 0
correctL1 = 0
correctL2 = 0
correctL3 = 0
count = 0
with open(simulatedFilenameL0) as f_simL0:
    with open(simulatedFilenameL1) as f_simL1:
        with open(simulatedFilenameL2) as f_simL2:
            with open(simulatedFilenameL3) as f_simL3:
                with open(inputFilename) as f_input:
                    numLines = len(f_simL0.readlines())
                    f_simL0.seek(0)
                    while True:
                        y_simulatedL0 = list(f_simL0.readline())
                        y_simulatedL1 = list(f_simL1.readline())
                        y_simulatedL2 = list(f_simL2.readline())
                        y_simulatedL3 = list(f_simL3.readline())
                        x_real = list(f_input.readline())
                        if len(y_simulatedL0):
                            y_simulatedL0.pop()  # Cause last value is a \n
                            y_simulatedL1.pop()  # Cause last value is a \n
                            y_simulatedL2.pop()  # Cause last value is a \n
                            y_simulatedL3.pop()  # Cause last value is a \n

                            line_simulatedL0 = np.array(y_simulatedL0, dtype=np.double)
                            line_simulatedL0[line_simulatedL0 == 0] = -1

                            line_simulatedL1 = np.array(y_simulatedL1, dtype=np.double)
                            line_simulatedL1[line_simulatedL1 == 0] = -1

                            line_simulatedL2 = np.array(y_simulatedL2, dtype=np.double)
                            line_simulatedL2[line_simulatedL2 == 0] = -1

                            line_simulatedL3 = np.array(y_simulatedL3, dtype=np.double)
                            line_simulatedL3[line_simulatedL3 == 0] = -1
                            
                            x_real.pop()
                            line_real = np.array(x_real, dtype=np.double)
                            predL0 = model.forwardOneLayer(torch.tensor(line_real[None, :]).type(torch.FloatTensor), 0)           
                            predL1 = model.forwardOneLayer(torch.tensor(predL0).type(torch.FloatTensor), 1)           
                            predL2 = model.forwardOneLayer(torch.tensor(predL1).type(torch.FloatTensor), 2)           
                            predL3 = model.forwardOneLayer(torch.tensor(predL2).type(torch.FloatTensor), 3)           
                            
                            if (predL0.cpu().detach().numpy()[0] == line_simulatedL0).all():
                                correctL0 += 1

                            if (predL1.cpu().detach().numpy()[0] == line_simulatedL1).all():
                                correctL1 += 1

                            if (predL2.cpu().detach().numpy()[0] == line_simulatedL2).all():
                                correctL2 += 1

                            if (predL3.cpu().detach().numpy()[0] == line_simulatedL3).all():
                                correctL3 += 1
                            
                            count += 1
                            if (count+1) % 5000 == 0:
                                print(f"Load inputs [{count+1:>5d}/{numLines:>5d}]")
                        else:
                            break     
            
print(f'Correct L0 {correctL0}/{numLines} {correctL0/numLines*100}%')
print(f'Correct L1 {correctL1}/{numLines} {correctL1/numLines*100}%')
print(f'Correct L2 {correctL2}/{numLines} {correctL2/numLines*100}%')
print(f'Correct L3 {correctL3}/{numLines} {correctL3/numLines*100}%')
    