import pandas as pd
import numpy as np
from ttUtilities.auxFunctions import integerToBinaryArray, binaryArrayToSingleValue
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from modules.binaryEnergyEfficiency import BinaryNeuralNetwork
import torch.nn as nn

# WARNING! The tt file is ordered

layer = 1

inputFilename = f'data/inputs/layer{layer}'
simulatedFilename = f'L{layer}_'
neuronPerLayer = 100
modelFilename = f'./src/modelCreation/savedModels/MNISTSignbinNN100Epoch{neuronPerLayer}NPLnllCriterion'

# Check mps maybe if working in MacOS
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = BinaryNeuralNetwork(neuronPerLayer).to(device)
model.load_state_dict(torch.load(modelFilename))

correct = 0
count = 0
with open(simulatedFilename) as f_sim:
    with open(inputFilename) as f_input:
        numLines = len(f_sim.readlines())
        f_sim.seek(0)
        while True:
            y_simulated = list(f_sim.readline())
            x_real = list(f_input.readline())
            if len(y_simulated):
                y_simulated.pop()
                line_simulated = np.array(y_simulated, dtype=np.double)
                
                x_real.pop()
                line_real = np.array(x_real, dtype=np.double)
                pred = model.forwardOneLayer(torch.tensor(line_real).type(torch.FloatTensor), layer)           
                
                if pred == line_simulated:
                    correct += 1
                
                count += 1
                if (count+1) % 5000 == 0:
                    print(f"Load inputs [{count+1:>5d}/{numLines:>5d}]")
            else:
                break     
            
print(f'Correct {correct}/{len(numLines)} {correct/len(numLines)*100}%')
    