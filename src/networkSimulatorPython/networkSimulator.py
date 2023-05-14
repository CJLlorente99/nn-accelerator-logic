import aiger
import pandas as pd
from utilities.auxTransformations import *
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader

aigFilename = "data/aigerFiles/layer0_MNISTSignbinNN100Epoch100NPLnllCriterionoutputL1N1.aig"

# Inputs for layer 0
'''
Importing MNIST dataset
'''
print(f'IMPORT DATASET\n')

training_data = datasets.MNIST(
    root='C:/Users/carlo/OneDrive/Documentos/Universidad/MUIT/Segundo/TFM/Code/data',
    train=True,
    download=False,
    transform=Compose([
            ToTensor(),
            ToBlackAndWhite(),
            ToSign()
        ])
)

test_data = datasets.MNIST(
    root='C:/Users/carlo/OneDrive/Documentos/Universidad/MUIT/Segundo/TFM/Code/data',
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
train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# Parser for ascii AIGER format.
aig = aiger.load(aigFilename)

# Simulation Coroutine
sim = aig.simulator()  # Coroutine

# Simulate layer 0
tags = [f'N{i}' for i in range(28*28)]
outputs = []
for i in range(len(training_data)):
    X, y = train_dataloader.dataset[i]
    auxDict = pd.DataFrame(X, columns=tags).to_dict()
    
    next(sim)  # Initialize
    outputs.apppend(sim.send(auxDict))

    if (i+1) % 500 == 0:
        print(f"Simulate layer 0 [{i+1:>5d}/{len(training_data):>5d}]")