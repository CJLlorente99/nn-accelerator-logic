import pandas as pd
from modelsCommon.auxTransformations import *
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader
from aigsim import Model, Reader

aigFilename = "data/aigerFiles/example.aig"

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
model = Model()
reader = Reader()

reader.openFile(aigFilename)
reader.readHeader(model)
reader.readModel(model)

model.printSelf()
model.initModel()

# Simulate layer 0
tags = [f'N{i}' for i in range(28*28)]
outputs = []
for i in range(len(training_data)):
    X, y = train_dataloader.dataset[i]
    stim = X.squeeze(0)
    
    stepNum = model.step(stim)
    print(stepNum)

    if (i+1) % 500 == 0:
        print(f"Simulate layer 0 [{i+1:>5d}/{len(training_data):>5d}]")