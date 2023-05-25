import torch
from modelsCommon.auxTransformations import ToBlackAndWhite, ToSign
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader
from modules.binaryEnergyEfficiency import BinaryNeuralNetwork
from ttUtilities.helpLayerNeuronGenerator import HelpGenerator

neuronPerLayer = 100
modelFilename = f'src\modelCreation\savedModels\MNISTSignbinNN100Epoch100NPLnllCriterion'

model = BinaryNeuralNetwork(neuronPerLayer)
model.load_state_dict(torch.load(modelFilename))

pass

