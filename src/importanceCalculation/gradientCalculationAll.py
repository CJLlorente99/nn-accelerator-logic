import torch
from modelsCommon.auxTransformations import ToBlackAndWhite, ToSign
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader
from modules.binaryEnergyEfficiencyAllGradients import BinaryNeuralNetwork
from ttUtilities.helpLayerNeuronGenerator import HelpGenerator

neuronPerLayer = 100
modelFilename = f'src\modelCreation\savedModels\MNISTSignbinNN100Epoch100NPLnllCriterion'
batch_size = 64
perGradientSampling = 1
# Check mps maybe if working in MacOS
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

sampleSize = int(perGradientSampling * len(training_data.data))  # sample size to be used for importance calculation

'''
Create DataLoader
'''
train_dataloader = DataLoader(training_data, batch_size=batch_size)

model = BinaryNeuralNetwork(neuronPerLayer)
model.load_state_dict(torch.load(modelFilename))

'''
Generate AccLayers and Neuron objects
'''

accLayers = HelpGenerator.getAccLayers(model)
HelpGenerator.getNeurons(accLayers)

print(f'Number of layers is {len(accLayers)}')
i = 0
for layer in accLayers:
    print(f'Layer {i} has {layer.nNeurons} neurons')
    i += 1

'''
Calculate importance per class per neuron
'''

# Input samples and get gradients and values in each neuron
print(f'GET GRADIENTS AND ACTIVATION VALUES\n')

model.registerHooks()
model.eval()

for i in range(sampleSize):
    X, y = train_dataloader.dataset[i]
    model.zero_grad()
    pred = model(X)
    pred[0, y].backward()

    if (i+1) % 500 == 0:
        print(f"Get Gradients and Activation Values [{i+1:>5d}/{sampleSize:>5d}]")

model.listToArray(neuronPerLayer)  # Hopefully improves memory usage
model.saveActivations(f'data/activations/activationsSignBin100epochs{neuronPerLayer}npl')
model.saveGradients(f'data/gradients/gradientsSignBin100epochs{neuronPerLayer}npl', training_data.targets.tolist()[:sampleSize])

