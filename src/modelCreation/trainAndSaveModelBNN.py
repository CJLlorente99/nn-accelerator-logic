import numpy as np
import torch
from modelsCommon.auxFunctionsBNN import trainAndTest, testReturn
from modelsCommon.auxTransformations import *
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader
from modules.binaryBNN import BNNBinaryNeuralNetwork
import torch.optim as optim
from torchmetrics.classification import MulticlassHingeLoss
import pandas as pd

batch_size = 128
neuronPerLayer = 4096
epochs = 200
pruned = True

# Check mps maybe if working in MacOS
device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
Importing MNIST dataset
'''
print(f'IMPORT DATASET\n')

training_data = datasets.MNIST(
    root='/srv/data/image_dataset/MNIST',
    train=True,
    download=False,
    transform=Compose([
            ToTensor(),
            ToBlackAndWhite(),
            ToSign()
        ])
)

test_data = datasets.MNIST(
    root='/srv/data/image_dataset/MNIST',
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

model = BNNBinaryNeuralNetwork(neuronPerLayer, False).to(device)

'''
Train and test
'''
print(f'TRAINING\n')

initialLr = 3e-3
finalLr = 3e-7
step = (finalLr/initialLr)**(1/epochs)

my_list = [f'l{i+1}.weight' for i in range(3)] + [f'l{i+1}.bias' for i in range(3)]
all_list = [f'l{i}.weight' for i in range(5)] + [f'l{i}.bias' for i in range(5)]
ioLayerParams = list(filter(lambda kv: kv[0] in ['l0.weight', 'l0.bias', 'l4.weight', 'l4.bias'], list(model.named_parameters())))
ioLayerParams = [i[1] for i in ioLayerParams]
hiddenLayersParams = list(filter(lambda kv: kv[0] in my_list, list(model.named_parameters())))
hiddenLayersParams = [i[1] for i in hiddenLayersParams]
base_params = list(filter(lambda kv: kv[0] not in all_list, model.named_parameters()))
base_params = [i[1] for i in base_params]

tailoredParamList = [{'params': base_params,
                      'lr': initialLr},
                     {'params': hiddenLayersParams,
                      'lr': initialLr / (np.sqrt(1.5 / (2 * neuronPerLayer)))},
                     {'params': ioLayerParams,
                      'lr': initialLr / (np.sqrt(1.5 / (10 + neuronPerLayer)))}]  # 10 = numClasses

opt = optim.Adam(tailoredParamList, lr=initialLr)

criterion = MulticlassHingeLoss(num_classes=10, squared=True, multiclass_mode='one-vs-all')

trainAndTest(epochs, train_dataloader, test_dataloader, model, opt, criterion, step)

'''
Save
'''
print(f'SAVING\n')

if pruned:
    prunedConnections = model.pruningSparsification(neuronPerLayer - 30)
    for layer in prunedConnections:
        columnTags = [f'N{i}' for i in range(prunedConnections[layer].shape[0])]
        df = pd.DataFrame(prunedConnections[layer].T, columns=columnTags)
        df.to_csv(f'savedModels/bnn_pruned_{epochs}ep_{neuronPerLayer}npl_prunnedInfo{layer}.csv', index=False)

metrics = '\n'.join(testReturn(test_dataloader, train_dataloader, model, criterion))
print(metrics)
with open(f'savedModels/bnn_pruned_{epochs}ep_{neuronPerLayer}npl.txt', 'w') as f:
    f.write(metrics)
    f.write('\n')
    f.write(f'epochs {epochs}\n')
    f.write(f'batch {batch_size}\n')
    f.close()

torch.save(model.state_dict(), f'savedModels/bnn_pruned_{epochs}ep_{neuronPerLayer}npl')


