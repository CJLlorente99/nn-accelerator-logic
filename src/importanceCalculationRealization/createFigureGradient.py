import matplotlib.pyplot as plt
from modelsCommon.auxTransformations import ToBlackAndWhite, ToSign
from torchvision.transforms import ToTensor, Compose
from torchvision import datasets
import pandas as pd
import numpy as np
import os

modelName = 'bnn/bnn_prunedBT6_100ep_4096npl'

ste1GradientFilename = f'data/gradients/{modelName}/STE1'
ste2GradientFilename = f'data/gradients/{modelName}/STE2'
ste3GradientFilename = f'data/gradients/{modelName}/STE3'

if not os.path.exists(f'img/gradientHeatmap/{modelName}'):
    os.makedirs(f'img/gradientHeatmap/{modelName}')

'''
Import dataset
'''

training_data = datasets.MNIST(
    root='./data',
    train=True,
    download=False,
    transform=Compose([
            ToTensor(),
            ToBlackAndWhite(),
            ToSign()
        ])
)

targetsDf = pd.DataFrame(training_data.targets.cpu().detach().numpy(), columns=['targets'])
targetsDf.sort_values(['targets'], inplace=True)

# Aux function

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

xTicks = np.arange(0, len(targetsDf) + len(targetsDf)/10, len(targetsDf)/10)

'''
Plot STE1
'''

df = pd.read_feather(ste1GradientFilename)
df = df.reindex(targetsDf.index)
aux = np.abs(df.to_numpy().T)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(aux, cmap='hot', interpolation='nearest')
plt.xlabel('MNIST train images, grouped by category')
plt.ylabel('Gradient value')
plt.colorbar()
plt.title(f'Layer 1 Gradients per Training Sample and Neuron')
ax.set_xticks(xTicks)
plt.xticks(rotation=45)
forceAspect(ax,aspect=1)
plt.tight_layout()
fig.savefig(f'img/gradientHeatmap/{modelName}/STE1.png', transparent=True)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(aux[:50,:], cmap='hot', interpolation='nearest')
plt.xlabel('MNIST train images, grouped by category')
plt.ylabel('Gradient value')
plt.colorbar()
plt.title(f'Layer 1 Gradients per Training Sample and Neuron (50 Neurons)')
ax.set_xticks(xTicks)
plt.xticks(rotation=45)
forceAspect(ax,aspect=1)
plt.tight_layout()
fig.savefig(f'img/gradientHeatmap/{modelName}/STE1detail.png', transparent=True)

'''
Plot STE2
'''

df = pd.read_feather(ste2GradientFilename)
df = df.reindex(targetsDf.index)
aux = np.abs(df.to_numpy().T)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(aux, cmap='hot', interpolation='nearest')
plt.xlabel('MNIST train images, grouped by category')
plt.ylabel('Gradient value')
plt.colorbar()
plt.title(f'Layer 2 Gradients per Training Sample and Neuron')
forceAspect(ax,aspect=1)
ax.set_xticks(xTicks)
plt.xticks(rotation=45)
plt.tight_layout()
fig.savefig(f'img/gradientHeatmap/{modelName}/STE2.png', transparent=True)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(aux[:50,:], cmap='hot', interpolation='nearest')
plt.xlabel('MNIST train images, grouped by category')
plt.ylabel('Gradient value')
plt.colorbar()
plt.title(f'Layer 2 Gradients per Training Sample and Neuron (50 Neurons)')
forceAspect(ax,aspect=1)
ax.set_xticks(xTicks)
plt.xticks(rotation=45)
plt.tight_layout()
fig.savefig(f'img/gradientHeatmap/{modelName}/STE2detail.png', transparent=True)

'''
Plot STE3
'''

df = pd.read_feather(ste1GradientFilename)
df = df.reindex(targetsDf.index)
aux = np.abs(df.to_numpy().T)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(aux, cmap='hot', interpolation='nearest')
plt.xlabel('MNIST train images, grouped by category')
plt.ylabel('Gradient value')
plt.colorbar()
plt.title(f'Layer 3 Gradients per Training Sample and Neuron')
forceAspect(ax,aspect=1)
ax.set_xticks(xTicks)
plt.xticks(rotation=45)
plt.tight_layout()
fig.savefig(f'img/gradientHeatmap/{modelName}/STE3.png', transparent=True)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(aux[:50,:], cmap='hot', interpolation='nearest')
plt.xlabel('MNIST train images, grouped by category')
plt.ylabel('Gradient value')
plt.colorbar()
plt.title(f'Layer 3 Gradients per Training Sample and Neuron (50 Neurons)')
forceAspect(ax,aspect=1)
ax.set_xticks(xTicks)
plt.xticks(rotation=45)
plt.tight_layout()
fig.savefig(f'img/gradientHeatmap/{modelName}/STE3detail.png', transparent=True)
