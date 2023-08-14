import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, Compose, Normalize, RandomHorizontalFlip, RandomCrop, Resize
from torchvision import datasets
import pandas as pd
import numpy as np
import os

# modelName = 'binaryVggSmall/binaryVGGSmall_prunedBT6_4'
# modelName = 'binaryVggSmall/binaryVGGSmall_prunedBT8_4'
# modelName = 'binaryVggSmall/binaryVGGSmall_prunedBT10_4'
# modelName = 'binaryVggSmall/binaryVGGSmall_prunedBT12_4'
# modelName = 'binaryVggVerySmall/binaryVGGVerySmall_prunedBT6_4'
# modelName = 'binaryVggVerySmall/binaryVGGVerySmall_prunedBT8_4'
# modelName = 'binaryVggVerySmall/binaryVGGVerySmall_prunedBT10_4'
# modelName = 'binaryVggVerySmall/binaryVGGVerySmall_prunedBT12_4'
resizeFactor = 4
threshold = 10e-5

ste1GradientFilename = f'data/gradients/{modelName}/STEL0'
ste2GradientFilename = f'data/gradients/{modelName}/STEL1'
ste3GradientFilename = f'data/gradients/{modelName}/STEL2'

if not os.path.exists(f'img/gradientHeatmapThreshold/{modelName}'):
    os.makedirs(f'img/gradientHeatmapThreshold/{modelName}')

'''
Import dataset
'''

training_data = datasets.CIFAR10(root='data', train=True, transform=Compose([
            RandomHorizontalFlip(),
            RandomCrop(32, 4),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            Resize(resizeFactor*32, antialias=False)]),
        download=False)

targetsDf = pd.DataFrame(np.array(training_data.targets), columns=['targets'])
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
df[df < threshold] = 0
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
fig.savefig(f'img/gradientHeatmapThreshold/{modelName}/STE1.png', transparent=True)

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
fig.savefig(f'img/gradientHeatmapThreshold/{modelName}/STE1detail.png', transparent=True)

'''
Plot STE2
'''

df = pd.read_feather(ste2GradientFilename)
df[df < threshold] = 0
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
fig.savefig(f'img/gradientHeatmapThreshold/{modelName}/STE2.png', transparent=True)

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
fig.savefig(f'img/gradientHeatmapThreshold/{modelName}/STE2detail.png', transparent=True)

'''
Plot STE3
'''

df = pd.read_feather(ste1GradientFilename)
df[df < threshold] = 0
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
fig.savefig(f'img/gradientHeatmapThreshold/{modelName}/STE3.png', transparent=True)

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
fig.savefig(f'img/gradientHeatmapThreshold/{modelName}/STE3detail.png', transparent=True)
