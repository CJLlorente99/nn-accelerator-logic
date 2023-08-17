import matplotlib.pyplot as plt
import matplotlib
from modelsCommon.auxTransformations import ToBlackAndWhite, ToSign
from torchvision.transforms import ToTensor, Compose
from torchvision import datasets
import pandas as pd
import numpy as np
import os

vmin = 10e-6
vmax = 10e-1
threshold = 10e-5

for modelName in ['bnn/bnn_prunedBT6_100ep_4096npl', 'bnn/bnn_prunedBT8_100ep_4096npl', 'bnn/bnn_prunedBT10_100ep_4096npl', 'bnn/bnn_prunedBT12_100ep_4096npl']:

    print(f'{modelName}')

    ste1GradientFilename = f'data/gradients/{modelName}/STE1'
    ste2GradientFilename = f'data/gradients/{modelName}/STE2'
    ste3GradientFilename = f'data/gradients/{modelName}/STE3'

    if not os.path.exists(f'img/gradientHeatmapThreshold/{modelName}'):
        os.makedirs(f'img/gradientHeatmapThreshold/{modelName}')

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

    xTicksLocations = np.arange(len(targetsDf)/20, len(targetsDf), len(targetsDf)/10)
    xTicksLabels = [f'Class{i}' for i in range(10)]

    vLines = np.cumsum(targetsDf.value_counts().sort_index().values)
    xLines = np.arange(0.5, 20.5)

    '''
    Plot STE1
    '''

    df = pd.read_feather(ste1GradientFilename)
    df[df.abs() < threshold] = 0
    df = df.reindex(targetsDf.index)
    aux = np.abs(df.to_numpy().T)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(aux, cmap='hot', interpolation='nearest', norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
    plt.vlines(vLines, ymin=-0.5, ymax=4095.5, color='blue')
    plt.xlabel('MNIST train images, grouped by category')
    plt.ylabel('Neuron index')
    plt.colorbar()
    plt.title(f'Layer 1 Gradients per Training Sample and Neuron')
    ax.set_xticks(xTicksLocations, xTicksLabels)
    plt.xticks(rotation=45)
    plt.ylim((-0.5, 4095.5))
    forceAspect(ax,aspect=1)
    plt.tight_layout()
    fig.savefig(f'img/gradientHeatmapThreshold/{modelName}/STE1.png', transparent=True)
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(aux[:20,:], cmap='hot', interpolation='nearest', norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
    plt.vlines(vLines, ymin=-0.5, ymax=19.5, color='blue')
    plt.hlines(xLines, xmin=0, xmax=len(targetsDf), color='blue')
    plt.xlabel('MNIST train images, grouped by category')
    plt.ylabel('Neuron index')
    plt.colorbar()
    plt.title(f'Layer 1 Gradients per Training Sample and Neuron (20 Neurons)')
    ax.set_xticks(xTicksLocations, xTicksLabels)
    ax.set_yticks(np.arange(0, 20, 1))
    plt.xticks(rotation=45)
    plt.ylim((-0.5, 19.5))
    forceAspect(ax,aspect=1)
    plt.tight_layout()
    fig.savefig(f'img/gradientHeatmapThreshold/{modelName}/STE1detail.png', transparent=True)
    plt.close(fig)

    '''
    Plot STE2
    '''

    df = pd.read_feather(ste2GradientFilename)
    df[df.abs() < threshold] = 0
    df = df.reindex(targetsDf.index)
    aux = np.abs(df.to_numpy().T)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(aux, cmap='hot', interpolation='nearest', norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
    plt.vlines(vLines, ymin=-0.5, ymax=4095.5, color='blue')
    plt.xlabel('MNIST train images, grouped by category')
    plt.ylabel('Neuron index')
    plt.colorbar()
    plt.title(f'Layer 2 Gradients per Training Sample and Neuron')
    forceAspect(ax,aspect=1)
    ax.set_xticks(xTicksLocations, xTicksLabels)
    plt.xticks(rotation=45)
    plt.ylim((-0.5, 4095.5))
    plt.tight_layout()
    fig.savefig(f'img/gradientHeatmapThreshold/{modelName}/STE2.png', transparent=True)
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(aux[:20,:], cmap='hot', interpolation='nearest', norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
    plt.vlines(vLines, ymin=-0.5, ymax=19.5, color='blue')
    plt.hlines(xLines, xmin=0, xmax=len(targetsDf), color='blue')
    plt.xlabel('MNIST train images, grouped by category')
    plt.ylabel('Neuron index')
    plt.colorbar()
    plt.title(f'Layer 2 Gradients per Training Sample and Neuron (20 Neurons)')
    forceAspect(ax,aspect=1)
    ax.set_xticks(xTicksLocations, xTicksLabels)
    ax.set_yticks(np.arange(0, 20, 1))
    plt.xticks(rotation=45)
    plt.ylim((-0.5, 19.5))
    plt.tight_layout()
    fig.savefig(f'img/gradientHeatmapThreshold/{modelName}/STE2detail.png', transparent=True)
    plt.close(fig)

    '''
    Plot STE3
    '''

    df = pd.read_feather(ste3GradientFilename)
    df[df.abs() < threshold] = 0
    df = df.reindex(targetsDf.index)
    aux = np.abs(df.to_numpy().T)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(aux, cmap='hot', interpolation='nearest', norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
    plt.vlines(vLines, ymin=-0.5, ymax=4095.5, color='blue')
    plt.xlabel('MNIST train images, grouped by category')
    plt.ylabel('Neuron index')
    plt.colorbar()
    plt.title(f'Layer 3 Gradients per Training Sample and Neuron')
    forceAspect(ax,aspect=1)
    ax.set_xticks(xTicksLocations, xTicksLabels)
    plt.xticks(rotation=45)
    plt.ylim((-0.5, 4095.5))
    plt.tight_layout()
    fig.savefig(f'img/gradientHeatmapThreshold/{modelName}/STE3.png', transparent=True)
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(aux[:20,:], cmap='hot', interpolation='nearest', norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
    plt.vlines(vLines, ymin=-0.5, ymax=19.5, color='blue')
    plt.hlines(xLines, xmin=0, xmax=len(targetsDf), color='blue')
    plt.xlabel('MNIST train images, grouped by category')
    plt.ylabel('Neuron index')
    plt.colorbar()
    plt.title(f'Layer 3 Gradients per Training Sample and Neuron (20 Neurons)')
    forceAspect(ax,aspect=1)
    ax.set_xticks(xTicksLocations, xTicksLabels)
    ax.set_yticks(np.arange(0, 20, 1))
    plt.xticks(rotation=45)
    plt.ylim((-0.5, 19.5))
    plt.tight_layout()
    fig.savefig(f'img/gradientHeatmapThreshold/{modelName}/STE3detail.png', transparent=True)
    plt.close(fig)

    print(f'{modelName} erfolgreich')
