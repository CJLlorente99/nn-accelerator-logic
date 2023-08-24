import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

for modelName in ['bnn/bnn_prunedBT6_100ep_4096npl', 'bnn/bnn_prunedBT8_100ep_4096npl', 'bnn/bnn_prunedBT10_100ep_4096npl', 'bnn/bnn_prunedBT12_100ep_4096npl']:
    if not os.path.exists(f'img/sparsity/{modelName}'):
        os.makedirs(f'img/sparsity/{modelName}')

    dfPrunedLayer1 = pd.read_csv(f'data/savedModels/{modelName}_prunedInfol1.csv').to_numpy()
    unique, count = np.unique(dfPrunedLayer1, return_counts=True)
    count.sort()

    fig = plt.figure()
    counts, _, _ = plt.hist(4096 - count, color='b', bins=np.arange(-0.5, 21.5), ec='black')
    plt.ylim([0, counts.max()+1])
    plt.ylabel('Frequency')
    new_list = range(-1, 21)
    plt.xticks(new_list, rotation=45)
    plt.xlabel('Number of times used as input')
    plt.title(f'Layer 1 Sparsity')
    plt.tight_layout()
    fig.savefig(f'img/sparsity/{modelName}/layer1.png', transparent=True)

    dfPrunedLayer2 = pd.read_csv(f'data/savedModels/{modelName}_prunedInfol2.csv').to_numpy()
    unique, count = np.unique(dfPrunedLayer2, return_counts=True)
    count.sort()

    fig = plt.figure()
    counts, _, _ = plt.hist(4096 - count, color='b', bins=np.arange(-0.5, 21.5), ec='black')
    plt.ylim([0, counts.max()+1])
    plt.ylabel('Frequency')
    new_list = range(-1, 21)
    plt.xticks(new_list, rotation=45)
    plt.xlabel('Number of times used as input')
    plt.title(f'Layer 2 Sparsity')
    plt.tight_layout()
    fig.savefig(f'img/sparsity/{modelName}/layer2.png', transparent=True)

    dfPrunedLayer3 = pd.read_csv(f'data/savedModels/{modelName}_prunedInfol3.csv').to_numpy()
    unique, count = np.unique(dfPrunedLayer3, return_counts=True)
    count.sort()

    fig = plt.figure()
    counts, _, _ = plt.hist(4096 - count, color='b', bins=np.arange(-0.5, 21.5), ec='black')
    plt.ylim([0, counts.max()+1])
    plt.ylabel('Frequency')
    new_list = range(-1, 21)
    plt.xticks(new_list, rotation=45)
    plt.xlabel('Number of times used as input')
    plt.title(f'Layer 3 Sparsity')
    plt.tight_layout()
    fig.savefig(f'img/sparsity/{modelName}/layer3.png', transparent=True)