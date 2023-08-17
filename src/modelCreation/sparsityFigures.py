import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

for modelName in ['binaryVggSmall/binaryVGGSmall_prunedBT6_4', 'binaryVggSmall/binaryVGGSmall_prunedBT8_4', 'binaryVggSmall/binaryVGGSmall_prunedBT10_4', 'binaryVggSmall/binaryVGGSmall_prunedBT12_4',
                  'binaryVggVerySmall/binaryVGGVerySmall_prunedBT6_4', 'binaryVggVerySmall/binaryVGGVerySmall_prunedBT8_4',
                  'binaryVggVerySmall/binaryVGGVerySmall_prunedBT10_4', 'binaryVggVerySmall/binaryVGGVerySmall_prunedBT12_4']:
    
    print(f'{modelName}')

    if not os.path.exists(f'img/sparsity/{modelName}'):
        os.makedirs(f'img/sparsity/{modelName}')

    dfPrunedLayer1 = pd.read_csv(f'data/savedModels/{modelName}_prunedInfo0.csv').to_numpy()
    unique, count = np.unique(dfPrunedLayer1, return_counts=True)
    count.sort()

    fig = plt.figure()
    counts, _, _ = plt.hist(count, color='b', bins=np.arange(4070, 4097), ec='black')
    plt.ylim([0, counts.max()+1])
    plt.ylabel('Frequency')
    new_list = range(4070, 4097)
    plt.xticks(new_list, rotation=45)
    plt.xlabel('Number of times used as input')
    plt.title(f'Layer 1 Sparsity')
    plt.tight_layout()
    fig.savefig(f'img/sparsity/{modelName}/layer1.png', transparent=True)
    plt.close(fig)

    dfPrunedLayer2 = pd.read_csv(f'data/savedModels/{modelName}_prunedInfo1.csv').to_numpy()
    unique, count = np.unique(dfPrunedLayer2, return_counts=True)
    count.sort()

    fig = plt.figure()
    counts, _, _ = plt.hist(count, color='b', bins=np.arange(4070, 4097), ec='black')
    plt.ylim([0, counts.max()+1])
    plt.ylabel('Frequency')
    new_list = range(4070, 4097)
    plt.xticks(new_list, rotation=45)
    plt.xlabel('Number of times used as input')
    plt.title(f'Layer 2 Sparsity')
    plt.tight_layout()
    fig.savefig(f'img/sparsity/{modelName}/layer2.png', transparent=True)
    plt.close(fig)

    dfPrunedLayer3 = pd.read_csv(f'data/savedModels/{modelName}_prunedInfo2.csv').to_numpy()
    unique, count = np.unique(dfPrunedLayer3, return_counts=True)
    count.sort()

    fig = plt.figure()
    counts, _, _ = plt.hist(count, color='b', bins=np.arange(970, 1001), ec='black')
    plt.ylim([0, counts.max()+1])
    plt.ylabel('Frequency')
    new_list = range(970, 1001)
    plt.xticks(new_list, rotation=45)
    plt.xlabel('Number of times used as input')
    plt.title(f'Layer 3 Sparsity')
    plt.tight_layout()
    fig.savefig(f'img/sparsity/{modelName}/layer3.png', transparent=True)
    plt.close(fig)

    print(f'{modelName} erfolgreich')