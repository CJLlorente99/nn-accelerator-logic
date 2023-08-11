import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# modelName = 'binaryVggSmall/binaryVGGSmall_prunedBT6_4'
# modelName = 'binaryVggSmall/binaryVGGSmall_prunedBT8_4'
# modelName = 'binaryVggSmall/binaryVGGSmall_prunedBT10_4'
# modelName = 'binaryVggSmall/binaryVGGSmall_prunedBT12_4'
# modelName = 'binaryVggVerySmall/binaryVGGVerySmall_prunedBT6_4'
# modelName = 'binaryVggVerySmall/binaryVGGVerySmall_prunedBT8_4'
# modelName = 'binaryVggVerySmall/binaryVGGVerySmall_prunedBT10_4'
# modelName = 'binaryVggVerySmall/binaryVGGVerySmall_prunedBT12_4'

if not os.path.exists(f'img/sparsity/{modelName}'):
    os.makedirs(f'img/sparsity/{modelName}')

dfPrunedLayer1 = pd.read_csv(f'data/savedModels/{modelName}_prunedInfo0.csv').to_numpy()
unique, count = np.unique(dfPrunedLayer1, return_counts=True)
count.sort()

fig = plt.figure()
plt.plot(unique, count, color='b')
plt.ylim([count.min()-1, 4097])
plt.ylabel('Number of times used as input')
plt.xlim([0, 8191])
plt.xlabel('Input neuron')
plt.title(f'Layer 1 Sparsity')
fig.savefig(f'img/sparsity/{modelName}/layer1.png', transparent=True)

dfPrunedLayer2 = pd.read_csv(f'data/savedModels/{modelName}_prunedInfo1.csv').to_numpy()
unique, count = np.unique(dfPrunedLayer2, return_counts=True)
count.sort()

fig = plt.figure()
plt.plot(unique, count, color='b')
plt.ylim([count.min()-1, 4097])
plt.ylabel('Number of times used as input')
plt.xlim([0, 4096])
plt.xlabel('Input neuron')
plt.title(f'Layer 2 Sparsity')
fig.savefig(f'img/sparsity/{modelName}/layer2.png', transparent=True)

dfPrunedLayer3 = pd.read_csv(f'data/savedModels/{modelName}_prunedInfo2.csv').to_numpy()
unique, count = np.unique(dfPrunedLayer3, return_counts=True)
count.sort()

fig = plt.figure()
plt.plot(unique, count, color='b')
plt.ylim([count.min()-1, 1001])
plt.ylabel('Number of times used as input')
plt.xlim([0, 4096])
plt.xlabel('Input neuron')
plt.title(f'Layer 3 Sparsity')
fig.savefig(f'img/sparsity/{modelName}/layer3.png', transparent=True)