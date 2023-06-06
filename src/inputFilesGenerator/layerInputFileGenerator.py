import pandas as pd
import numpy as np
from ttUtilities.auxFunctions import integerToBinaryArray, binaryArrayToSingleValue

ttFilename = 'data/layersTT/layer1_MNISTSignbinNN100Epoch100NPLnllCriterion'
nNeurons = 100

df = pd.read_feather(ttFilename)

outputTags = [col for col in df if col.startswith('output')]
activationTags = [col for col in df if col.startswith('activation')]
lengthActivationTags = [col for col in df if col.startswith('lengthActivation')]
discriminationTags = [col for col in df if col.startswith('discriminator')]
lengthDiscriminationTags = [col for col in df if col.startswith('lengthDiscriminator')]

def toBinaryArrayActivations(row):
    return integerToBinaryArray(row[activationTags], row[lengthActivationTags])

tag = [f'N{i}' for i in range(nNeurons)]

print(f'Unrolling to binary array')
setOnOff = df.drop(outputTags + discriminationTags + lengthDiscriminationTags, axis=1).apply(toBinaryArrayActivations, axis=1)
setOnOff = np.array([np.array(i) for i in setOnOff])
setOnOff = pd.DataFrame(setOnOff, columns=tag).astype('uint8')
print(f'Unrolled to binary array')

with open(f'data/inputs/layer1', 'w') as f:
    for index, row in setOnOff.iterrows():
        text = ''.join(row.to_string(header=False, index=False).split('\n'))
        f.write(f'{text}\n')
    