import pandas as pd
import numpy as np
from ttUtilities.auxFunctions import integerToBinaryArray, binaryArrayToSingleValue

layer = '1'

ttFilename = f'data/layersTT/layer{layer}_MNISTSignbinNN100Epoch100NPLnllCriterion'
simulatedFilename = f'L{layer}_'
nNeurons = 100

df = pd.read_feather(ttFilename)

outputTags = [col for col in df if col.startswith('output')]
activationTags = [col for col in df if col.startswith('activation')]
lengthActivationTags = [col for col in df if col.startswith('lengthActivation')]
discriminationTags = [col for col in df if col.startswith('discriminator')]
lengthDiscriminationTags = [col for col in df if col.startswith('lengthDiscriminator')]

df = df[outputTags]
df = df.reindex(sorted(df.columns), axis=1)

correct = 0
with open(simulatedFilename, 'r') as f:
    for index, row in df.iterrows():
        text = ''.join(row.to_string(header=False, index=False).split('\n'))
        if text == f.readline():
            correct += 1
            
print(f'Correct {correct}/{len(df)} {correct/len(df)}')
    