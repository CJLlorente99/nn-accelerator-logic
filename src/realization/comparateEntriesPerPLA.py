import os
import pandas as pd
import matplotlib.pyplot as plt

dataFolder = 'data'
modelName = 'eeb_100ep_100npl'

ttFolderName = f'{dataFolder}/layersTT/{modelName}/notOptimized'
ttFolderNamePerEntry = f'{dataFolder}/layersTT/{modelName}/optimizedPerEntry'
ttFolderNamePerClass = f'{dataFolder}/layersTT/{modelName}/optimizedPerClass'

ttSize = {}

# Care about the output folder
dirs = os.scandir(ttFolderName)

for dir in dirs:
    files = os.scandir(f'{ttFolderName}/{dir.name}')
    ttSize[dir.name] = {}
    for file in files:
        ttSize[dir.name][file.name] = {}
        df = pd.read_csv(f'{ttFolderName}/{dir.name}/{file.name}', index_col=False)
        df = df.astype('int')
        df.drop_duplicates(inplace=True)
        ttSize[dir.name][file.name]['notOptimized'] = df.shape[0]
        print(f'{dir.name}/{file.name}')

# Care about the output folder
dirs = os.scandir(ttFolderNamePerEntry)

for dir in dirs:
    files = os.scandir(f'{ttFolderNamePerEntry}/{dir.name}')
    for file in files:
        df = pd.read_csv(f'{ttFolderNamePerEntry}/{dir.name}/{file.name}', index_col=False)
        df = df.astype('int')
        df.drop_duplicates(inplace=True)
        ttSize[dir.name][file.name]['optimizedPerEntry'] = df.shape[0]
        print(f'{dir.name}/{file.name}')

# Care about the output folder
dirs = os.scandir(ttFolderNamePerClass)

for dir in dirs:
    files = os.scandir(f'{ttFolderNamePerClass}/{dir.name}')
    for file in files:
        df = pd.read_csv(f'{ttFolderNamePerClass}/{dir.name}/{file.name}', index_col=False)
        df = df.astype('int')
        df.drop_duplicates(inplace=True)
        ttSize[dir.name][file.name]['optimizedPerClass'] = df.shape[0]
        print(f'{dir.name}/{file.name}')

for layer in ttSize:
    df = pd.DataFrame.from_dict(ttSize[layer])
    print(df)
    pass