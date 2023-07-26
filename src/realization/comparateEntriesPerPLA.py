import os
import pandas as pd
import matplotlib.pyplot as plt

dataFolder = 'data'
modelName = 'eeb_prunedIrregular20_100ep_100npl'

ttFolderName = f'{dataFolder}/layersTT/{modelName}/notOptimized'
ttFolderNamePerEntry = f'{dataFolder}/layersTT/{modelName}/optimizedPerEntry'
ttFolderNamePerClass = f'{dataFolder}/layersTT/{modelName}/optimizedPerClass'

ttSize = {}

# Care about the output folder
dirs = os.scandir(ttFolderName)

width = {}

for dir in dirs:
    files = os.scandir(f'{ttFolderName}/{dir.name}')
    ttSize[dir.name] = {}
    for file in files: 
        ttSize[dir.name][file.name] = {}
        df = pd.read_csv(f'{ttFolderName}/{dir.name}/{file.name}', index_col=False)
        df = df.astype('int')
        df.drop_duplicates(inplace=True)
        ttSize[dir.name][file.name]['notOptimized'] = df.shape[0]
        width[dir.name] = df.shape[1]
        print(f'{ttFolderName}/{dir.name}/{file.name}')

# Care about the output folder
dirs = os.scandir(ttFolderNamePerEntry)

for dir in dirs:
    files = os.scandir(f'{ttFolderNamePerEntry}/{dir.name}')
    for file in files:
        df = pd.read_csv(f'{ttFolderNamePerEntry}/{dir.name}/{file.name}', index_col=False)
        df = df.astype('int')
        df.drop_duplicates(inplace=True)
        ttSize[dir.name][file.name]['optimizedPerEntry'] = df.shape[0]
        width[dir.name] = df.shape[1]
        print(f'{ttFolderNamePerEntry}/{dir.name}/{file.name}')

# Care about the output folder
dirs = os.scandir(ttFolderNamePerClass)

for dir in dirs:
    files = os.scandir(f'{ttFolderNamePerClass}/{dir.name}')
    for file in files:
        df = pd.read_csv(f'{ttFolderNamePerClass}/{dir.name}/{file.name}', index_col=False)
        df = df.astype('int')
        df.drop_duplicates(inplace=True)
        ttSize[dir.name][file.name]['optimizedPerClass'] = df.shape[0]
        width[dir.name] = df.shape[1]
        print(f'{ttFolderNamePerClass}/{dir.name}/{file.name}')

# Create folder for images
if not os.path.exists(f'img/plaSizeComparative/{modelName}'):
    os.makedirs(f'img/plaSizeComparative/{modelName}')

for layer in ttSize:
    df = pd.DataFrame.from_dict(ttSize[layer]).transpose()

    # Number of entries as substraction from notOptimized
    df['optimizedPerEntryGain'] = df['notOptimized'] - df['optimizedPerEntry']
    df['optimizedPerClassGain'] = df['notOptimized'] - df['optimizedPerClass']

    # Number of entries as percentage saved from notOptimized
    df['optimizedPerEntryPer'] = (df['notOptimized'] - df['optimizedPerEntry'])/df['notOptimized'] * 100
    df['optimizedPerClassPer'] = (df['notOptimized'] - df['optimizedPerClass'])/df['notOptimized'] * 100

    # Global performance (number of entries spared)
    optimizedPerEntryEntriesSparedAbs = (df['optimizedPerEntryGain'] * width[layer]).sum()
    optimizedPerClassEntriesSparedAbs = (df['optimizedPerClassGain'] * width[layer]).sum()
    originalNumberEntries = (df['notOptimized'] * width[layer]).sum()

    optimizedPerEntryEntriesSparedPer = optimizedPerEntryEntriesSparedAbs/originalNumberEntries * 100
    optimizedPerClassEntriesSparedPer = optimizedPerClassEntriesSparedAbs/originalNumberEntries * 100

    infoStr = []
    infoStr.append(f'Original number of entries: {originalNumberEntries}')
    infoStr.append(f'Entries spared following optimization per entry: {optimizedPerEntryEntriesSparedAbs} ({optimizedPerEntryEntriesSparedPer:.2f}%)')
    infoStr.append(f'Entries spared following optimization per class: {optimizedPerClassEntriesSparedAbs} ({optimizedPerClassEntriesSparedPer:.2f}%)')
    infoStr = '\n'.join(infoStr)
    with open(f'img/plaSizeComparative/{modelName}/entriesSpared{layer}.txt', 'w') as f:
        f.write(infoStr)
        f.close()

    
    fig = plt.figure()
    df_sorted = df.sort_values('optimizedPerEntryGain')
    plt.bar(df.index, df_sorted['optimizedPerEntryGain'], color='b')
    plt.title(f'Entries spared per Neuron in {layer} (PE)')
    plt.xlabel('Neuron')
    plt.ylabel('Entries spared')
    plt.xticks([])
    plt.savefig(f'img/plaSizeComparative/{modelName}/absEntriesSparedPE{layer}.png')

    fig = plt.figure()
    df_sorted = df.sort_values('optimizedPerEntryPer')
    plt.bar(df.index, df_sorted['optimizedPerEntryPer'], color='b')
    plt.title(f'% Entries spared per Neuron in {layer} (PE)')
    plt.xlabel('Neuron')
    plt.ylabel(f'% Entries spared')
    plt.xticks([])
    plt.savefig(f'img/plaSizeComparative/{modelName}/perEntriesSparedPE{layer}.png')

    fig = plt.figure()
    df_sorted = df.sort_values('optimizedPerClassGain')
    plt.bar(df.index, df_sorted['optimizedPerClassGain'], color='b')
    plt.title(f'Entries spared per Neuron in {layer} (PC))')
    plt.xlabel('Neuron')
    plt.ylabel(f'Entries spared')
    plt.xticks([])
    plt.savefig(f'img/plaSizeComparative/{modelName}/absEntriesSparedPC{layer}.png')

    fig = plt.figure()
    df_sorted = df.sort_values('optimizedPerClassPer')
    plt.bar(df.index, df_sorted['optimizedPerClassPer'], color='b')
    plt.title(f'% Entries spared per Neuron in {layer} (PC)')
    plt.xlabel('Neuron')
    plt.ylabel(f'% Entries spared')
    plt.xticks([])
    plt.savefig(f'img/plaSizeComparative/{modelName}/perEntriesSparedPC{layer}.png')
