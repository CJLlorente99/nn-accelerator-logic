import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

dataFolder = 'data'

for modelName in ['eeb/eeb_prunedBT12_100ep_100npl_all']:
    stringData = re.search(f'BT(.*)_100ep', modelName)
    maxTTSize = 2**int(stringData.group(1))
    if not os.path.exists(f'img/plaSizeComparative/{modelName}/afterPruneESPRESSO'):
        os.makedirs(f'img/plaSizeComparative/{modelName}/afterPruneESPRESSO')

    plasFolderNameABC = f'{dataFolder}/plas/{modelName}/ABC'
    plasFolderName = f'{dataFolder}/plas/{modelName}/ESPRESSO'
    plasFolderNamePerEntry_0 = f'{dataFolder}/plas/{modelName}/ESPRESSOOptimizedPerEntry_0'
    plasFolderNamePerEntry_2 = f'{dataFolder}/plas/{modelName}/ESPRESSOOptimizedPerEntry_2'
    pruned = True
    prunedBaseFilename = f'{dataFolder}/savedModels/{modelName}_prunedInfol'
    ttSize = {}
    width = {}

    def fillPLAABC(folderName, key):
        for layer in [f'layer{i}' for i in range(4)]:
            files = os.scandir(f'{folderName}/{layer}')
            if not layer in ttSize.keys():
                ttSize[layer] = {}
            for file in files:
                if not file.name[:-4]  in ttSize[layer].keys():
                    ttSize[layer][file.name[:-4]] = {}
                f = open(f'{folderName}/{layer}/{file.name}', 'r')
                if not layer in width.keys():
                    width[layer] = f.readline()[0][3:-1] # Strip ".i " and "\n"
                    f.seek(0)
                ttSize[layer][file.name[:-4]][key] = len(f.readlines()) - 4  # 4 are the extra lines in ABC_PLA
                print(f'{folderName}/{layer}/{file.name}')

    fillPLAABC(plasFolderNameABC, 'notOptimized')

    def fillPLAESPRESSO(folderName, key):
        for layer in [f'layer{i}' for i in range(4)]:
            files = os.scandir(f'{folderName}/{layer}_espresso')
            if not layer in ttSize.keys():
                ttSize[layer] = {}
            for file in files:
                if not file.name[:-4]  in ttSize[layer].keys():
                    ttSize[layer][file.name[:-4]] = {}
                f = open(f'{folderName}/{layer}_espresso/{file.name}', 'r')
                if not layer in width.keys():
                    width[layer] = f.readline()[1][3:-1] # Strip ".i " and "\n"
                    f.seek(0)
                ttSize[layer][file.name[:-4]][key] = len(f.readlines()) - 7  # 7 are the extra lines in ESPRESSO_PLA
                print(f'{folderName}/{layer}_espresso/{file.name}')

    fillPLAESPRESSO(plasFolderName, 'notOptimizedESPRESSO')
    fillPLAESPRESSO(plasFolderNamePerEntry_0, 'optimizedPerEntryESPRESSO_0')
    fillPLAESPRESSO(plasFolderNamePerEntry_2, 'optimizedPerEntryESPRESSO_2')

    for layer in ttSize:
        df = pd.DataFrame.from_dict(ttSize[layer]).transpose()

        # Number of entries as substraction from espresso version
        df['notOptimizedESPRESSOGain'] = df['notOptimized'] - df['notOptimizedESPRESSO']
        df['optimizedPerEntryESPRESSO_0Gain'] = df['notOptimized'] - df['optimizedPerEntryESPRESSO_0']
        df['optimizedPerEntryESPRESSO_2Gain'] = df['notOptimized'] - df['optimizedPerEntryESPRESSO_2']

        # Number of entries as percentage saved from notOptimized
        df['notOptimizedESPRESSOPer'] = (df['notOptimized'] - df['notOptimizedESPRESSO']) / df['notOptimized'] * 100
        df['optimizedPerEntryESPRESSO_0Per'] = (df['notOptimized'] - df['optimizedPerEntryESPRESSO_0']) / df['notOptimized'] * 100
        df['optimizedPerEntryESPRESSO_2Per'] = (df['notOptimized'] - df['optimizedPerEntryESPRESSO_2']) / df['notOptimized'] * 100

        # Global performance (number of entries spared)
        notOptimizedEntriesSparedAbs = df['notOptimizedESPRESSOGain'].sum()
        optimizedPerEntry_0EntriesSparedAbs = df['optimizedPerEntryESPRESSO_0Gain'].sum()
        optimizedPerEntry_2EntriesSparedAbs = df['optimizedPerEntryESPRESSO_2Gain'].sum()
        originalNumberEntries = df['notOptimized'].sum()

        notOptimizedEntriesSparedPer = notOptimizedEntriesSparedAbs/originalNumberEntries * 100
        optimizedPerEntry_0EntriesSparedPer = optimizedPerEntry_0EntriesSparedAbs/originalNumberEntries * 100
        optimizedPerEntry_2EntriesSparedPer = optimizedPerEntry_2EntriesSparedAbs/originalNumberEntries * 100

        infoStr = []
        infoStr.append(f'Original number of entries: {originalNumberEntries}')
        infoStr.append(f'Entries spared following No Opt: {notOptimizedEntriesSparedAbs} ({notOptimizedEntriesSparedPer:.2f}%)')
        infoStr.append(f'Entries spared following optimization per entry (0): {optimizedPerEntry_0EntriesSparedAbs} ({optimizedPerEntry_0EntriesSparedPer:.2f}%)')
        infoStr.append(f'Entries spared following optimization per entry (2): {optimizedPerEntry_2EntriesSparedAbs} ({optimizedPerEntry_2EntriesSparedPer:.2f}%)')
        infoStr = '\n'.join(infoStr)
        fname = f'img/plaSizeComparative/{modelName}/afterPruneESPRESSO/entriesSpared{layer}.txt'
        with open(fname, 'w') as f:
            f.write(infoStr)
            f.close()

        fig = plt.figure()
        df_sorted = df.sort_values('notOptimizedESPRESSOGain')
        plt.hist(df_sorted['notOptimizedESPRESSOGain'], color='b', ec='black')
        plt.title(f'Entries spared per Neuron in {layer} (no Opt.)')
        plt.xlabel('Entries spared')
        plt.xticks(np.arange(0, maxTTSize+1, maxTTSize/10).astype(int), rotation=45)
        plt.ylabel('Frequency')
        fname = f'img/plaSizeComparative/{modelName}/afterPruneESPRESSO/absEntriesSparedNO{layer}.png'
        plt.savefig(fname)
        plt.close()

        fig = plt.figure()
        df_sorted = df.sort_values('notOptimizedESPRESSOPer')
        plt.hist(df_sorted['notOptimizedESPRESSOPer'], color='b', ec='black')
        plt.title(f'% Entries spared per Neuron in {layer} (no Opt.)')
        plt.xlabel(f'% Entries spared')
        plt.ylabel('Frequency')
        plt.xticks(np.arange(0, 110, 10))
        fname = f'img/plaSizeComparative/{modelName}/afterPruneESPRESSO/perEntriesSparedNO{layer}.png'
        plt.savefig(fname)
        plt.close()
        
        fig = plt.figure()
        df_sorted = df.sort_values('optimizedPerEntryESPRESSO_0Gain')
        plt.hist(df_sorted['optimizedPerEntryESPRESSO_0Gain'], color='b', ec='black')
        plt.title(f'Entries spared per Neuron in {layer} (PE_0)')
        plt.xlabel('Entries spared')
        plt.ylabel('Frequency')
        plt.xticks(np.arange(0, maxTTSize+1, maxTTSize/10).astype(int), rotation=45)
        fname = f'img/plaSizeComparative/{modelName}/afterPruneESPRESSO/absEntriesSparedPE_0{layer}.png'
        plt.savefig(fname)
        plt.close()

        fig = plt.figure()
        df_sorted = df.sort_values('optimizedPerEntryESPRESSO_0Per')
        plt.hist(df_sorted['optimizedPerEntryESPRESSO_0Per'], color='b', ec='black')
        plt.title(f'% Entries spared per Neuron in {layer} (PE_0)')
        plt.xlabel(f'% Entries spared')
        plt.ylabel('Frequency')
        plt.xticks(np.arange(0, 110, 10))
        fname = f'img/plaSizeComparative/{modelName}/afterPruneESPRESSO/perEntriesSparedPE_0{layer}.png'
        plt.savefig(fname)
        plt.close()

        fig = plt.figure()
        df_sorted = df.sort_values('optimizedPerEntryESPRESSO_2Gain')
        plt.hist(df_sorted['optimizedPerEntryESPRESSO_2Gain'], color='b', ec='black')
        plt.title(f'Entries spared per Neuron in {layer} (PE_2)')
        plt.xlabel('Entries spared')
        plt.ylabel('Frequency')
        plt.xticks(np.arange(0, maxTTSize+1, maxTTSize/10).astype(int), rotation=45)
        fname = f'img/plaSizeComparative/{modelName}/afterPruneESPRESSO/absEntriesSparedPE_2{layer}.png'
        plt.savefig(fname)
        plt.close()

        fig = plt.figure()
        df_sorted = df.sort_values('optimizedPerEntryESPRESSO_2Per')
        plt.hist(df_sorted['optimizedPerEntryESPRESSO_2Per'], color='b', ec='black')
        plt.title(f'% Entries spared per Neuron in {layer} (PE_2)')
        plt.xlabel(f'% Entries spared')
        plt.ylabel('Frequency')
        plt.xticks(np.arange(0, 110, 10))
        fname = f'img/plaSizeComparative/{modelName}/afterPruneESPRESSO/perEntriesSparedPE_2{layer}.png'
        plt.savefig(fname)
        plt.close()
        dataFname = f'img/plaSizeComparative/{modelName}/afterPruneESPRESSO/data{layer}.csv'
        df.to_csv(dataFname)
