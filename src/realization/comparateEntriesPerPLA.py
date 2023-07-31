import os
import pandas as pd
import matplotlib.pyplot as plt

dataFolder = 'data'
modelName = 'eeb/eeb_prunedBT10_100ep_100npl'

ttFolderName = f'{dataFolder}/layersTT/{modelName}/notOptimized'
ttFolderNamePerEntry = f'{dataFolder}/layersTT/{modelName}/optimizedPerEntry'
ttFolderNamePerClass = f'{dataFolder}/layersTT/{modelName}/optimizedPerClass'
plasFolderName = f'{dataFolder}/plas/{modelName}/ESPRESSO'
plasFolderNamePerEntry = f'{dataFolder}/plas/{modelName}/ESPRESSOOptimizedPerEntry_0'
# plasFolderNamePerEntry = f'{dataFolder}/plas/{modelName}/ESPRESSOOptimizedPerEntry_2'
plasFolderNamePerClass = f'{dataFolder}/plas/{modelName}/ESPRESSOOptimizedPerClass'
pruned = True
prunedBaseFilename = f'{dataFolder}/savedModels/{modelName}_prunedInfol'

def main(prun):
    ttSize = {}
    width = {}

    def fillLayersTT(folderName, key):
        dirs = os.scandir(folderName)
        i = 1
        for dir in dirs:
            files = os.scandir(f'{folderName}/{dir.name}')
            if not dir.name in ttSize.keys():
                ttSize[dir.name] = {}
            if prun:
                dfPruned = pd.read_csv(f'{prunedBaseFilename}{i}.csv')
            i += 1
            for file in files:
                if not file.name in ttSize[dir.name]: 
                    ttSize[dir.name][file.name] = {}
                df = pd.read_feather(f'{folderName}/{dir.name}/{file.name}')
                df = df.astype('int')
                df.drop_duplicates(inplace=True)
                if prun:
                    df.drop(df.columns[dfPruned[file.name].values], axis=1, inplace=True)
                    df.drop_duplicates(inplace=True)
                ttSize[dir.name][file.name][key] = df.shape[0]
                width[dir.name] = df.shape[1] - 1
                print(f'{folderName}/{dir.name}/{file.name}')

    fillLayersTT(ttFolderName, 'notOptimized')
    fillLayersTT(ttFolderNamePerClass, 'optimizedPerClass')
    fillLayersTT(ttFolderNamePerEntry, 'optimizedPerEntry')

    def fillPLAESPRESSO(folderName, key):
        for layer in [f'layer{i}' for i in range(1, 4)]:
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

    if prun:
        fillPLAESPRESSO(plasFolderName, 'notOptimizedESPRESSO')
        fillPLAESPRESSO(plasFolderNamePerEntry, 'optimizedPerEntryESPRESSO')
        fillPLAESPRESSO(plasFolderNamePerClass, 'optimizedPerClassESPRESSO')

    for layer in ttSize:
        df = pd.DataFrame.from_dict(ttSize[layer]).transpose()

        # Number of entries as substraction from notOptimized
        df['optimizedPerEntryGain'] = df['notOptimized'] - df['optimizedPerEntry']
        df['optimizedPerClassGain'] = df['notOptimized'] - df['optimizedPerClass']

        # Number of entries as percentage saved from notOptimized
        df['optimizedPerEntryPer'] = (df['notOptimized'] - df['optimizedPerEntry'])/df['notOptimized'] * 100
        df['optimizedPerClassPer'] = (df['notOptimized'] - df['optimizedPerClass'])/df['notOptimized'] * 100

        # Global performance (number of entries spared)
        optimizedPerEntryEntriesSparedAbs = df['optimizedPerEntryGain'].sum()
        optimizedPerClassEntriesSparedAbs = df['optimizedPerClassGain'].sum()
        originalNumberEntries = df['notOptimized'].sum()

        optimizedPerEntryEntriesSparedPer = optimizedPerEntryEntriesSparedAbs/originalNumberEntries * 100
        optimizedPerClassEntriesSparedPer = optimizedPerClassEntriesSparedAbs/originalNumberEntries * 100

        infoStr = []
        infoStr.append(f'Original number of entries: {originalNumberEntries}')
        infoStr.append(f'Entries spared following optimization per entry: {optimizedPerEntryEntriesSparedAbs} ({optimizedPerEntryEntriesSparedPer:.2f}%)')
        infoStr.append(f'Entries spared following optimization per class: {optimizedPerClassEntriesSparedAbs} ({optimizedPerClassEntriesSparedPer:.2f}%)')
        infoStr = '\n'.join(infoStr)
        fname = f'img/plaSizeComparative/{modelName}/beforePrune/entriesSpared{layer}.txt' if not prun else f'img/plaSizeComparative/{modelName}/afterPrune/entriesSpared{layer}.txt'
        with open(fname, 'w') as f:
            f.write(infoStr)
            f.close()

        
        fig = plt.figure()
        df_sorted = df.sort_values('optimizedPerEntryGain')
        plt.bar(df.index, df_sorted['optimizedPerEntryGain'], color='b')
        plt.title(f'Entries spared per Neuron in {layer} (PE)')
        plt.xlabel('Neuron')
        plt.ylabel('Entries spared')
        plt.xticks([])
        fname = f'img/plaSizeComparative/{modelName}/beforePrune/absEntriesSparedPE{layer}.png' if not prun else f'img/plaSizeComparative/{modelName}/afterPrune/absEntriesSparedPE{layer}.png'
        plt.savefig(fname)
        plt.close()

        fig = plt.figure()
        df_sorted = df.sort_values('optimizedPerEntryPer')
        plt.bar(df.index, df_sorted['optimizedPerEntryPer'], color='b')
        plt.title(f'% Entries spared per Neuron in {layer} (PE)')
        plt.xlabel('Neuron')
        plt.ylabel(f'% Entries spared')
        plt.xticks([])
        fname = f'img/plaSizeComparative/{modelName}/beforePrune/perEntriesSparedPE{layer}.png' if not prun else f'img/plaSizeComparative/{modelName}/afterPrune/perEntriesSparedPE{layer}.png'
        plt.savefig(fname)
        plt.close()

        fig = plt.figure()
        df_sorted = df.sort_values('optimizedPerClassGain')
        plt.bar(df.index, df_sorted['optimizedPerClassGain'], color='b')
        plt.title(f'Entries spared per Neuron in {layer} (PC))')
        plt.xlabel('Neuron')
        plt.ylabel(f'Entries spared')
        plt.xticks([])
        fname = f'img/plaSizeComparative/{modelName}/beforePrune/absEntriesSparedPC{layer}.png' if not prun else f'img/plaSizeComparative/{modelName}/afterPrune/absEntriesSparedPC{layer}.png'
        plt.savefig(fname)
        plt.close()

        fig = plt.figure()
        df_sorted = df.sort_values('optimizedPerClassPer')
        plt.bar(df.index, df_sorted['optimizedPerClassPer'], color='b')
        plt.title(f'% Entries spared per Neuron in {layer} (PC)')
        plt.xlabel('Neuron')
        plt.ylabel(f'% Entries spared')
        plt.xticks([])
        fname = f'img/plaSizeComparative/{modelName}/beforePrune/perEntriesSparedPC{layer}.png' if not prun else f'img/plaSizeComparative/{modelName}/afterPrune/perEntriesSparedPC{layer}.png'
        plt.savefig(fname)
        plt.close()
        dataFname = f'img/plaSizeComparative/{modelName}/beforePrune/data{layer}.csv' if not prun else f'img/plaSizeComparative/{modelName}/afterPrune/data{layer}.csv'
        df.to_csv(dataFname)

        if prun: # Info between ESPRESSO and NotESPRESSOed
            # Number of entries as substraction from espresso version
            df['notOptimizedESPRESSOGain'] = df['notOptimized'] - df['notOptimizedESPRESSO']
            df['optimizedPerEntryESPRESSOGain'] = df['notOptimized'] - df['optimizedPerEntryESPRESSO']
            df['optimizedPerClassESPRESSOGain'] = df['notOptimized'] - df['optimizedPerClassESPRESSO']

            # Number of entries as percentage saved from notOptimized
            df['notOptimizedESPRESSOPer'] = (df['notOptimized'] - df['notOptimizedESPRESSO']) / df['notOptimized'] * 100
            df['optimizedPerEntryESPRESSOPer'] = (df['notOptimized'] - df['optimizedPerEntryESPRESSO']) / df['notOptimized'] * 100
            df['optimizedPerClassESPRESSOPer'] = (df['notOptimized'] - df['optimizedPerClassESPRESSO']) / df['notOptimized'] * 100

            # Global performance (number of entries spared)
            notOptimizedEntriesSparedAbs = df['notOptimizedESPRESSOGain'].sum()
            optimizedPerEntryEntriesSparedAbs = df['optimizedPerEntryESPRESSOGain'].sum()
            optimizedPerClassEntriesSparedAbs = df['optimizedPerClassESPRESSOGain'].sum()
            originalNumberEntries = df['notOptimized'].sum()

            notOptimizedEntriesSparedPer = notOptimizedEntriesSparedAbs/originalNumberEntries * 100
            optimizedPerEntryEntriesSparedPer = optimizedPerEntryEntriesSparedAbs/originalNumberEntries * 100
            optimizedPerClassEntriesSparedPer = optimizedPerClassEntriesSparedAbs/originalNumberEntries * 100

            infoStr = []
            infoStr.append(f'Original number of entries: {originalNumberEntries}')
            infoStr.append(f'Entries spared following No Opt: {notOptimizedEntriesSparedAbs} ({notOptimizedEntriesSparedPer:.2f}%)')
            infoStr.append(f'Entries spared following optimization per entry: {optimizedPerEntryEntriesSparedAbs} ({optimizedPerEntryEntriesSparedPer:.2f}%)')
            infoStr.append(f'Entries spared following optimization per class: {optimizedPerClassEntriesSparedAbs} ({optimizedPerClassEntriesSparedPer:.2f}%)')
            infoStr = '\n'.join(infoStr)
            fname = f'img/plaSizeComparative/{modelName}/afterPruneESPRESSO/entriesSpared{layer}.txt'
            with open(fname, 'w') as f:
                f.write(infoStr)
                f.close()

            fig = plt.figure()
            df_sorted = df.sort_values('notOptimizedESPRESSOGain')
            plt.bar(df.index, df_sorted['notOptimizedESPRESSOGain'], color='b')
            plt.title(f'Entries spared per Neuron in {layer} (no Opt.)')
            plt.xlabel('Neuron')
            plt.ylabel('Entries spared')
            plt.xticks([])
            fname = f'img/plaSizeComparative/{modelName}/afterPruneESPRESSO/absEntriesSparedNO{layer}.png'
            plt.savefig(fname)
            plt.close()

            fig = plt.figure()
            df_sorted = df.sort_values('notOptimizedESPRESSOPer')
            plt.bar(df.index, df_sorted['notOptimizedESPRESSOPer'], color='b')
            plt.title(f'% Entries spared per Neuron in {layer} (no Opt.)')
            plt.xlabel('Neuron')
            plt.ylabel(f'% Entries spared')
            plt.xticks([])
            fname = f'img/plaSizeComparative/{modelName}/afterPruneESPRESSO/perEntriesSparedNO{layer}.png'
            plt.savefig(fname)
            plt.close()
            
            fig = plt.figure()
            df_sorted = df.sort_values('optimizedPerEntryESPRESSOGain')
            plt.bar(df.index, df_sorted['optimizedPerEntryESPRESSOGain'], color='b')
            plt.title(f'Entries spared per Neuron in {layer} (PE)')
            plt.xlabel('Neuron')
            plt.ylabel('Entries spared')
            plt.xticks([])
            fname = f'img/plaSizeComparative/{modelName}/afterPruneESPRESSO/absEntriesSparedPE{layer}.png'
            plt.savefig(fname)
            plt.close()

            fig = plt.figure()
            df_sorted = df.sort_values('optimizedPerEntryESPRESSOPer')
            plt.bar(df.index, df_sorted['optimizedPerEntryESPRESSOPer'], color='b')
            plt.title(f'% Entries spared per Neuron in {layer} (PE)')
            plt.xlabel('Neuron')
            plt.ylabel(f'% Entries spared')
            plt.xticks([])
            fname = f'img/plaSizeComparative/{modelName}/afterPruneESPRESSO/perEntriesSparedPE{layer}.png'
            plt.savefig(fname)
            plt.close()

            fig = plt.figure()
            df_sorted = df.sort_values('optimizedPerClassESPRESSOGain')
            plt.bar(df.index, df_sorted['optimizedPerClassESPRESSOGain'], color='b')
            plt.title(f'Entries spared per Neuron in {layer} (PC))')
            plt.xlabel('Neuron')
            plt.ylabel(f'Entries spared')
            plt.xticks([])
            fname = f'img/plaSizeComparative/{modelName}/afterPruneESPRESSO/absEntriesSparedPC{layer}.png'
            plt.savefig(fname)
            plt.close()

            fig = plt.figure()
            df_sorted = df.sort_values('optimizedPerClassESPRESSOPer')
            plt.bar(df.index, df_sorted['optimizedPerClassESPRESSOPer'], color='b')
            plt.title(f'% Entries spared per Neuron in {layer} (PC)')
            plt.xlabel('Neuron')
            plt.ylabel(f'% Entries spared')
            plt.xticks([])
            fname = f'img/plaSizeComparative/{modelName}/afterPruneESPRESSO/perEntriesSparedPC{layer}.png'
            plt.savefig(fname)
            plt.close()
            dataFname = f'img/plaSizeComparative/{modelName}/afterPruneESPRESSO/data{layer}.csv'
            df.to_csv(dataFname)

if __name__ == '__main__':
    # Create folder for images
    if not os.path.exists(f'img/plaSizeComparative/{modelName}'):
        if pruned:
            os.makedirs(f'img/plaSizeComparative/{modelName}/afterPrune')
            os.makedirs(f'img/plaSizeComparative/{modelName}/afterPruneESPRESSO')
        os.makedirs(f'img/plaSizeComparative/{modelName}/beforePrune')

    df = main(False)
    if pruned:
        df = main(True)
