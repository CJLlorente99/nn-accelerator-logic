import re
import pandas as pd
import matplotlib.pyplot as plt
import os

for modelName in ['eeb/eeb_prunedBT6_100ep_100npl', 'eeb/eeb_prunedBT8_100ep_100npl', 'eeb/eeb_prunedBT10_100ep_100npl', 'eeb/eeb_prunedBT12_100ep_100npl']:

    if not os.path.exists(f'img/aigerStats/{modelName}'):
        os.makedirs(f'img/aigerStats/{modelName}')

    aigerStatsFilename = f'data/aiger/{modelName}/aigerStats.txt'

    data = {}

    with open(aigerStatsFilename, 'r') as f:
        while True:
            line = f.readline()

            if not line:
                break

            stringData = re.search(f'{modelName}/(.*)/(.*)/(.*):', line)
            subfolder = stringData.group(1)
            layer = stringData.group(2)
            neuron = stringData.group(3)
            stringData = re.search(f'and = (.*) lev = (.*)\n', line)
            nAnd = int(stringData.group(1))
            nLevel = int(stringData.group(2))
            if not layer in data:
                data[layer] = {}
            if not subfolder in data[layer]:
                data[layer][subfolder] = {}
            if not 'nAnd' in data[layer][subfolder]:
                data[layer][subfolder]['nAnd'] = {}
            if not 'nLevel' in data[layer][subfolder]:
                data[layer][subfolder]['nLevel'] = {}
            
            data[layer][subfolder]['nAnd'][neuron] = nAnd
            data[layer][subfolder]['nLevel'][neuron] = nLevel

            print(f'{subfolder}/{layer}/{neuron} done')

    dfAnd = {}
    dfLevel = {}
    totalAnd = pd.DataFrame()
    totalLevel = pd.DataFrame()
    for layer in data:
        dfAnd[layer] = pd.DataFrame()
        dfLevel[layer] = pd.DataFrame()
        for subfolder in data[layer]:
            auxAnd = pd.DataFrame(data[layer][subfolder]['nAnd'], index=[subfolder]).transpose()
            auxLevel = pd.DataFrame(data[layer][subfolder]['nLevel'], index=[subfolder]).transpose()

            dfAnd[layer] = pd.concat([dfAnd[layer], auxAnd], axis=1)
            dfLevel[layer] = pd.concat([dfLevel[layer], auxLevel], axis=1)

            print(f'Appended {layer}/{subfolder}')
        dfAnd[layer] = dfAnd[layer].sort_values(['ABC'])
        # dfLevel[layer] = dfLevel[layer].reindex(dfAnd[layer].index)
        dfLevel[layer] = dfLevel[layer].sort_values(['ABC'])

        dfAnd[layer] = dfAnd[layer].reset_index(drop=True)
        dfLevel[layer] = dfLevel[layer].reset_index(drop=True)

        totalAnd = pd.concat([totalAnd, dfAnd[layer].sum(axis=0)], axis=1)
        totalLevel = pd.concat([totalLevel, dfLevel[layer].sum(axis=0)], axis=1)

    for layer in data:
        # Plot 1
        fig, axes = plt.subplots(nrows=1, ncols=1)
        fig.suptitle('AIG synthesis results')
        ax = dfAnd[layer][['ABC', 'ESPRESSO']].plot.area(stacked=False, title='ANDs')
        ax.set_ylabel('Number of ANDs in AIG')
        ax.set_xlim([0, len(dfAnd[layer]) - 1])
        # ax.set_xlabel('Neuron label')
        ax.legend(loc='lower center', bbox_to_anchor=(0.5,-0.25), ncol=1)
        plt.tight_layout()
        plt.savefig(f'img/aigerStats/{modelName}/{layer}_ABC_ESPRESSO_and.png', bbox_inches='tight')

        fig, axes = plt.subplots(nrows=1, ncols=1)
        ax = dfLevel[layer][['ABC', 'ESPRESSO']].plot.area(stacked=False, title='Levels')
        ax.set_ylabel('Number of Levels in AIG')
        ax.set_xlim([0, len(dfAnd[layer]) - 1])
        # ax.set_xlabel('Neuron label')
        ax.legend(loc='lower center', bbox_to_anchor=(0.5,-0.25), ncol=1)
        plt.tight_layout()
        plt.savefig(f'img/aigerStats/{modelName}/{layer}_ABC_ESPRESSO_level.png', bbox_inches='tight')

        # Plot 2
        fig, axes = plt.subplots(nrows=1, ncols=1)
        fig.suptitle('AIG synthesis results')
        ax = dfAnd[layer][['ESPRESSO', 'ESPRESSOOptimizedPerEntry_2']].plot.area(stacked=False, title='ANDs')
        ax.set_ylabel('Number of ANDs in AIG')
        ax.set_xlim([0, len(dfAnd[layer]) - 1])
        # ax.set_xlabel('Neuron label')
        ax.legend(loc='lower center', bbox_to_anchor=(0.5,-0.25), ncol=1)
        plt.tight_layout()
        plt.savefig(f'img/aigerStats/{modelName}/{layer}_ESPRESSO_ESPRESSOOptimizedPerEntry_2_and.png', bbox_inches='tight')

        fig, axes = plt.subplots(nrows=1, ncols=1)
        ax = dfLevel[layer][['ESPRESSO', 'ESPRESSOOptimizedPerEntry_2']].plot.area(stacked=False, title='Levels')
        ax.set_ylabel('Number of Levels in AIG')
        ax.set_xlim([0, len(dfAnd[layer]) - 1])
        # ax.set_xlabel('Neuron label')
        ax.legend(loc='lower center', bbox_to_anchor=(0.5,-0.25), ncol=1)
        plt.tight_layout()
        plt.savefig(f'img/aigerStats/{modelName}/{layer}_ESPRESSO_ESPRESSOOptimizedPerEntry_2_level.png', bbox_inches='tight')

        # Plot 3
        fig, axes = plt.subplots(nrows=1, ncols=1)
        fig.suptitle('AIG synthesis results')
        ax = dfAnd[layer][['ESPRESSOOptimizedPerEntry_2', 'ESPRESSOOptimizedPerEntry_0']].plot.area(stacked=False, title='ANDs')
        ax.set_ylabel('Number of ANDs in AIG')
        ax.set_xlim([0, len(dfAnd[layer]) - 1])
        # ax.set_xlabel('Neuron label')
        ax.legend(loc='lower center', bbox_to_anchor=(0.5,-0.25), ncol=1)
        plt.tight_layout()
        plt.savefig(f'img/aigerStats/{modelName}/{layer}_ESPRESSOOptimizedPerEntry_0_ESPRESSOOptimizedPerEntry_2_and.png', bbox_inches='tight')

        fig, axes = plt.subplots(nrows=1, ncols=1)
        ax = dfLevel[layer][['ESPRESSOOptimizedPerEntry_2', 'ESPRESSOOptimizedPerEntry_0']].plot.area(stacked=False, title='Levels')
        ax.set_ylabel('Number of Levels in AIG')
        ax.set_xlim([0, len(dfAnd[layer]) - 1])
        # ax.set_xlabel('Neuron label')
        ax.legend(loc='lower center', bbox_to_anchor=(0.5,-0.25), ncol=1)
        plt.tight_layout()
        plt.savefig(f'img/aigerStats/{modelName}/{layer}_ESPRESSOOptimizedPerEntry_0_ESPRESSOOptimizedPerEntry_2_level.png', bbox_inches='tight')

    totalAnd.columns = data.keys()
    totalAnd = totalAnd.transpose()
    totalLevel.columns = data.keys()
    totalLevel = totalLevel.transpose()
    # Plot comparative
    fig, axes = plt.subplots(nrows=1, ncols=1)
    fig.suptitle('AIG synthesis results')
    ax = totalAnd[['ABC', 'ESPRESSO', 'ESPRESSOOptimizedPerEntry_2', 'ESPRESSOOptimizedPerEntry_0']].plot.bar(title='ANDs')
    ax.set_ylabel('Number of ANDs in AIG')
    # ax.set_xlabel('Neuron label')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5,-0.25), ncol=2)
    plt.tight_layout()
    plt.savefig(f'img/aigerStats/{modelName}/totalComparisonAnd.png', bbox_inches='tight')

    fig, axes = plt.subplots(nrows=1, ncols=1)
    ax = totalLevel[['ABC', 'ESPRESSO', 'ESPRESSOOptimizedPerEntry_2', 'ESPRESSOOptimizedPerEntry_0']].plot.bar(title='Levels')
    ax.set_ylabel('Number of Levels in AIG')
    # ax.set_xlabel('Neuron label')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5,-0.25), ncol=2)
    plt.tight_layout()
    plt.savefig(f'img/aigerStats/{modelName}/totalComparisonLevel.png', bbox_inches='tight')

    with open(f'img/aigerStats/{modelName}/data.txt', 'w') as f:
        totalAnd.loc['total'] = totalAnd.sum(axis=0)
        abcTotalAnd = totalAnd['ABC']
        totalAnd = -totalAnd.subtract(totalAnd['ABC'], axis=0)
        totalAnd = totalAnd.divide(abcTotalAnd, axis=0)*100
        totalAnd = totalAnd.transpose()
        totalAnd = totalAnd.sort_values(['total'], ascending=False)

        totalLevel.loc['total'] = totalLevel.sum(axis=0)
        abcTotalLevel = totalLevel['ABC']
        totalLevel = -totalLevel.subtract(totalLevel['ABC'], axis=0)
        totalLevel = totalLevel.divide(abcTotalLevel, axis=0)*100
        totalLevel = totalLevel.transpose()
        totalLevel = totalLevel.sort_values(['total'], ascending=False)

        f.write('================================================================\n')
        f.write('AND RESULTS\n')
        f.write('================================================================\n')
        f.write(abcTotalAnd.to_string())
        f.write('\n================================================================\n')
        f.write(totalAnd.to_string())
        f.write('\n\n')
        f.write('================================================================\n')
        f.write('LEVEL RESULTS\n')
        f.write('================================================================\n')
        f.write(abcTotalLevel.to_string())
        f.write('\n================================================================\n')
        f.write(totalLevel.to_string())
    f.close()

        