import pandas as pd
import matplotlib.pyplot as plt
import os
import json

data = {}

for modelName in ['eeb/eeb_prunedBT12_100ep_100npl_all']:

    print(f'{modelName}')

    if not os.path.exists(f'img/yosysStats/{modelName}'):
        os.makedirs(f'img/yosysStats/{modelName}')

    for subfolder in ['ESPRESSO', 'ESPRESSOOptimizedPerEntry_0', 'ESPRESSOOptimizedPerEntry_2']:
        print(f'{modelName} {subfolder}')

        data[subfolder] = {'layer0' : None,
                           'layer1' : None,
                           'layer2' : None,
                           'layer3' : None}

        layer0Filename = f'data/plas/{modelName}/{subfolder}/layer0_stats.txt'
        layer1Filename = f'data/plas/{modelName}/{subfolder}/layer1_stats.txt'
        layer2Filename = f'data/plas/{modelName}/{subfolder}/layer2_stats.txt'
        layer3Filename = f'data/plas/{modelName}/{subfolder}/layer3_stats.txt'

        i = 0
        for layerFilename in [layer0Filename, layer1Filename, layer2Filename, layer3Filename]:    
            with open(layerFilename, 'r') as f:
                flag = False
                jsonString = []
                while True:
                    line = f.readline()

                    if (not line.startswith('6.16. Finished OPT passes.')) and (not flag):
                        continue
                    flag = True

                    if line.startswith('Warnings:'):
                        break

                    jsonString.append(line)
            
            data[subfolder][f'layer{i}'] = json.loads(''.join(jsonString[1:]))
            i += 1

    # Group cells and area
    jsonKeys = data[subfolder]['layer1']['design'].keys()
    dfGeneralData = {'layer0' : pd.DataFrame(),
                     'layer1' : pd.DataFrame(),
                     'layer2' : pd.DataFrame(),
                     'layer3' : pd.DataFrame()}
    
    dfCellsData = {'layer0' : pd.DataFrame(),
                   'layer1' : pd.DataFrame(),
                   'layer2' : pd.DataFrame(),
                   'layer3' : pd.DataFrame()}
    
    for subfolder in data:
        for layer in data[subfolder]:
            dfCellsData[layer] = pd.concat([dfCellsData[layer], pd.DataFrame(data[subfolder][layer]['design']['num_cells_by_type'], index=[subfolder])], axis=0)
            data[subfolder][layer]['design'].pop('num_cells_by_type')
            dfGeneralData[layer] = pd.concat([dfGeneralData[layer], pd.DataFrame(data[subfolder][layer]['design'], index=[subfolder])], axis=0)

    with pd.ExcelWriter(f'img/yosysStats/{modelName}/cellsData.xlsx') as writer:
        for layer in dfCellsData:
            dfCellsData[layer].to_excel(writer, sheet_name=layer)

    with pd.ExcelWriter(f'img/yosysStats/{modelName}/generalData.xlsx') as writer:
        for layer in dfGeneralData:
            dfGeneralData[layer][['area', 'num_wires']].to_excel(writer, sheet_name=layer)

    # Plot comparative
    aux = pd.concat([dfGeneralData['layer0']['area'], dfGeneralData['layer1']['area'], dfGeneralData['layer2']['area'], dfGeneralData['layer3']['area']], axis=1)
    aux.columns = ['layer0', 'layer1', 'layer2', 'layer3']
    aux = aux.transpose()
    fig, axes = plt.subplots(nrows=1, ncols=1)
    fig.suptitle('YOSYS synthesis results')
    ax = aux.plot.bar(title='Area')
    ax.set_ylabel('Area')
    # ax.set_xlabel('Neuron label')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5,-0.25), ncol=2)
    plt.tight_layout()
    plt.savefig(f'img/yosysStats/{modelName}/areaComparison.png', bbox_inches='tight')
    plt.close(fig)

    aux = pd.concat([dfGeneralData['layer0']['num_cells'], dfGeneralData['layer1']['num_cells'], dfGeneralData['layer2']['num_cells'], dfGeneralData['layer3']['num_cells']], axis=1)
    aux.columns = ['layer0', 'layer1', 'layer2', 'layer3']
    aux = aux.transpose()
    fig, axes = plt.subplots(nrows=1, ncols=1)
    fig.suptitle('YOSYS synthesis results')
    ax = aux.plot.bar(title='Number of Cells')
    ax.set_ylabel('Number of Cells')
    # ax.set_xlabel('Neuron label')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5,-0.25), ncol=2)
    plt.tight_layout()
    plt.savefig(f'img/yosysStats/{modelName}/numCellsComparison.png', bbox_inches='tight')
    plt.close(fig)

    aux = pd.concat([dfGeneralData['layer0']['num_wires'], dfGeneralData['layer1']['num_wires'], dfGeneralData['layer2']['num_wires'], dfGeneralData['layer3']['num_wires']], axis=1)
    aux.columns = ['layer0', 'layer1', 'layer2', 'layer3']
    aux = aux.transpose()
    fig, axes = plt.subplots(nrows=1, ncols=1)
    fig.suptitle('YOSYS synthesis results')
    ax = aux.plot.bar(title='Number of Wires')
    ax.set_ylabel('Number of Wires')
    # ax.set_xlabel('Neuron label')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5,-0.25), ncol=2)
    plt.tight_layout()
    plt.savefig(f'img/yosysStats/{modelName}/numWiresComparison.png', bbox_inches='tight')
    plt.close(fig)

    print(f'{modelName} erfolgreich')

        