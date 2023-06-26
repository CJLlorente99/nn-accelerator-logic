import pandas as pd
import os

modelName = 'binaryVGGVerySmall_1000001_2'
outputOptmizedFolder = f'data/optimizedTT/{modelName}'
outputNonOptmizedFolder = f'data/nonOptimizedTT/{modelName}'
plaFolderOptimized = f'data/plas/{modelName}/optimized'
plaFolderNonOptimized = f'data/plas/{modelName}/nonOptimized'

# NonOptimized Writing
header = True
for file in os.listdir(outputNonOptmizedFolder):
    nChunk = 0
    for chunk in pd.read_csv(f'{outputNonOptmizedFolder}/{file}', index_col=False, chunksize=5000):
        inputTags = [tag for tag in chunk.columns if tag.startswith('IN')]
        outputTags = [tag for tag in chunk.columns if tag.startswith('OUT')]

        # Write PLA file header for every neuron
        if header:
            for outputTag in outputTags:
                with open(f'{plaFolderNonOptimized}/{file[:-4]}_{outputTag}.pla', 'w') as f:
                    f.write(f'.i {len(inputTags)}\n')
                    f.write(f'.o {1}\n')
                    f.write(f'.p 50000\n')
                print(f'Header of {plaFolderNonOptimized}/{file[:-4]}_{outputTag}.pla written')
            header = False

        # Append TT
        for outputTag in outputTags:
            with open(f'{plaFolderNonOptimized}/{file[:-4]}_{outputTag}.pla', 'a') as f:
                for index, row in chunk.iterrows():
                    row[row == -1] = 0
                    text = ''.join(row[inputTags].astype(int).to_string(header=False, index=False).split('\n'))
                    outputText = row[outputTag].astype(int)
                    f.write(f'{text} {outputText}\n')
                print(f'Append [{int((nChunk+1)*5000)}/50000] of {plaFolderNonOptimized}/{file[:-4]}_{outputTag}.pla')
    nChunk += 1

for file in os.listdir(outputNonOptmizedFolder):
    for outputTag in outputTags:
        with open(f'{plaFolderNonOptimized}/{file[:-4]}_{outputTag}.pla', 'a') as f:
            f.write(f'.e')

# Optimized Writing
header = True
nOptimizedEntries = {}
for file in os.listdir(outputOptmizedFolder):
    nOptimizedEntries[file] = {}
    nChunk = 0
    for chunk in pd.read_csv(f'{outputOptmizedFolder}/{file}', index_col=False, chunksize=5000):
        inputTags = [tag for tag in chunk.columns if tag.startswith('IN')]
        outputTags = [tag for tag in chunk.columns if tag.startswith('OUT')]

        # Write PLA file header for every neuron
        if header:
            for outputTag in outputTags:
                nOptimizedEntries[file][outputTag] = 0
                with open(f'{plaFolderOptimized}/{file[:-4]}_{outputTag}.pla', 'w') as f:
                    f.write(f'.i {len(inputTags)}\n')
                    f.write(f'.o {1}\n')
                    f.write(f'.p XXXXX\n')
                print(f'Header of {plaFolderOptimized}/{file[:-4]}_{outputTag}.pla written')
            header = False

        # Append TT
        for outputTag in outputTags:
            with open(f'{plaFolderOptimized}/{file[:-4]}_{outputTag}.pla', 'a') as f:
                for index, row in chunk.iterrows():
                    row[row == -1] = 0
                    text = ''.join(row[inputTags].astype(int).to_string(header=False, index=False).split('\n'))
                    outputText = row[outputTag].astype(int)
                    if outputText == -5:
                        continue
                    nOptimizedEntries[file][outputTag] += 1
                    f.write(f'{text} {outputText}\n')
                print(f'Append [{int((nChunk+1)*5000)}/50000] of {plaFolderOptimized}/{file[:-4]}_{outputTag}.pla')
    nChunk += 1

for file in os.listdir(outputOptmizedFolder):
    for outputTag in outputTags:
        with open(f'{plaFolderOptimized}/{file[:-4]}_{outputTag}.pla', 'a') as f:
            f.write(f'.e')
            f.close()

        with open(f'{plaFolderOptimized}/{file[:-4]}_{outputTag}.pla', 'r') as f:
            data = f.readlines()
        data[2] = f'.p {int(nOptimizedEntries[file][outputTag])}\n'

        with open(f'{plaFolderOptimized}/{file[:-4]}_{outputTag}.pla', 'w') as f:
            f.writelines(data)

