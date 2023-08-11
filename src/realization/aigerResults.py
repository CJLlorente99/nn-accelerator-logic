import re

modelName = 'eeb/eeb_prunedBT10_100ep_100npl'
nLayers = 3

aigerStatsFilename = f'data/aiger/{modelName}/aigerStats.txt'

data = {}

with open(aigerStatsFilename, 'r') as f:
    while True:
        line = f.readline()

        if not line:
            break

        subfolder = re.search(f'{modelName}/(.*)/layer', line)
        data[subfolder] = {f'layer{i}' : {} for i in range(nLayers)}

        