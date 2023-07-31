import os
import pandas as pd

dirs = os.listdir(dir)

for d in dirs:
    subdirs = os.listdir(f'{dir}/{d}')

    for sd in subdirs:
        files = os.listdir(f'{dir}/{d}/{sd}')
        for file in files:
            df = pd.read_csv(f'{dir}/{d}/{sd}/{file}', index_col=False)
            df.to_feather(f'{dir}/{d}/{sd}/{file}')
            print(f'{dir}/{d}/{sd}/{file} done')
