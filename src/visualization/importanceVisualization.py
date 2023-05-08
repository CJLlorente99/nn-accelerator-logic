from os import listdir
from os.path import isfile, join, basename
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

importanceDataPath = "..\data\importanceCharacterization"

files = [join(importanceDataPath, f) for f in listdir(importanceDataPath) if isfile(join(importanceDataPath, f))]

data = {}
for f in files:
	data[basename(f)[:-11]] = pd.read_csv(f, index_col=0)

listNumNeurons = [10, 30, 80, 100]
titles = [f'Mean Importance Cardinality ({n} neurons per layer)' for n in listNumNeurons]

fig = make_subplots(rows=2, cols=2,
					subplot_titles=titles)
i = 0
for numNeurons in listNumNeurons:
	df = data[str(numNeurons)].sort_values(['numImportance'])

	fig.add_trace(go.Bar(name=f'Mean Importance Cardinality ({numNeurons})', x=df['name'], y=df['numImportance']),
				  row=(i // 2) + 1, col=(i % 2) + 1)

	i += 1

fig.update_layout(showlegend=False)
fig.show()






