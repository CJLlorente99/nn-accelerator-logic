import pandas as pd
import plotly.graph_objects as go

importanceFolder = 'C:/Users/carlo/OneDrive/Documentos/Universidad/MUIT/Segundo/TFM/Code/data/layersImportance/'
importanceFile = 'layer1ImportanceGradientBinarySignBNN50epochs4096npl'
importanceFilename = importanceFolder + importanceFile

df = pd.read_feather(importanceFilename)

classTags = [f'class{i}' for i in range(10)]
df['numImportant'] = (df[classTags] > 0).sum(axis=1)
df.sort_values(['numImportant'], inplace=True)

fig = go.Figure()

fig.add_trace(go.Scatter(name='Number of Important Classes', x=df['name'], y=df['numImportant']))

# fig.update_xaxes(type='category', tickmode='linear')
fig.update_layout(title='Number of Important Classes Layer 1', barmode='stack', hovermode="x unified")
fig.show()

