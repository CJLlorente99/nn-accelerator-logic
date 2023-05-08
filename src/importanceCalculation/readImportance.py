import pandas as pd
import plotly.graph_objects as go

importanceFolder = 'data/layersImportance/'
importanceFile = 'layer3Importance1e1GradientBinarySignBNN50epochs100npl'
importanceFilename = importanceFolder + importanceFile

df = pd.read_feather(importanceFilename)

classTags = [f'class{i}' for i in range(10)]
df['numImportant'] = (df[classTags] > 0).sum(axis=1)
df.sort_values(['numImportant'], inplace=True)

# Plot the number of important classes
fig = go.Figure()

fig.add_trace(go.Scatter(name='Number of Important Classes', x=df['name'], y=df['numImportant']))

fig.update_xaxes(type='category', tickmode='linear')
fig.update_layout(title='Number of Important Classes', barmode='stack', hovermode="x unified")

fig.show()

# Plot the importance of each class
df.sort_values(['importance'], inplace=True)
for i in classTags:
    fig = go.Figure()

    fig.add_trace(go.Scatter(name=f'Importance {i}', x=df['name'], y=df[i]))

    fig.update_xaxes(type='category', tickmode='linear')
    fig.update_layout(title=f'Importance per Neuron {i}', barmode='stack', hovermode="x unified")
    fig.show()

