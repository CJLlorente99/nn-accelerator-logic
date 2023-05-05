import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

legacyFileLayer0 = '../data/layersImportance/2023_04_18_layer0ImportanceLegacyBinary20epoch100nnpl'
gradientFileLayer0 = '../data/layersImportance/2023_04_18_layer0ImportanceGradientBinary20epochs100nnpl'

legacyFileLayer1 = '../data/layersImportance/2023_04_18_layer1ImportanceLegacyBinary20epoch100nnpl'
gradientFileLayer1 = '../data/layersImportance/2023_04_18_layer1ImportanceGradientBinary20epochs100nnpl'

'''
Layer 0
'''

# Load df

legacyDfLayer0 = pd.read_feather(legacyFileLayer0).set_index(['name'])
gradientDfLayer0 = pd.read_feather(gradientFileLayer0).set_index(['name'])

legacyDfLayer0['normalized' + legacyDfLayer0.columns] = (legacyDfLayer0-legacyDfLayer0.min())/(legacyDfLayer0.max()-legacyDfLayer0.min())
gradientDfLayer0['normalized' + gradientDfLayer0.columns] = (gradientDfLayer0-gradientDfLayer0.min())/(gradientDfLayer0.max()-gradientDfLayer0.min())

legacyDfLayer0['normalizedError'] = gradientDfLayer0['normalizedimportance'] - legacyDfLayer0['normalizedimportance']

for i in range(10):
	legacyDfLayer0[f'normalizedError{i}'] = gradientDfLayer0[f'normalizedclass{i}'] - legacyDfLayer0[f'normalizedclass{i}']

# Create comparison plot

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(go.Bar(x=legacyDfLayer0.index, y=legacyDfLayer0['importance'], name='Legacy'),
			  secondary_y=False)
fig.add_trace(go.Scatter(x=gradientDfLayer0.index, y=gradientDfLayer0['importance'], name='Gradient', mode='markers'),
			  secondary_y=True)

fig.update_layout(
   xaxis = dict(
      tickmode='linear'
   ),
	hovermode="x unified",
	title='Importance Layer 0'
)

fig.show()

# Normalized comparison plot

fig = go.Figure()

fig.add_trace(go.Bar(x=legacyDfLayer0.index, y=legacyDfLayer0['normalizedimportance'], name='Legacy',
					 error_y=dict(
					 type='data',
					 symmetric=False,
					 array=legacyDfLayer0['normalizedError'].tolist())))
fig.add_trace(go.Scatter(x=gradientDfLayer0.index, y=gradientDfLayer0['normalizedimportance'], name='Gradient', mode='markers'))

fig.update_layout(
   xaxis = dict(
      tickmode='linear'
   ),
	hovermode="x unified",
	title='Normalized Importance Layer 0'
)

fig.show()

# Create comparison plot per class
# TODO. Results seem uncoupled

# fig = make_subplots(rows=2, cols=2,
# 					subplot_titles=[f'class{i}' for i in range(4)])
#
# for i in range(4):
# 	fig.add_trace(go.Bar(x=legacyDfLayer0.index, y=legacyDfLayer0[f'normalizedclass{i}'], name=f'Legacy Class {i}',
# 						 error_y=dict(
# 						 type='data',
# 						 symmetric=False,
# 						 array=legacyDfLayer0[f'normalizedError{i}'].tolist())), row=(i // 2) + 1, col=(i % 2) + 1)
# 	# fig.add_trace(go.Scatter(x=gradientDfLayer0.index, y=gradientDfLayer0[f'normalizedclass{i}'], name=f'Gradient Class {i}',
# 	# 						 mode='markers'), row=(i // 2) + 1, col=(i % 2) + 1)
#
# fig.show()

'''
Layer 1
'''

# Load df

legacyDfLayer1 = pd.read_feather(legacyFileLayer1).set_index(['name'])
gradientDfLayer1 = pd.read_feather(gradientFileLayer1).set_index(['name'])

legacyDfLayer1['normalized' + legacyDfLayer1.columns] = (legacyDfLayer1-legacyDfLayer1.min())/(legacyDfLayer1.max()-legacyDfLayer1.min())
gradientDfLayer1['normalized' + gradientDfLayer1.columns] = (gradientDfLayer1-gradientDfLayer1.min())/(gradientDfLayer1.max()-gradientDfLayer1.min())

legacyDfLayer1['normalizedError'] = gradientDfLayer1['normalizedimportance'] - legacyDfLayer1['normalizedimportance']

for i in range(10):
	legacyDfLayer1[f'normalizedError{i}'] = gradientDfLayer1[f'normalizedclass{i}'] - legacyDfLayer1[f'normalizedclass{i}']

# Create comparison plot

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(go.Bar(x=legacyDfLayer1.index, y=legacyDfLayer1['importance'], name='Legacy'),
			  secondary_y=False)
fig.add_trace(go.Scatter(x=gradientDfLayer1.index, y=gradientDfLayer1['importance'], name='Gradient', mode='markers'),
			  secondary_y=True)

fig.update_layout(
   xaxis = dict(
      tickmode='linear'
   ),
	hovermode="x unified",
	title='Importance Layer 1'
)

fig.show()

# Normalized comparison plot

fig = go.Figure()

fig.add_trace(go.Bar(x=legacyDfLayer1.index, y=legacyDfLayer1['normalizedimportance'], name='Legacy',
					 error_y=dict(
						 type='data',
						 symmetric=False,
						 array=legacyDfLayer1['normalizedError'].tolist())))
fig.add_trace(go.Scatter(x=gradientDfLayer1.index, y=gradientDfLayer1['normalizedimportance'], name='Gradient', mode='markers'))

fig.update_layout(
   xaxis = dict(
      tickmode='linear'
   ),
	hovermode="x unified",
	title='Normalized Importance Layer 1'
)

fig.show()

# Create comparison plot per class
# TODO. Results seem uncoupled

# fig = make_subplots(rows=2, cols=2,
# 					subplot_titles=[f'class{i}' for i in range(4)])
#
# for i in range(4):
# 	fig.add_trace(go.Bar(x=legacyDfLayer1.index, y=legacyDfLayer1[f'normalizedclass{i}'], name=f'Legacy Class {i}',
# 						 error_y=dict(
# 						 type='data',
# 						 symmetric=False,
# 						 array=legacyDfLayer1[f'normalizedError{i}'].tolist())), row=(i // 2) + 1, col=(i % 2) + 1)
# 	# fig.add_trace(go.Scatter(x=gradientDfLayer1.index, y=gradientDfLayer1[f'normalizedclass{i}'], name=f'Gradient Class {i}',
# 	# 						 mode='markers'), row=(i // 2) + 1, col=(i % 2) + 1)
#
# fig.show()
