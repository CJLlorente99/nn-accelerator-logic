from steFunction import StraightThroughEstimator
from torch import nn
import torch.nn.functional as F


# TODO. Take decision to add dropout
class BinaryNeuralNetwork(nn.Module):
	def __init__(self, neuronPerLayer):
		super(BinaryNeuralNetwork, self).__init__()
		self.flatten = nn.Flatten()
		self.stack = nn.Sequential(
			nn.Linear(28 * 28, neuronPerLayer),
			nn.BatchNorm1d(neuronPerLayer),
			StraightThroughEstimator(),
			nn.Linear(neuronPerLayer, neuronPerLayer),
			nn.BatchNorm1d(neuronPerLayer),
			StraightThroughEstimator(),
			nn.Linear(neuronPerLayer, neuronPerLayer),
			nn.BatchNorm1d(neuronPerLayer),
			StraightThroughEstimator(),
			nn.Linear(neuronPerLayer, neuronPerLayer),
			nn.BatchNorm1d(neuronPerLayer),
			StraightThroughEstimator(),
			nn.Linear(neuronPerLayer, 10),
			nn.BatchNorm1d(10),
		)

	def forward(self, x):
		x = self.flatten(x)
		logits = self.stack(x)
		return F.log_softmax(logits)
