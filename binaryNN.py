from steFunction import StraightThroughEstimator
from torch import nn
import torch.nn.functional as F


# TODO. Take decision to add dropout
class BinaryNeuralNetwork(nn.Module):
	def __init__(self, neuronPerLayer):
		super(BinaryNeuralNetwork, self).__init__()
		self.flatten = nn.Flatten()
		self.l0 = nn.Linear(28 * 28, neuronPerLayer)
		self.bn0 = nn.BatchNorm1d(neuronPerLayer)
		self.ste0 = StraightThroughEstimator()
		self.l1 = nn.Linear(neuronPerLayer, neuronPerLayer)
		self.bn1 = nn.BatchNorm1d(neuronPerLayer)
		self.ste1 = StraightThroughEstimator()
		self.l2 = nn.Linear(neuronPerLayer, neuronPerLayer)
		self.bn2 = nn.BatchNorm1d(neuronPerLayer)
		self.ste2 = StraightThroughEstimator()
		self.l3 = nn.Linear(neuronPerLayer, neuronPerLayer)
		self.bn3 = nn.BatchNorm1d(neuronPerLayer)
		self.ste3 = StraightThroughEstimator()
		self.l4 = nn.Linear(neuronPerLayer, 10)
		self.bn4 = nn.BatchNorm1d(10)

	def forward(self, x):
		x = self.flatten(x)
		x = self.l0(x)
		x = self.bn0(x)
		x = self.ste0(x)
		x = self.l1(x)
		x = self.bn1(x)
		x = self.ste1(x)
		x = self.l2(x)
		x = self.bn2(x)
		x = self.ste2(x)
		x = self.l3(x)
		x = self.bn3(x)
		x = self.ste3(x)
		x = self.l4(x)
		x = self.bn4(x)
		return F.log_softmax(x, dim=1)
