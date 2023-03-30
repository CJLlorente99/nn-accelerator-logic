from steFunction import StraightThroughEstimator
from torch import nn
import torch.nn.functional as F


class FPNeuralNetwork(nn.Module):
	def __init__(self):
		super(FPNeuralNetwork, self).__init__()
		self.flatten = nn.Flatten()
		self.stack = nn.Sequential(
			nn.Linear(28 * 28, 100),
			nn.BatchNorm1d(100),
			nn.Dropout(p=0.1),
			nn.ReLU(),
			nn.Linear(100, 100),
			nn.BatchNorm1d(100),
			nn.Dropout(p=0.1),
			nn.ReLU(),
			nn.Linear(100, 100),
			nn.BatchNorm1d(100),
			nn.Dropout(p=0.1),
			nn.ReLU(),
			nn.Linear(100, 100),
			nn.BatchNorm1d(100),
			nn.Dropout(p=0.1),
			nn.ReLU(),
			nn.Linear(100, 10),
			nn.BatchNorm1d(10),
		)

	def forward(self, x):
		x = self.flatten(x)
		logits = self.stack(x)
		return F.log_softmax(logits)
