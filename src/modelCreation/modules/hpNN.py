from steFunction import StraightThroughEstimator
from torch import nn
import torch.nn.functional as F
import torch

'''
ReLU but outputs quantized to float16
'''
relu = nn.ReLU()


def reluForwardF16(mod, input, output):
	return output.to(torch.float16).to(torch.float32)


relu.register_forward_hook(reluForwardF16)


class HPNeuralNetwork(nn.Module):
	def __init__(self):
		super(HPNeuralNetwork, self).__init__()
		self.flatten = nn.Flatten()
		self.stack = nn.Sequential(
			nn.Linear(28 * 28, 100),
			nn.BatchNorm1d(100),
			nn.Dropout(p=0.1),
			relu,
			nn.Linear(100, 100),
			nn.BatchNorm1d(100),
			nn.Dropout(p=0.1),
			relu,
			nn.Linear(100, 100),
			nn.BatchNorm1d(100),
			nn.Dropout(p=0.1),
			relu,
			nn.Linear(100, 100),
			nn.BatchNorm1d(100),
			nn.Dropout(p=0.1),
			relu,
			nn.Linear(100, 10),
			nn.BatchNorm1d(10),
		)

	def forward(self, x):
		x = self.flatten(x)
		logits = self.stack(x)
		return F.log_softmax(logits)
