import torch
import torch.nn.functional as F


class StraightThroughEstimator(torch.autograd.Function):

	@staticmethod
	def forward(ctx, input):
		input[input >= 0] = 1
		input[input < 0] = -1
		return input

	@staticmethod
	def backward(ctx, grad_output):
		return F.hardtanh(grad_output)


class STEFunction(torch.nn.Module):
	def __init__(self):
		super(STEFunction, self).__init__()

	def forward(self, input):
		return StraightThroughEstimator.apply(input)
