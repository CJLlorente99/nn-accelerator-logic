import torch
import torch.nn.functional as F


class StraightThroughEstimator(torch.autograd.Function):

	@staticmethod
	def forward(ctx, input):
		return (input > 0).float()

	@staticmethod
	def backward(ctx, grad_output):
		return F.hardtanh(grad_output)


class STEFunction(torch.nn.Module):
	def __init__(self):
		super(STEFunction, self).__init__()

	def forward(self, input):
		return StraightThroughEstimator.apply(input)
