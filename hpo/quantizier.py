import math
import torch
import torch.nn.functional as F


class StraightThroughEstimator(torch.autograd.Function):

	@staticmethod
	def forward(ctx, input):
		ctx.save_for_backward(input)

		# Clip xc
		xc = input
		xc[xc >= 1] = 1
		xc[xc <= 0] = 0

		# Quantizy
		numBits = 10
		N = 2**numBits
		quantizationStep = 1 / (N - 1)
		xr = torch.round(xc / quantizationStep) * quantizationStep

		return xr

	@staticmethod
	def backward(ctx, grad_output):
		input, = ctx.saved_tensors
		grad_input = grad_output.clone()
		grad_input[input < 0] = 0
		return grad_input


class STEFunctionQuant(torch.nn.Module):
	def __init__(self):
		super(STEFunctionQuant, self).__init__()

	def forward(self, input):
		return StraightThroughEstimator.apply(input)
