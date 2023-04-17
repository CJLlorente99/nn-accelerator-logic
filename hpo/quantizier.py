import math

import torch
import torch.nn.functional as F

numBits = 1


class StraightThroughEstimator(torch.autograd.Function):

	@staticmethod
	def forward(ctx, input):
		# Clip xc
		xc = input
		xc[xc >= 1] = 1
		xc[xc <= 0] = 0

		# Quantizy
		N = 2**numBits
		quantizationStep = 1 / (N - 1)
		xr = torch.round(xc / quantizationStep)

		return xr

	@staticmethod
	def backward(ctx, grad_output):
		return F.hardtanh(grad_output)


class STEFunctionQuant(torch.nn.Module):
	def __init__(self, bits):
		super(STEFunctionQuant, self).__init__()
		global numBits
		numBits = bits

	def forward(self, input):
		return StraightThroughEstimator.apply(input)
