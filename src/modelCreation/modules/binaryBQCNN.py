import pandas as pd
from modelsCommon.steFunction import STEFunction
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from modelsCommon.customPruning import random_pruning_per_neuron
import math

class BQCNN(nn.Module):
	def __init__(self):
		super(BQCNN, self).__init__()
		self.resizeFactor = resizeFactor

		# Layer 0
		self.conv0 = nn.Conv2d(1, 10, kernel_size=3)
		self.bn0 = nn.BatchNorm2d(10)
		self.relu0 = STEFunction()
		self.maxpool0 = nn.MaxPool2d(kernel_size=2)

		# Layer 1
		self.conv1 = nn.Conv2d(10, 20, kernel_size=3)
		self.bn1 = nn.BatchNorm2d(20)
		self.relu1 = STEFunction()
		self.maxpool1 = nn.MaxPool2d(kernel_size=2)
  
		# Output layer
		self.l0 = nn.Linear(1000, 10)
		self.bnl0 = nn.BatchNorm2d(10)
    

	def forward(self, x):
		# Layer 0
		x = self.conv0(x)
		x = self.bn0(x)
		x = self.relu0(x)
		x = self.maxpool0(x)

		# Layer 1
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu1(x)
		x = self.maxpool1(x)
  
		x = x.reshape(x.size(0), -1)
  
		x = self.l0(x)
		x = self.bnl0(x)

		return F.log_softmax(x, dim=1)