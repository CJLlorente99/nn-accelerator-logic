import pandas as pd
from modelsCommon.steFunction import STEFunction
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from ttUtilities.auxFunctions import binaryArrayToSingleValue, integerToBinaryArray
import math

class binaryVGGVerySmall(nn.Module):
	def __init__(self, resizeFactor):
		super(binaryVGGVerySmall, self).__init__()

		# Layer 0
		self.conv0 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
		self.bn0 = nn.BatchNorm2d(64)
		self.relu0 = nn.ReLU()
		self.maxpool0 = nn.MaxPool2d(kernel_size=2, stride=2)

		# Layer 1
		self.conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm2d(128)
		self.relu1 = nn.ReLU()
		self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
  
		# Layer 2.1
		self.conv21 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
		self.bn21 = nn.BatchNorm2d(256)
		self.relu21 = STEFunction()
		self.maxpool22 = nn.MaxPool2d(kernel_size=2, stride=2)
  
		# Layer 3.1
		self.conv31 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
		self.bn31 = nn.BatchNorm2d(512)
		self.relu31 = STEFunction()
		self.maxpool32 = nn.MaxPool2d(kernel_size=2, stride=2)
  
		# Layer 4.1
		self.conv41 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.bn41 = nn.BatchNorm2d(512)
		self.relu41 = STEFunction()
		self.maxpool42 = nn.MaxPool2d(kernel_size=2, stride=2)
  
		# Layer FC0
		self.l0 = nn.Linear(resizeFactor*resizeFactor*512, 1024)
		self.bnl0 = nn.BatchNorm1d(1024)
		self.relul0 = STEFunction()
  
		# Layer FC1
		self.l1 = nn.Linear(1024, 250)
		self.bnl1 = nn.BatchNorm1d(250)
		self.relul1 = STEFunction()
  
		# Layer FC2
		self.l2 = nn.Linear(250, 10)
  
		# Initialize
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				m.bias.data.zero_()
    

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
  
		# Layer 2.1
		x = self.conv21(x)
		x = self.bn21(x)
		x = self.relu21(x)
		x = self.maxpool22(x)
  
		# Layer 3.1
		x = self.conv31(x)
		x = self.bn31(x)
		x = self.relu31(x)
		x = self.maxpool32(x)
  
		# Layer 4.1
		x = self.conv41(x)
		x = self.bn41(x)
		x = self.relu41(x)
		x = self.maxpool42(x)
  
		x = x.reshape(x.size(0), -1)
  
		# Layer FC0
		x = self.l0(x)
		x = self.bnl0(x)
		x = self.relul0(x)
  
		# Layer FC1
		x = self.l1(x)
		x = self.bnl1(x)
		x = self.relul1(x)
  
		# Layer FC2
		x = self.l2(x)

		return x
