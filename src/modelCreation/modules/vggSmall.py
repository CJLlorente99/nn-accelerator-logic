import pandas as pd
from modelsCommon.steFunction import STEFunction
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from ttUtilities.auxFunctions import binaryArrayToSingleValue, integerToBinaryArray
import math

class VGGSmall(nn.Module):
	def __init__(self, resizeFactor):
		super(VGGSmall, self).__init__()

		# Layer 0
		self.conv0 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
		self.relu0 = nn.ReLU()
		self.maxpool0 = nn.MaxPool2d(kernel_size=2, stride=2)

		# Layer 1
		self.conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
		self.relu1 = nn.ReLU()
		self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
  
		# Layer 2.1
		self.conv21 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
		self.relu21 = nn.ReLU()
  
		# Layer 2.2
		self.conv22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.relu22 = nn.ReLU()
		self.maxpool22 = nn.MaxPool2d(kernel_size=2, stride=2)
  
		# Layer 3.1
		self.conv31 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
		self.relu31 = nn.ReLU()

		# Layer 3.2
		self.conv32 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.relu32 = nn.ReLU()
		self.maxpool32 = nn.MaxPool2d(kernel_size=2, stride=2)
  
		# Layer 4.1
		self.conv41 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.relu41 = nn.ReLU()
  
		# Layer 4.2
		self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.relu42 = nn.ReLU()
		self.maxpool42 = nn.MaxPool2d(kernel_size=2, stride=2)
  
		# Layer FC0
		self.dropoutl0 = nn.Dropout(0.5)
		self.l0 = nn.Linear(resizeFactor*resizeFactor*512, 4096)
		self.relul0 = nn.ReLU()
  
		# Layer FC1
		self.dropoutl1 = nn.Dropout(0.5)
		self.l1 = nn.Linear(4096, 4096)
		self.relul1 = nn.ReLU()

		# Layer FC2
		self.dropoutl2 = nn.Dropout(0.5)
		self.l2 = nn.Linear(4096, 1000)
		self.relul2 = nn.ReLU()
  
		# Layer FC2
		self.l3 = nn.Linear(1000, 10)
  
		# Initialize
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				m.bias.data.zero_()
    

	def forward(self, x):
		# Layer 0
		x = self.conv0(x)
		x = self.relu0(x)
		x = self.maxpool0(x)

		# Layer 1
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.maxpool1(x)
  
		# Layer 2.1
		x = self.conv21(x)
		x = self.relu21(x)
  
		# Layer 2.2
		x = self.conv22(x)
		x = self.relu22(x)
		x = self.maxpool22(x)
  
		# Layer 3.1
		x = self.conv31(x)
		x = self.relu31(x)

		# Layer 3.2
		x = self.conv32(x)
		x = self.relu32(x)
		x = self.maxpool32(x)
  
		# Layer 4.1
		x = self.conv41(x)
		x = self.relu41(x)
  
		# Layer 4.2
		x = self.conv42(x)
		x = self.relu42(x)
		x = self.maxpool42(x)
  
		x = x.reshape(x.size(0), -1)
  
		# Layer FC0
		x = self.dropoutl0(x)
		x = self.l0(x)
		x = self.relul0(x)
  
		# Layer FC1
		x = self.dropoutl1(x)
		x = self.l1(x)
		x = self.relul1(x)

		# Layer FC1
		x = self.dropoutl2(x)
		x = self.l2(x)
		x = self.relul2(x)
  
		# Layer FC2
		x = self.l3(x)

		return x
