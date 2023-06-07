import pandas as pd
from modelsCommon.steFunction import STEFunction
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from ttUtilities.auxFunctions import binaryArrayToSingleValue, integerToBinaryArray
import math

class VGGSmall(nn.Module):
	def __init__(self):
		super(VGGSmall, self).__init__()

		# Layer 0
		self.conv0 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
		self.ste0 = nn.ReLU()
		self.maxpool0 = nn.MaxPool2d(kernel_size=2, stride=2)

		# Layer 1
		self.conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
		self.ste1 = STEFunction()
		self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
  
		# Layer 2.1
		self.conv21 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
		self.ste21 = STEFunction()
  
		# Layer 2.2
		self.conv22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.ste22 = STEFunction()
		self.maxpool22 = nn.MaxPool2d(kernel_size=2, stride=2)
  
		# Layer 3.1
		self.conv31 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
		self.ste31 = STEFunction()

		# Layer 3.2
		self.conv32 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.ste32 = STEFunction()
		self.maxpool32 = nn.MaxPool2d(kernel_size=2, stride=2)
  
		# Layer 4.1
		self.conv41 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.ste41 = STEFunction()
  
		# Layer 4.2
		self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.ste42 = STEFunction()
		self.maxpool42 = nn.MaxPool2d(kernel_size=2, stride=2)
  
		# Layer FC0
		self.dropoutl0 = nn.Dropout(0.5)
		self.l0 = nn.Linear(512, 512)
		self.stel0 = STEFunction()
  
		# Layer FC1
		self.dropoutl1 = nn.Dropout(0.5)
		self.l1 = nn.Linear(512, 512)
		self.stel1 = STEFunction()
  
		# Layer FC2
		self.l2 = nn.Linear(512, 10)
  
		# Initialize
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				m.bias.data.zero_()
    

	def forward(self, x):
		# Layer 0
		x = self.conv0(x)
		x = self.ste0(x)
		x = self.maxpool0(x)

		# Layer 1
		x = self.conv1(x)
		x = self.ste1(x)
		x = self.maxpool1(x)
  
		# Layer 2.1
		x = self.conv21(x)
		x = self.ste21(x)
  
		# Layer 2.2
		x = self.conv22(x)
		x = self.ste22(x)
		x = self.maxpool22(x)
  
		# Layer 3.1
		x = self.conv31(x)
		x = self.ste31(x)

		# Layer 3.2
		x = self.conv32(x)
		x = self.ste32(x)
		x = self.maxpool32(x)
  
		# Layer 4.1
		x = self.conv41(x)
		x = self.ste41(x)
  
		# Layer 4.2
		x = self.conv42(x)
		x = self.ste42(x)
		x = self.maxpool42(x)
  
		x = x.reshape(x.size(0), -1)
  
		# Layer FC0
		x = self.dropoutl0(x)
		x = self.l0(x)
		x = self.stel0(x)
  
		# Layer FC1
		x = self.dropoutl1(x)
		x = self.l1(x)
		x = self.stel1(x)
  
		# Layer FC2
		x = self.l2(x)

		return x
