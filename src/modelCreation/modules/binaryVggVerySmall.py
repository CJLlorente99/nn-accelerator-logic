import pandas as pd
from modelsCommon.steFunction import STEFunction
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import math
from modelsCommon.customPruning import random_pruning_per_neuron

class binaryVGGVerySmall(nn.Module):
	def __init__(self, resizeFactor, relus: list, connectionsAfterPrune=0):
		super(binaryVGGVerySmall, self).__init__()
		self.resizeFactor = resizeFactor

		# Layer 0
		self.conv0 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
		self.bn0 = nn.BatchNorm2d(64)
		if relus[0]:
			self.relu0 = nn.ReLU()
		else:
			self.relu0 = STEFunction()
		self.maxpool0 = nn.MaxPool2d(kernel_size=2, stride=2)

		# Layer 1
		self.conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm2d(128)
		if relus[1]:
			self.relu1 = nn.ReLU()
		else:
			self.relu1 = STEFunction()
		self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
  
		# Layer 2.1
		self.conv21 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
		self.bn21 = nn.BatchNorm2d(256)
		if relus[2]:
			self.relu21 = nn.ReLU()
		else:
			self.relu21 = STEFunction()
		self.maxpool22 = nn.MaxPool2d(kernel_size=2, stride=2)
  
		# Layer 3.1
		self.conv31 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
		self.bn31 = nn.BatchNorm2d(512)
		if relus[3]:
			self.relu31 = nn.ReLU()
		else:
			self.relu31 = STEFunction()
		self.maxpool32 = nn.MaxPool2d(kernel_size=2, stride=2)
  
		# Layer 4.1
		self.conv41 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.bn41 = nn.BatchNorm2d(512)
		if relus[4]:
			self.relu41 = nn.ReLU()
		else:
			self.relu41 = STEFunction()
		self.maxpool42 = nn.MaxPool2d(kernel_size=2, stride=2)
  
		# Layer FC0
		self.l0 = nn.Linear(resizeFactor*resizeFactor*512, 4096)
		self.bnl0 = nn.BatchNorm1d(4096)
		if relus[5]:
			self.relul0 = nn.ReLU()
		else:
			self.relul0 = STEFunction()
  
		# Layer FC1
		self.l1 = nn.Linear(4096, 4096)
		self.bnl1 = nn.BatchNorm1d(4096)
		if relus[6]:
			self.relul1 = nn.ReLU()
		else:
			self.relul1 = STEFunction()

		# Layer FC2
		self.l2 = nn.Linear(4096, 1000)
		self.bnl2 = nn.BatchNorm1d(1000)
		if relus[7]:
			self.relul2 = nn.ReLU()
		else:
			self.relul2 = STEFunction()
  
		# Layer FC2
		self.l3 = nn.Linear(1000, 10)

		# Regular pruning
		if connectionsAfterPrune != 0:
			self.l0 = random_pruning_per_neuron(self.l0, name="weight", connectionsToPrune=resizeFactor*resizeFactor*512-connectionsAfterPrune)
			self.l1 = random_pruning_per_neuron(self.l1, name="weight", connectionsToPrune=4096-connectionsAfterPrune)
			self.l2 = random_pruning_per_neuron(self.l2, name="weight", connectionsToPrune=4096-connectionsAfterPrune)
  
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
		x = self.bnl2(x)
		x = self.relul2(x)
  
		# Layer FC3
		x = self.l3(x)

		return x
	
	def pruningSparsification(self, inputsAfterPrune):
		prunedConnections = {}

		with torch.no_grad():
			# First hidden layer
			prunedConnections['l0'] = []
			inputsToPrune = self.resizeFactor*self.resizeFactor*512 - inputsAfterPrune
			for j in range(len(self.l0.weight)):
				weights = self.l0.weight[j, :]
				auxWeights = torch.abs(weights).cpu().detach().numpy()
				idxToPrune = np.argpartition(auxWeights, inputsToPrune)[:inputsToPrune]
				nums = weights[idxToPrune]
				weights[idxToPrune] = 0
				prunedConnections['l0'].append(idxToPrune)
			prunedConnections['l0'] = np.array(prunedConnections['l0'])

			# Second hidden layer
			prunedConnections['l1'] = []
			inputsToPrune = 4096 - inputsAfterPrune
			for j in range(len(self.l1.weight)):
				weights = self.l1.weight[j, :]
				auxWeights = torch.abs(weights).cpu().detach().numpy()
				idxToPrune = np.argpartition(auxWeights, inputsToPrune)[:inputsToPrune]
				nums = weights[idxToPrune]
				weights[idxToPrune] = 0
				prunedConnections['l1'].append(idxToPrune)
			prunedConnections['l1'] = np.array(prunedConnections['l1'])

			# Third hidden layer
			prunedConnections['l2'] = []
			inputsToPrune = 4096 - inputsAfterPrune
			for j in range(len(self.l2.weight)):
				weights = self.l2.weight[j, :]
				auxWeights = torch.abs(weights).cpu().detach().numpy()
				idxToPrune = np.argpartition(auxWeights, inputsToPrune)[:inputsToPrune]
				nums = weights[idxToPrune]
				weights[idxToPrune] = 0
				prunedConnections['l2'].append(idxToPrune)
			prunedConnections['l2'] = np.array(prunedConnections['l2'])

		return prunedConnections
