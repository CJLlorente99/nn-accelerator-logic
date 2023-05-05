import torch
import torch.nn.functional as F
from hpo.binaryNNHPO import BinaryNeuralNetwork
import torch.optim as optim
import os
from ray import tune
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader
from models.auxFunctions import ToBlackAndWhite

'''
Train and test functions
'''

# Check mps maybe if working in MacOS
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(config, checkpoint_dir='checkpoint_dir'):
	model = BinaryNeuralNetwork(config['hiddenLayers'], config['npl']).to(device)

	optimizer = optim.Adamax(model.parameters(), lr=config['lr'], weight_decay=config['wd'])

	training_data = datasets.MNIST(
		root='C:/Users/carlo/OneDrive/Documentos/Universidad/MUIT/Segundo/TFM/Code/data',
		train=True,
		download=False,
		transform=Compose([
			ToBlackAndWhite(),
			ToTensor()
		])
	)

	test_data = datasets.MNIST(
		root='C:/Users/carlo/OneDrive/Documentos/Universidad/MUIT/Segundo/TFM/Code/data',
		train=False,
		download=False,
		transform=Compose([
			ToBlackAndWhite(),
			ToTensor()
		])
	)

	train_dataloader = DataLoader(training_data, batch_size=32)
	test_dataloader = DataLoader(test_data, batch_size=32)

	if checkpoint_dir:
		model_state, optimizer_state = torch.load(
			os.path.join(checkpoint_dir, "checkpoint"))
		model.load_state_dict(model_state)
		optimizer.load_state_dict(optimizer_state)

	for epoch in range(10):  # loop over the dataset multiple times
		running_loss = 0.0
		epoch_steps = 0
		for i, data in enumerate(train_dataloader, 0):
			# get the inputs; data is a list of [inputs, labels]
			inputs, labels = data
			inputs, labels = inputs.to(device), labels.to(device)

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = model(inputs)
			loss = F.nll_loss(outputs, labels)
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()
			epoch_steps += 1
			if i % 2000 == 1999:  # print every 2000 mini-batches
				print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
												running_loss / epoch_steps))
				running_loss = 0.0

		val_loss = 0.0
		val_steps = 0
		total = 0
		correct = 0
		for i, data in enumerate(test_dataloader, 0):
			with torch.no_grad():
				inputs, labels = data
				inputs, labels = inputs.to(device), labels.to(device)

				outputs = model(inputs)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()

				loss = F.nll_loss(outputs, labels)
				val_loss += loss.cpu().numpy()
				val_steps += 1

		with tune.checkpoint_dir(epoch) as checkpoint_dir:
			path = os.path.join(checkpoint_dir, "checkpoint")
			torch.save((model.state_dict(), optimizer.state_dict()), path)

		tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
