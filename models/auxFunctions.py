import torch
import torch.nn.functional as F

'''
Train and test functions
'''

# Check mps maybe if working in MacOS
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(dataloader, model, optimizer):
	size = len(dataloader.dataset)
	model.train()
	for batch, (X, y) in enumerate(dataloader):
		X, y = X.to(device), y.to(device)

		# Compute prediction error
		pred = model(X)
		loss = F.nll_loss(pred, y)

		# Backpropagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if batch % 100 == 0:
			loss, current = loss.item(), (batch + 1) * len(X)
			print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model):
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	model.eval()
	test_loss, correct = 0, 0
	with torch.no_grad():
		for X, y in dataloader:
			X, y = X.to(device), y.to(device)
			pred = model(X)
			test_loss += F.nll_loss(pred, y).item()
			correct += (pred.argmax(1) == y).type(torch.float).sum().item()
	test_loss /= num_batches
	correct /= size
	print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def trainAndTest(epochs, train_dataloader, test_dataloader, model, opt):
	for t in range(epochs):
		print(f"Epoch {t + 1}\n-------------------------------")
		train(train_dataloader, model, opt)
		test(test_dataloader, model)

# Transform to black (1) and white (0)


class ToBlackAndWhite(object):

	def __call__(self, sample):
		return sample.convert('1')
