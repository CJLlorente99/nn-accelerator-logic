import torch
import torch.nn.functional as F

'''
Train and test functions
'''

# Check mps maybe if working in MacOS
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(dataloader, model, optimizer, criterion):
	size = len(dataloader.dataset)
	model.train()
	for batch, (X, y) in enumerate(dataloader):
		X, y = X.to(device), y.to(device)

		# Compute prediction error
		pred = model(X)
		# loss = F.nll_loss(pred, y)
		loss = criterion(pred, y)

		# Backpropagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if batch % 100 == 0:
			loss, current = loss.item(), (batch + 1) * len(X)
			print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, dataloader2, model, criterion):
	sizeTotal = len(dataloader.dataset) + len(dataloader2.dataset)
	size = len(dataloader.dataset)
	num_batchesTotal = len(dataloader) + len(dataloader2)
	num_batches = len(dataloader)
	model.eval()
	test_loss, correct = 0, 0
	with torch.no_grad():
		for X, y in dataloader:
			X, y = X.to(device), y.to(device)
			pred = model(X)
			test_loss += criterion(pred, y).item()
			correct += (pred.argmax(1) == y).type(torch.float).sum().item()

		print(f"Test Error Test: \n Accuracy: {(100 * correct / size):>0.2f}%, Avg loss: {test_loss / num_batches:>8f} \n")

		for X, y in dataloader2:
			X, y = X.to(device), y.to(device)
			pred = model(X)
			test_loss += criterion(pred, y).item()
			correct += (pred.argmax(1) == y).type(torch.float).sum().item()
	test_loss /= num_batchesTotal
	correct /= sizeTotal
	print(f"Test Error Total: \n Accuracy: {(100*correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")


def trainAndTest(epochs, train_dataloader, test_dataloader, model, opt, criterion):
	for t in range(epochs):
		print(f"Epoch {t + 1}\n-------------------------------")
		if t in [99, 149, 299]:
			opt.param_groups['lr'] /= 10
		train(train_dataloader, model, opt, criterion)
		test(test_dataloader, train_dataloader, model, criterion)
  
def getAccInfo(train_dataloader, test_dataloader, model, criterion):
	retStr = ""
	sizeTotal = len(train_dataloader.dataset) + len(test_dataloader.dataset)
	size = len(train_dataloader.dataset)
	num_batchesTotal = len(train_dataloader) + len(test_dataloader)
	num_batches = len(train_dataloader)
	model.eval()
	test_loss, correct = 0, 0
	with torch.no_grad():
		for X, y in train_dataloader:
			X, y = X.to(device), y.to(device)
			pred = model(X)
			test_loss += criterion(pred, y).item()
			correct += (pred.argmax(1) == y).type(torch.float).sum().item()

		retStr.append(f"Test Error Test: \n Accuracy: {(100 * correct / size):>0.2f}%, Avg loss: {test_loss / num_batches:>8f} \n")

		for X, y in test_dataloader:
			X, y = X.to(device), y.to(device)
			pred = model(X)
			test_loss += criterion(pred, y).item()
			correct += (pred.argmax(1) == y).type(torch.float).sum().item()
	test_loss /= num_batchesTotal
	correct /= sizeTotal
	retStr.append(f"Test Error Total: \n Accuracy: {(100*correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")
 
	return retStr

