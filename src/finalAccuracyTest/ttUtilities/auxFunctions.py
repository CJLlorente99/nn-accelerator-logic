from functools import reduce
from itertools import islice
import math
import numpy as np


# Array of binary to single value
def combine(x, y):
	return 2 * x + y


def binaryArrayToSingleValue(arrayList, n=50):
	# TODO. Max number that can be represented with float64 might be higher
	res = []
	lengths = []
	# if value is bigger than what can be represented with a single uint64, return various values
	for i in range(0, len(arrayList), n):
		aux = reduce(combine, arrayList[i:i + n])
		assert aux >= 0
		res.append(aux)
		aux = len(arrayList[i:i + n])
		assert aux > 0
		lengths.append(aux)
	return res + lengths


# Single value to array of binary
def integerToBinaryArray(value, lengths):

	res = []
	j = 0
	for i in value:
		aux = []
		while i > 0:
			aux.append(i % 2)
			i //= 2
		aux.reverse()
		aux = np.pad(np.array(aux), (int(lengths[j]) - len(aux), 0))
		res.append(aux)
		j += 1
	res = [item for sublist in res for item in sublist]
	return res
