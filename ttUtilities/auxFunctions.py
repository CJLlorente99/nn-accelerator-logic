from functools import reduce
from itertools import islice
import math
import numpy as np


# Array of binary to single value
def combine(x, y):
	return 2 * x + y


def binaryArrayToSingleValue(arrayList):
	n = 64  # max bits in uint64 representation (number of binary activation)
	res = []
	# if value is bigger than what can be represented with a single uint64, return various values
	for i in range(0, len(arrayList), n):
		res.append(reduce(combine, arrayList[i:i + n]))

	return res


# Single value to array of binary
def integerToBinaryArray(value):

	res = []
	for i in value:
		aux = []
		while i > 0:
			aux.append(i % 2)
			i //= 2
		aux.reverse()
		res.append(aux)
	res = [item for sublist in res for item in sublist]
	return res
