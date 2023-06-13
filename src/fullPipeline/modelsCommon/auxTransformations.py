# Transform to black (1) and white (0)
class ToBlackAndWhite(object):

	def __call__(self, sample):
		sample[sample > 0] = 1
		sample[sample == 0] = 0
		return sample


# Transform to black (1) and white (-1)
class ToSign(object):

	def __call__(self, sample):
		sample[sample == 0] = -1
		return sample