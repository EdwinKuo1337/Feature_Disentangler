class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def getAverageMeter(numLosses):
	return [AverageMeter() for _ in range(numLosses)]


def lossesUpdate(losses, batchSize, lossValues):
	for loss, value in zip(losses, lossValues):
		loss.update(value.item(), batchSize)

	return losses