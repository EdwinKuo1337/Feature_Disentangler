from __future__ import print_function, absolute_import
import numpy as np
import copy
from tqdm import tqdm


def evaluate(distmat, queryPersonIds, galleryPersonIds, maxRank=30):
	numQuery, numGallery = distmat.shape
	if numGallery < maxRank:
		maxRank = numGallery
		print("Note: number of gallery samples is quite small, got {}".format(numGallery))
	
	indices = np.argsort(distmat, axis=1)
	matches = (galleryPersonIds[indices] == queryPersonIds[:, np.newaxis]).astype(np.int32)

	# compute cmc curve for each query
	all_cmc = []
	all_AP = []
	num_valid_q = 0.

	for queryIndex in tqdm(range(numQuery), desc="Computing CMC and mAP"):
		# get query pid and camid
		q_pid = queryPersonIds[queryIndex]
		# remove gallery samples that have the same pid and camid with query
		# order = indices[queryIndex]
		# remove = (galleryPersonIds[order] == q_pid)
		# keep = np.invert(remove)

		# compute cmc curve
		# orig_cmc = matches[queryIndex][keep] # binary vector, positions with value 1 are correct matches
		orig_cmc = matches[queryIndex]
		# if not np.any(orig_cmc):
		#     # this condition is true when query identity does not appear in gallery
		#     continue
		cmc = orig_cmc.cumsum()
		cmc[cmc > 1] = 1
		all_cmc.append(cmc[:maxRank])
		num_valid_q += 1.

		# compute average precision
		# reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
		num_rel = orig_cmc.sum()
		tmp_cmc = orig_cmc.cumsum()
		tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
		tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
		AP = tmp_cmc.sum() / num_rel
		all_AP.append(AP)

	assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

	all_cmc = np.asarray(all_cmc).astype(np.float32)
	all_cmc = all_cmc.sum(0) / num_valid_q
	mAP = np.mean(all_AP)
	return all_cmc, mAP