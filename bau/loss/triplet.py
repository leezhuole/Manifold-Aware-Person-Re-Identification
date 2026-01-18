from __future__ import absolute_import

from typing import Tuple, cast

import torch
from torch import nn
import torch.nn.functional as F


def euclidean_dist(x: torch.Tensor, y: torch.Tensor, alpha=None) -> torch.Tensor:
	"""
	Compute euclidean distance between x and y. This distance function transforms to the canonical 
	Randers distance is alpha is defined as not None. 
	
	:param x: Input tensor 1 (m, D)
	:param y: Input tensor 2 (n, D)
	:param alpha: Rander's alpha value for Finsler spaces 

	"""
	m, n = x.size(0), y.size(0)
	d = x.size(1)
	assert d == y.size(1) 	# Sanity Check
	xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
	yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
	dist = xx + yy
	dist = dist -2 * x@y.t()
	dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
	
	if alpha is not None:
		assert alpha < 1 		# Sanity Check 
		y_last = y[:, -1].unsqueeze(0) 	# (1, n)
		x_last = x[:, -1].unsqueeze(1) 	# (m, 1)
		finsler_term = (y_last - x_last) * alpha
		assert finsler_term.shape == dist.shape, "Finsler term shape mismatch"
		
		# DEBUG: Sanity check for alpha=0
		if alpha == 0:
			# If everything is correct, dist + finsler_term should be identical to dist, or at least
			# within a tiny float tolerarnce.

			# 1. Check for NaNs (e.g., if inputs had Inf, Inf * 0 = NaN)
			if torch.isnan(finsler_term).any():
				print(f"[DEBUG CRITICAL] alpha=0 but Finsler term contains NaNs! Check inputs for Infinity.")

			# 2. Check for values > 0 (should all be zeros)
			if not torch.allclose(finsler_term, torch.zeros_like(finsler_term)):
				max_val = finsler_term.abs().max().item()
				print(f"[DEBUG CRITICAL] alpha=0 but Finsler term is not zero! Max absolute value: {max_val}")

			# 3. Direct comparison of before/after distance matrices
			new_dist = dist + finsler_term
			diff = (dist - new_dist).abs()
			if diff.max() > 1e-7:  # Tolerance for float32 additions
				print(f"[DEBUG CRITICAL] alpha=0 distance matrix mismatch! Max diff: {diff.max().item()}")
		
		print(f"[DEBUG] alpha=0 sanity check completed.")
		dist = dist + finsler_term

	return dist


def cosine_dist(x, y):
	bs1, bs2 = x.size(0), y.size(0)
	frac_up = torch.matmul(x, y.transpose(0, 1))
	frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
	            (torch.sqrt(torch.sum(torch.pow(y, 2), 1))).view(1, bs2).repeat(bs1, 1)
	cosine = frac_up / frac_down
	return 1-cosine


def _batch_hard(mat_distance, mat_similarity, indice=False):
	sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1, descending=True)
	hard_p = sorted_mat_distance[:, 0]
	hard_p_indice = positive_indices[:, 0]
	sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
	hard_n = sorted_mat_distance[:, 0]
	hard_n_indice = negative_indices[:, 0]
	if(indice):
		return hard_p, hard_n, hard_p_indice, hard_n_indice
	return hard_p, hard_n


# def _batch_hard_neg(mat_distance, mat_similarity, indice=False):
# 	sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999999.) * (mat_similarity), dim=1, descending=True)
# 	hard_p = sorted_mat_distance[:, 0]
# 	hard_p_indice = positive_indices[:, 0]
# 	sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999999.) * (1-mat_similarity), dim=1, descending=False)
# 	hard_n = sorted_mat_distance[:, 0]
# 	hard_n_indice = negative_indices[:, 0]
# 	if(indice):
# 		return hard_p, hard_n, hard_p_indice, hard_n_indice
# 	return hard_p, hard_n


class TripletLoss(nn.Module):
	'''
	Compute Triplet loss augmented with Batch Hard
	Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
	'''

	def __init__(self, margin=None, normalize_feature=False, manifold=None, alpha=None):
		"""
		Docstring for __init__
		
		:param margin: margin value for MarginRankingLoss
		:param normalize_feature: whether to normalize feature to unit length
		:param manifold: whether the manifold is euclidean or hyperbolic
		:param alpha: rander's alpha value for finsler spaces
		"""
		super(TripletLoss, self).__init__()
		self.margin = margin
		self.normalize_feature = normalize_feature
		self.manifold = manifold
		self.alpha = alpha
		if margin is not None:
			self.margin_loss = nn.MarginRankingLoss(margin=float(margin)).cuda()
		else:
			self.margin_loss = nn.SoftMarginLoss()

	def forward(self, emb, label):
		# Sanity checks
		if self.manifold is not None and self.alpha is not None:
			raise ValueError("Finsler spaces is not supported for non-euclidean manifolds.")

		# Case 1: Finsler spaces and Euclidean spaces
		if self.manifold is None:
			if self.normalize_feature:
				# equal to cosine similarity
				emb = F.normalize(emb)
			mat_dist = euclidean_dist(x=emb, y=emb, alpha=self.alpha)
		
		# Case 2: Non-euclidean manifold		
		else:
			mat_dist = self.manifold.dist(
				emb.unsqueeze(1), emb.unsqueeze(0), dim=-1
			)

			
		assert mat_dist.size(0) == mat_dist.size(1)
		N = mat_dist.size(0)
		mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()

		dist_ap, dist_an = cast(
			Tuple[torch.Tensor, torch.Tensor],
			_batch_hard(mat_dist, mat_sim),
		)
		assert dist_an.size(0)==dist_ap.size(0)
		y = torch.ones_like(dist_ap)
		
		if self.margin is not None:
			loss = self.margin_loss(dist_an, dist_ap, y)
		else:
			loss = self.margin_loss(dist_an - dist_ap, y)
		# prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
		return loss # , prec


# class SoftTripletLoss(nn.Module):
# 	def __init__(self, margin=0.0):
# 		super(SoftTripletLoss, self).__init__()
# 		self.margin = margin

# 	def forward(self, emb, label):
# 		mat_dist = euclidean_dist(emb, emb)
# 		assert mat_dist.size(0) == mat_dist.size(1)
# 		N = mat_dist.size(0)
# 		mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()

# 		dist_ap, dist_an, ap_idx, an_idx = cast(
# 			Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
# 			_batch_hard(mat_dist, mat_sim, indice=True),
# 		)
# 		assert dist_an.size(0) == dist_ap.size(0)
# 		triple_dist = torch.stack((dist_ap, dist_an), dim=1)
# 		triple_dist = F.log_softmax(triple_dist, dim=1)
# 		loss = (- self.margin * triple_dist[:, 0] - (1 - self.margin) * triple_dist[:, 1]).mean()

# 		return loss
