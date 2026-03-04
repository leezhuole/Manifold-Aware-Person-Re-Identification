from __future__ import absolute_import

from typing import Tuple, cast

import torch
from torch import nn
import torch.nn.functional as F


class AlphaParameter(nn.Module):
	"""
	Learnable global alpha with centered parameterization.

	Mapping:
		alpha = 2 * max_alpha * |sigmoid(raw / temperature) - 0.5|

	This ensures alpha = 0 at raw = 0 (Euclidean baseline), while still
	keeping alpha within [0, max_alpha].
	"""

	def __init__(self, init: float = 0.0, max_alpha: float = 1.0, temperature: float = 1.0):
		super().__init__()
		if max_alpha <= 0 or max_alpha > 1:
			raise ValueError("max_alpha must be positive and in (0, 1]")
		if temperature <= 0:
			raise ValueError("temperature must be positive")

		self.max_alpha = float(max_alpha)
		self.temperature = float(temperature)

		# effective_init = init if init > 1e-3 else 1e-3
		# self.raw_alpha = nn.Parameter(self._init_raw_alpha(effective_init))
		self.raw_alpha = nn.Parameter(self._init_raw_alpha(init))

	def value(self) -> torch.Tensor:
		# Centered, symmetric parameterization
		sig = torch.sigmoid(self.raw_alpha / self.temperature)
		val = 2.0 * self.max_alpha * torch.abs(sig - 0.5)

		# Simple monotonic sigmoid (Range: [0, max_alpha]))
		# val = self.max_alpha * torch.sigmoid(self.raw_alpha / self.temperature) 	
		return val
	
	def raw_value(self) -> torch.Tensor:
		return self.raw_alpha

	def _init_raw_alpha(self, init: float) -> torch.Tensor:
		if init < 0.0 or init >= 1.0:
			raise ValueError("Initial alpha must be non-negative and less than 1.0")

		init_clamped = min(max(float(init), 0.0), self.max_alpha)
		eps = 1e-6

		# Centered, symmetric parameterization
		y_norm = init_clamped / (2.0 * self.max_alpha)
		target_sigmoid = y_norm + 0.5
		target_sigmoid = min(max(target_sigmoid, 0.5 + eps), 1.0 - eps)
		raw = (torch.log(torch.tensor(target_sigmoid)) - torch.log(torch.tensor(1.0 - target_sigmoid)))

		# Simple monotonic sigmoid (Range: [0, max_alpha]))
		# p = init_clamped / self.max_alpha
		# raw = torch.log(torch.tensor(p)) - torch.log(torch.tensor(1.0 - p))
		return raw * self.temperature


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
		if isinstance(alpha, torch.Tensor) and alpha.device != dist.device:
			alpha = alpha.to(device=dist.device)
		y_last = y[:, -1].unsqueeze(0) 	# (1, n)
		x_last = x[:, -1].unsqueeze(1) 	# (m, 1)
		finsler_term = (y_last - x_last) * alpha
		assert finsler_term.shape == dist.shape, "Finsler term shape mismatch"
		dist = dist + finsler_term

	return dist


def finsler_drift_dist(
	x: torch.Tensor,
	y: torch.Tensor,
	alpha=None,
	identity_dim: int | None = None,
) -> torch.Tensor:
	"""
	Compute Randers/Finsler distance using concatenated [identity | drift] embeddings.

	If identity_dim is provided, it defines the split point between identity and drift.
	Otherwise, the feature dimension is split evenly.
	"""
	if x.size(1) != y.size(1):
		raise ValueError("finsler_drift_dist requires x and y to have the same feature dimension")

	feature_dim = x.size(1)
	if identity_dim is None:
		if feature_dim % 2 != 0:
			raise ValueError("finsler_drift_dist requires an even feature dimension when identity_dim is None")
		identity_dim = feature_dim // 2
	if identity_dim <= 0 or identity_dim >= feature_dim:
		raise ValueError("identity_dim must be in (0, feature_dim) for finsler_drift_dist")

	m, n = x.size(0), y.size(0)
	identity_x = x[:, :identity_dim]
	drift_x = x[:, identity_dim:]
	identity_y = y[:, :identity_dim]
	drift_y = y[:, identity_dim:]
	if drift_x.size(1) != drift_y.size(1):
		raise ValueError("Drift components of x and y must have the same dimension")
 
	xx = torch.pow(identity_x, 2).sum(1, keepdim=True).expand(m, n)
	yy = torch.pow(identity_y, 2).sum(1, keepdim=True).expand(n, m).t()
	dist = xx + yy
	dist = dist - 2 * identity_x @ identity_y.t()
	dist = dist.clamp(min=1e-12).sqrt()

	shared_dim = min(drift_x.size(1), identity_x.size(1))
	if shared_dim <= 0:
		raise ValueError("finsler_drift_dist requires a non-zero drift component")
	drift_x_shared = drift_x[:, :shared_dim]
	drift_y_shared = drift_y[:, :shared_dim]
	
	identity_x_shared = identity_x[:, :shared_dim]
	identity_y_shared = identity_y[:, :shared_dim]

	# Option 1: Assume drift vector constant along trajectory
	#Note: This distance function discards the rest of the dimensions that aren't shared
	# term_a = torch.mm(drift_x_shared, identity_y_shared.t())
	# term_b = (drift_x_shared * identity_x_shared).sum(dim=1, keepdim=True).expand(m, n)
	# asymmetry = term_a - term_b

	# Option 2: Symmetric Trapezoidal Approximation
	# 1. x_w * y_id (MxN)
	term_xy = torch.mm(drift_x_shared, identity_y_shared.t())

	# 2. x_w * x_id (Mx1 -> MxN)
	term_xx = (drift_x_shared * identity_x_shared).sum(dim=1, keepdim=True).expand(m, n)

	# 3. y_w * y_id (Nx1 -> NxM -> MxN via transpose)
	term_yy = (drift_y_shared * identity_y_shared).sum(dim=1, keepdim=True).expand(n, m).t()

	# 4. y_w * x_id (NxM -> MxN via transpose of the matmul result)
	# Note: (y_w @ x_id.T).T is equivalent to x_id @ y_w.T
	term_yx = torch.mm(identity_x_shared, drift_y_shared.t())

	asymmetry = 0.5 * ((term_xy - term_xx) + (term_yy - term_yx))

	# Option 3: Spherical Trapezoidal Approximation (Geodesic Path)
	# Removes the Euclidean straight-line assumption. Since identity features are L2 normalized, 
	# the path is a great circle arc. We project the path tangents onto the hypersphere.

	# # 1. x_w * y_id (MxN)
	# term_xy = torch.mm(drift_x_shared, identity_y_shared.t())

	# # 2. x_w * x_id (Mx1 -> MxN)
	# term_xx = (drift_x_shared * identity_x_shared).sum(dim=1, keepdim=True).expand(m, n)

	# # 3. y_w * y_id (Nx1 -> NxM -> MxN via transpose)
	# term_yy = (drift_y_shared * identity_y_shared).sum(dim=1, keepdim=True).expand(n, m).t()

	# # 4. y_w * x_id (NxM -> MxN via transpose of the matmul result)
	# # Note: (y_w @ x_id.T).T is equivalent to x_id @ y_w.T
	# term_yx = torch.mm(identity_x_shared, drift_y_shared.t())
	
	# cos_sim = torch.mm(identity_x_shared, identity_y_shared.t())

	# # Tangent at x pointing to y: y - cos(theta)*x
	# # Tangent at y coming from x: y*cos(theta) - x
	# asymmetry = 0.5 * ((term_xy - cos_sim * term_xx) + (cos_sim * term_yy - term_yx))

	# --- PROPOSED SCALING TEST ---   
    # Uncomment ONE of the following scaling methods to test:
    # # 1. The N/S Test (Thomas's suggestion to prove the artifact)
	scaling_factor = identity_x.size(1) / shared_dim 
    
    # # 2. Dimension-Invariant Normalization (To fairly compare dimensions)
	# # import math
	# # scaling_factor = 1.0 / math.sqrt(shared_dim) 
	
	asymmetry = asymmetry * scaling_factor
	# -----------------------------

	return dist + asymmetry


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

	def __init__(self, margin=None, normalize_feature=False, manifold=None, alpha=None, dist_func=euclidean_dist, bidirectional=False):
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
		self.dist_func = dist_func
		self.bidirectional = bidirectional
		if margin is not None:
			self.margin_loss = nn.MarginRankingLoss(margin=float(margin)).cuda()
		else:
			self.margin_loss = nn.SoftMarginLoss()

	def forward(self, emb, label, alpha=None):
		"""
		
		emb: (N, D) embedding features
		label: (N,) labels corresponding to the embeddings
		
		Note that we define the input argument alpha here as the override value for Rander's alpha.
		Intended to be used by the trainer each step, which is passed into forward() during training.

		self.alpha is used as the default value if no override is provided (e.g., fixed or module-provided alpha).

		"""
		# Sanity checks
		alpha_value = self.alpha if alpha is None else alpha
		if self.manifold is not None and alpha_value is not None:
			raise ValueError("Finsler spaces is not supported for non-euclidean manifolds.")

		# Case 1: Finsler spaces and Euclidean spaces
		if self.manifold is None:
			if self.normalize_feature:
				# equal to cosine similarity
				emb = F.normalize(emb)
			mat_dist = self.dist_func(x=emb, y=emb, alpha=alpha_value)
		
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

		if self.bidirectional:
			dist_ap_inv, dist_an_inv = cast(
				Tuple[torch.Tensor, torch.Tensor],
				_batch_hard(mat_dist.t(), mat_sim.t()),
			)
			assert dist_an_inv.size(0)==dist_ap_inv.size(0)
			if self.margin is not None:
				loss_inv = self.margin_loss(dist_an_inv, dist_ap_inv, y)
			else:
				loss_inv = self.margin_loss(dist_an_inv - dist_ap_inv, y)
			loss = 0.5 * (loss + loss_inv)
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
