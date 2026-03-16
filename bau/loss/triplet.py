from __future__ import absolute_import

from typing import Tuple, cast
from functools import partial

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


def _angular_extrapolation_bound(dtype: torch.dtype) -> float:
	if dtype in (torch.float16, torch.bfloat16):
		return 1.0 - 1e-4
	return 1.0 - 1e-5


def _safe_acos_linear_extrapolation(x: torch.Tensor, bound: float) -> torch.Tensor:
	if not 0.0 < bound < 1.0:
		raise ValueError("bound must lie strictly between 0 and 1")

	clamped = x.clamp(min=-bound, max=bound)
	acos_clamped = torch.acos(clamped)

	bound_tensor = torch.full_like(x, bound)
	neg_bound_tensor = -bound_tensor
	eps = torch.finfo(x.dtype).eps

	upper_slope = -1.0 / torch.sqrt(torch.clamp(1.0 - bound_tensor * bound_tensor, min=eps))
	lower_slope = -1.0 / torch.sqrt(torch.clamp(1.0 - neg_bound_tensor * neg_bound_tensor, min=eps))

	upper_linear = torch.acos(bound_tensor) + (x - bound_tensor) * upper_slope
	lower_linear = torch.acos(neg_bound_tensor) + (x - neg_bound_tensor) * lower_slope

	return torch.where(x > bound_tensor, upper_linear, torch.where(x < neg_bound_tensor, lower_linear, acos_clamped))


def _stable_theta_sin_ratio(theta: torch.Tensor, threshold: float = 1e-3) -> torch.Tensor:
	theta_sq = theta * theta
	taylor = 1.0 + theta_sq / 6.0 + 7.0 * theta_sq * theta_sq / 360.0
	eps = torch.finfo(theta.dtype).eps
	exact = theta / torch.sin(theta).clamp(min=eps)
	return torch.where(theta.abs() < threshold, taylor, exact)


def finsler_drift_dist(
	x: torch.Tensor,
	y: torch.Tensor,
	alpha=None,
	identity_dim: int | None = None,
	method: str = "symmetric_trapezoidal",
) -> torch.Tensor:
	"""
	Compute Randers/Finsler distance using concatenated [identity | drift] embeddings.

	If identity_dim is provided, it defines the split point between identity and drift.
	Otherwise, the feature dimension is split evenly.
	
	Available methods:
	- "constant_drift": Option 1, Assume drift vector constant along trajectory
	- "symmetric_trapezoidal": Option 2, Symmetric Trapezoidal Approximation
	- "slerp": Option 3.1, SLERP Approximation (Geodesic Path)
	- "analytical": Option 3.2, Exact Analytical Integration of the Parallel-Transported Field
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

	# 1. x_w * y_id (MxN)
	term_xy = torch.mm(drift_x_shared, identity_y_shared.t())

	# 2. x_w * x_id (Mx1 -> MxN)
	term_xx = (drift_x_shared * identity_x_shared).sum(dim=1, keepdim=True).expand(m, n)

	# 3. y_w * y_id (Nx1 -> NxM -> MxN via transpose)
	term_yy = (drift_y_shared * identity_y_shared).sum(dim=1, keepdim=True).expand(n, m).t()

	# 4. y_w * x_id (NxM -> MxN via transpose of the matmul result)
	# Note: (y_w @ x_id.T).T is equivalent to x_id @ y_w.T
	term_yx = torch.mm(identity_x_shared, drift_y_shared.t())

	if method == "constant_drift":
		# Option 1: Assume drift vector constant along trajectory
		asymmetry = term_xy - term_xx
	elif method == "symmetric_trapezoidal":
		# Option 2: Symmetric Trapezoidal Approximation
		asymmetry = 0.5 * ((term_xy - term_xx) + (term_yy - term_yx))
	elif method == "slerp":
		# Option 3.1: SLERP Approximation (Geodesic Path)
		# Compute angular terms in float32 to avoid autocast-induced boundary hits.
		identity_x_angle = identity_x_shared.float()
		identity_y_angle = identity_y_shared.float()
		drift_x_angle = drift_x_shared.float()
		drift_y_angle = drift_y_shared.float()

		term_xy = torch.mm(drift_x_angle, identity_y_angle.t())
		term_xx = (drift_x_angle * identity_x_angle).sum(dim=1, keepdim=True).expand(m, n)
		term_yy = (drift_y_angle * identity_y_angle).sum(dim=1, keepdim=True).expand(n, m).t()
		term_yx = torch.mm(identity_x_angle, drift_y_angle.t())

		bound = _angular_extrapolation_bound(identity_x_shared.dtype)
		cos_theta = torch.mm(identity_x_angle, identity_y_angle.t())
		theta = _safe_acos_linear_extrapolation(cos_theta, bound)
		theta_sin_ratio = _stable_theta_sin_ratio(theta)

		# Discretize the Path (Riemann sum)
		K = 10
		dt = 1.0 / K
		t_steps = torch.linspace(dt / 2.0, 1.0 - dt / 2.0, steps=K, device=dist.device, dtype=theta.dtype)
		
		asymmetry = torch.zeros_like(theta)

		# Compute Trajectory, Velocity, Drift Vector Field, and Riemann Sum
		for t in t_steps:
			# Velocity coefficients
			A_t = - theta_sin_ratio * torch.cos((1.0 - t) * theta)
			B_t = theta_sin_ratio * torch.cos(t * theta)
			
			vel_overlap = (1.0 - t) * A_t * term_xx + \
						  (1.0 - t) * B_t * term_xy + \
						  t * A_t * term_yx + \
						  t * B_t * term_yy
			
			asymmetry += vel_overlap * dt
	
	elif method == "analytical":
		# Option 3.2: Exact Analytical Integration of the Parallel-Transported Field
		identity_x_angle = identity_x_shared.float()
		identity_y_angle = identity_y_shared.float()
		drift_x_angle = drift_x_shared.float()
		drift_y_angle = drift_y_shared.float()

		term_xy = torch.mm(drift_x_angle, identity_y_angle.t())
		term_yx = torch.mm(identity_x_angle, drift_y_angle.t())

		bound = _angular_extrapolation_bound(identity_x_shared.dtype)
		cos_theta = torch.mm(identity_x_angle, identity_y_angle.t())
		theta = _safe_acos_linear_extrapolation(cos_theta, bound)
		theta_sin_ratio = _stable_theta_sin_ratio(theta)

		asymmetry = 0.5 * theta_sin_ratio * (term_xy - term_yx)
	else:
		raise ValueError(f"Unknown finsler drift computation method: '{method}'")

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
		
		# Intercept and force symmetric_trapezoidal if finsler_drift_dist is used
		if isinstance(dist_func, partial) and getattr(dist_func.func, "__name__", "") == "finsler_drift_dist":
			new_kwargs = dist_func.keywords.copy()
			new_kwargs["method"] = "symmetric_trapezoidal"
			self.dist_func = partial(dist_func.func, *dist_func.args, **new_kwargs)
		elif getattr(dist_func, "__name__", "") == "finsler_drift_dist":
			self.dist_func = partial(dist_func, method="symmetric_trapezoidal")
		else:
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
