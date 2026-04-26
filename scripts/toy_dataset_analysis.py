"""
Toy dataset drift vector analysis: extract embeddings, compute metrics, generate figures.

Validates three claims:
  C1: Identity vector z^id is invariant to corruption severity.
  C2: Drift vector z^omega magnitude correlates monotonically with severity.
  C3: Compare retrieval under Euclidean BAU (resnet50 checkpoint) vs Finsler BAU (resnet50_finsler + d_F).

Retrieval mAP compares *two independently trained checkpoints*: embeddings from
``--euclidean-checkpoint`` (BAU ``resnet50``) ranked with Euclidean distance vs.
embeddings from ``--finsler-checkpoint`` (``resnet50_finsler``) ranked with
``finsler_drift_dist`` on the full [identity|drift] vector.

Usage (v3.0 dataset — two source crops, asymmetry diagnostics enabled automatically):
    python scripts/toy_dataset_analysis.py \
        --finsler-checkpoint /home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification/logs/finsler_primary/job_1521403_primary_unified_1c_w0.1_driftInst/best.pth \
        --euclidean-checkpoint logs/AGReIDv2_sweep/job_1502418_dt_cuhk03_EucBaseline/best.pth \
        --dataset-dir /home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification/examples/data/ToyCorruption \
        --output-dir results/toy_analysis_v3 \
        --viz-tsne

Low-dimensional drift plots (optional):
    --viz-num-identities 15 --viz-seed 42 \\
    --fig-omega-identity-pca fig_omega_identity_pca.pdf \\
    --viz-tsne --fig-omega-tsne fig_omega_tsne.pdf

v3.0 asymmetry diagnostic options (all optional; defaults shown):
    --omega-cos-transport ambient   # or "parallel" (Pennec 2006 sphere transport)
    --projection-report B_primary   # or "all" (includes P_A and P_C in JSON)
    --projection-shuffle-reps 200   # permutation null iterations
    --skip-v3-diagnostics           # revert to v2.0-only output on v3 data

v3.0 filename convention (`scripts/generate_toy_dataset.py`): ``{pid:04d}_c{source_idx}s{sev+1}_...jpg``.
Under v3.0, cam_id encodes the **source crop index** (1 or 2); severity is in the Market-1501 seq field.
v2.0 fallback (cam_id = severity + 1) is auto-detected via the metadata.json `toy_dataset_version`.

Asymmetry diagnostics (Points 1a/1b/2/3) spelled out in
`changelogs/toy_dataset_asymmetry_diagnostics.md` and invoked from `main()` when v3.0 data is present.
"""

import argparse
import json
import os
import os.path as osp
import re
import sys
from collections import OrderedDict, defaultdict
from functools import partial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sklearn
from torch.utils.data import DataLoader
from torchvision import transforms as T

_SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, osp.join(_SCRIPT_DIR, ".."))
from bau.models.model import resnet50_finsler, resnet50
from bau.loss.triplet import finsler_drift_dist, euclidean_dist
from bau.evaluation_metrics import cmc, mean_ap

from toy_paper_figures import (
    copy_paper_outputs,
    make_corruption_strip,
    make_cosine_block_bar,
    make_cross_severity_heatmap,
    make_drift_absorption_plot,
    make_drift_alignment_plot,
    make_retrieval_mAP_plot,
)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

class ToyDataset(torch.utils.data.Dataset):
    """Minimal dataset that returns (img_tensor, fname, pid, source_idx, severity).

    Supports v3.0 filename convention ``{pid}_c{source_idx}s{sev+1}_...jpg``
    (two-field parse: cam encodes source, seq encodes severity+1). Falls back to the
    v2.0 layout ``{pid}_c{cam_id}s1_...jpg`` (cam encodes severity+1, seq fixed at 1)
    for backward-compatible runs on unregenerated datasets.
    """

    _PATTERN_V3 = re.compile(r"(\d+)_c(\d+)s(\d+)")

    def __init__(self, image_dir, transform=None, dataset_version="3.0"):
        self.image_dir = image_dir
        self.transform = transform
        self.dataset_version = str(dataset_version)
        v3 = self.dataset_version.startswith("3")
        self.samples = []
        for fname in sorted(os.listdir(image_dir)):
            if not fname.endswith(".jpg"):
                continue
            m = self._PATTERN_V3.search(fname)
            if not m:
                continue
            pid = int(m.group(1))
            first = int(m.group(2))
            second = int(m.group(3))
            if v3:
                source_idx = first
                severity = second - 1  # seq field encodes severity+1
            else:
                source_idx = 1
                severity = first - 1  # legacy: cam field encoded severity+1
            self.samples.append((fname, pid, source_idx, severity))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, pid, source_idx, severity = self.samples[idx]
        img = Image.open(osp.join(self.image_dir, fname)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, fname, pid, source_idx, severity


def _detect_dataset_version(dataset_dir: str) -> str:
    """Read toy_dataset_version from metadata.json if present; default to 3.0."""
    meta_path = osp.join(dataset_dir, "metadata.json")
    if osp.isfile(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            return str(meta.get("dataset", {}).get("toy_dataset_version", "3.0"))
        except (json.JSONDecodeError, OSError):
            pass
    return "3.0"


def get_test_transform(height=256, width=128):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return T.Compose([T.Resize((height, width), interpolation=3), T.ToTensor(), normalizer])


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_finsler_model(checkpoint_path, num_classes=13726, num_domains=3):
    """Load the best Finsler model from a training checkpoint."""
    model = resnet50_finsler(
        num_classes=num_classes,
        pretrained=False,
        use_drift_in_eval=True,
        memory_bank_mode="full",
        drift_dim=2048,
        drift_method="symmetric_trapezoidal",
        drift_conditioning="instance",
        num_domains=num_domains,
        domain_embed_dim=64,
        infer_domain_conditioning=True,
        domain_temperature=1.0,
        domain_residual_scale=0.1,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    # Strip 'module.' prefix from DataParallel wrapping
    new_sd = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_sd[new_key] = v

    model.load_state_dict(new_sd, strict=False)
    model.eval()
    model.cuda()
    return model


def load_bau_euclidean_resnet50(checkpoint_path, num_classes=13726, in_stages=(1, 2, 3)):
    """Load standard BAU Euclidean backbone (``resnet50``), not ``resnet50_finsler``."""
    model = resnet50(num_classes=num_classes, pretrained=False, manifold=None, in_stages=in_stages)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    new_sd = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_sd[new_key] = v

    model.load_state_dict(new_sd, strict=False)
    model.eval()
    model.cuda()
    return model


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(model, data_loader):
    """Extract features and metadata from a data loader.

    ``meta[fname]`` keys:
      - ``pid`` (int), ``severity`` (int, 0..4), ``source_idx`` (int, 1..num_sources)
      - ``camid`` is set to ``source_idx`` so that downstream mAP / CMC filtering
        uses source-crop identity as the camera, matching the v3.0 filename contract.
    """
    model.eval()
    features = OrderedDict()
    meta = OrderedDict()

    with torch.no_grad():
        for batch in data_loader:
            imgs, fnames, pids, source_idxs, severities = batch
            imgs = imgs.cuda()
            outputs = model(imgs)
            outputs = outputs.data.cpu()
            for fname, output, pid, source_idx, severity in zip(
                fnames, outputs, pids, source_idxs, severities
            ):
                features[fname] = output
                meta[fname] = {
                    "pid": int(pid),
                    "source_idx": int(source_idx),
                    "camid": int(source_idx),
                    "severity": int(severity),
                }

    return features, meta


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def compute_drift_identity_metrics(features, meta, identity_dim=2048):
    """Compute per-severity drift magnitude and identity distance from clean reference."""
    severity_drift_norms = defaultdict(list)
    severity_identity_dists = defaultdict(list)

    # Group features by PID
    pid_features = defaultdict(dict)
    for fname, feat in features.items():
        m = meta[fname]
        pid_features[m["pid"]][m["severity"]] = feat

    for pid, sev_feats in pid_features.items():
        if 0 not in sev_feats:
            continue
        clean_identity = sev_feats[0][:identity_dim]
        clean_identity_norm = F.normalize(clean_identity.unsqueeze(0), dim=1).squeeze(0)

        for sev, feat in sev_feats.items():
            identity = feat[:identity_dim]
            drift = feat[identity_dim:]

            drift_norm = torch.norm(drift, p=2).item()
            severity_drift_norms[sev].append(drift_norm)

            identity_norm = F.normalize(identity.unsqueeze(0), dim=1).squeeze(0)
            identity_dist = torch.norm(identity_norm - clean_identity_norm, p=2).item()
            severity_identity_dists[sev].append(identity_dist)

    results = {}
    for sev in sorted(severity_drift_norms.keys()):
        dn = np.array(severity_drift_norms[sev])
        di = np.array(severity_identity_dists[sev])
        results[sev] = {
            "drift_norm_mean": float(np.mean(dn)),
            "drift_norm_std": float(np.std(dn)),
            "identity_dist_mean": float(np.mean(di)),
            "identity_dist_std": float(np.std(di)),
        }

    # Spearman correlation: severity vs drift magnitude
    all_sevs = []
    all_drifts = []
    for sev in sorted(severity_drift_norms.keys()):
        for d in severity_drift_norms[sev]:
            all_sevs.append(sev)
            all_drifts.append(d)
    rho, pval = stats.spearmanr(all_sevs, all_drifts)
    results["spearman_rho"] = float(rho)
    results["spearman_pval"] = float(pval)

    return results


def compute_retrieval_metrics(
    features,
    meta,
    identity_dim=2048,
    dist_func=None,
    use_full_embedding=False,
    baseline_euclidean_resnet50=False,
):
    """Compute mAP and Rank-1 for query (severity 0) vs gallery (each severity 1-4).

    Retained for backward compatibility with the v2.0 reporting. Under v3.0 the
    ``meta[fname]["camid"]`` equals the source index, so cmc/mean_ap's same-(pid, cam)
    filter automatically restricts each query to its *cross-source* gallery — this
    function therefore reports the ``sigma_q=0`` row of the Point 1a cross-source matrix.

    If ``baseline_euclidean_resnet50`` is True, ``features`` are 2048-d ``resnet50`` BAU
    embeddings; ranking uses Euclidean distance on the full vector. Otherwise features
    are Finsler 4096-d and behavior follows ``use_full_embedding`` / ``dist_func``.
    """
    # Separate query (severity 0) and gallery (severity > 0)
    query_items = []
    gallery_by_severity = defaultdict(list)

    for fname, feat in features.items():
        m = meta[fname]
        if m["severity"] == 0:
            query_items.append((fname, m["pid"], m["camid"]))
        else:
            gallery_by_severity[m["severity"]].append((fname, m["pid"], m["camid"]))

    results = {}
    for sev in sorted(gallery_by_severity.keys()):
        gallery_items = gallery_by_severity[sev]

        # Build feature tensors
        q_feats = torch.stack([features[f] for f, _, _ in query_items])
        g_feats = torch.stack([features[f] for f, _, _ in gallery_items])

        if baseline_euclidean_resnet50:
            q_id = F.normalize(q_feats, dim=1)
            g_id = F.normalize(g_feats, dim=1)
            distmat = euclidean_dist(q_id, g_id)
        elif use_full_embedding and dist_func is not None:
            distmat = dist_func(q_feats, g_feats, alpha=None)
        else:
            # Euclidean on identity slice (Finsler checkpoint, identity-only)
            q_id = F.normalize(q_feats[:, :identity_dim], dim=1)
            g_id = F.normalize(g_feats[:, :identity_dim], dim=1)
            distmat = euclidean_dist(q_id, g_id)

        distmat = distmat.cpu().numpy()
        q_pids = np.array([pid for _, pid, _ in query_items])
        g_pids = np.array([pid for _, pid, _ in gallery_items])
        q_cams = np.array([cam for _, _, cam in query_items])
        g_cams = np.array([cam for _, _, cam in gallery_items])

        mAP = mean_ap(distmat, q_pids, g_pids, q_cams, g_cams)

        cmc_scores = cmc(
            distmat, q_pids, g_pids, q_cams, g_cams,
            separate_camera_set=False, single_gallery_shot=False, first_match_break=True,
        )
        rank1 = cmc_scores[0] if len(cmc_scores) > 0 else 0.0

        results[sev] = {"mAP": float(mAP), "Rank1": float(rank1)}

    return results


def compute_retrieval_topk(
    features,
    meta,
    identity_dim=2048,
    dist_func=None,
    use_full_embedding=False,
    top_k=5,
    baseline_euclidean_resnet50=False,
):
    """Return per-query top-k retrieval lists for qualitative visualization."""
    query_items = []
    gallery_items = []

    for fname, feat in features.items():
        m = meta[fname]
        if m["severity"] == 0:
            query_items.append((fname, m["pid"], m["camid"]))
        else:
            gallery_items.append((fname, m["pid"], m["camid"]))

    if not query_items or not gallery_items:
        return {}

    q_feats = torch.stack([features[f] for f, _, _ in query_items])
    g_feats = torch.stack([features[f] for f, _, _ in gallery_items])

    if baseline_euclidean_resnet50:
        q_id = F.normalize(q_feats, dim=1)
        g_id = F.normalize(g_feats, dim=1)
        distmat = euclidean_dist(q_id, g_id)
    elif use_full_embedding and dist_func is not None:
        distmat = dist_func(q_feats, g_feats, alpha=None)
    else:
        q_id = F.normalize(q_feats[:, :identity_dim], dim=1)
        g_id = F.normalize(g_feats[:, :identity_dim], dim=1)
        distmat = euclidean_dist(q_id, g_id)

    distmat = distmat.cpu().numpy()
    results = {}
    for i, (qf, qpid, qcam) in enumerate(query_items):
        dists = distmat[i]
        sorted_idx = np.argsort(dists)[:top_k]
        topk = []
        for j in sorted_idx:
            gf, gpid, gcam = gallery_items[j]
            topk.append({
                "fname": gf,
                "pid": int(gpid),
                "camid": int(gcam),
                "severity": int(meta[gf]["severity"]),
                "source_idx": int(meta[gf].get("source_idx", 1)),
                "dist": float(dists[j]),
                "correct": int(gpid) == int(qpid),
            })
        results[qf] = {"query_pid": int(qpid), "topk": topk}

    return results


# ---------------------------------------------------------------------------
# Low-dimensional drift / identity visualization (PCA, optional t-SNE)
# ---------------------------------------------------------------------------

def subsample_pids_with_full_severity(meta, num_identities, seed):
    """Choose PIDs with full severity coverage for at least source_idx=1.

    Under v3.0 the dataset has 2 sources per PID; for visual PCA panels we only require
    source 1 to cover severities 0..4 (source 2 is optional). This keeps the legacy
    PCA figure unchanged while tolerating any single-source dropouts.
    """
    pid_src_sev = defaultdict(lambda: defaultdict(set))
    for m in meta.values():
        pid_src_sev[m["pid"]][m.get("source_idx", 1)].add(int(m["severity"]))
    eligible = []
    need = {0, 1, 2, 3, 4}
    for pid in sorted(pid_src_sev.keys()):
        if need.issubset(pid_src_sev[pid].get(1, set())):
            eligible.append(pid)
    if not eligible:
        return []
    rng = np.random.RandomState(int(seed))
    k = min(int(num_identities), len(eligible))
    chosen = rng.choice(eligible, size=k, replace=False).tolist()
    return sorted(chosen)


def gather_identity_drift_arrays(features, meta, pids, identity_dim=2048, source_idx=1):
    """Stack identity and drift slices for selected PIDs × severities 0..4.

    ``source_idx`` restricts to one source (default 1) so that the PCA panel still
    shows single-source severity trajectories (5 points per PID). Under v2.0 data the
    default ``source_idx=1`` matches the implicit single-source contract.
    """
    pid_feats = defaultdict(dict)
    for fname, feat in features.items():
        m = meta[fname]
        if int(m.get("source_idx", 1)) != int(source_idx):
            continue
        pid_feats[m["pid"]][m["severity"]] = feat

    z_id_list, omega_list, sev_list, pid_list = [], [], [], []
    for pid in pids:
        for s in range(5):
            if s not in pid_feats[pid]:
                raise KeyError(f"Missing severity {s} for pid {pid} (source {source_idx})")
            feat = pid_feats[pid][s]
            z_id_list.append(feat[:identity_dim].cpu().numpy())
            omega_list.append(feat[identity_dim:].cpu().numpy())
            sev_list.append(s)
            pid_list.append(pid)

    return (
        np.stack(z_id_list, axis=0),
        np.stack(omega_list, axis=0),
        np.array(sev_list, dtype=np.int64),
        np.array(pid_list, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Asymmetry diagnostics (Points 1a / 1b / 2 / 3) — see
# changelogs/toy_dataset_asymmetry_diagnostics.md for the mathematical contract.
# ---------------------------------------------------------------------------

_SEVERITIES = (0, 1, 2, 3, 4)


def _group_by_pid_source_severity(features, meta):
    """Return nested dict ``{pid: {source_idx: {severity: (fname, feat)}}}``."""
    out: dict = defaultdict(lambda: defaultdict(dict))
    for fname, feat in features.items():
        m = meta[fname]
        out[m["pid"]][int(m.get("source_idx", 1))][int(m["severity"])] = (fname, feat)
    return out


def _score_distances(
    q_feats: torch.Tensor,
    g_feats: torch.Tensor,
    *,
    identity_dim: int,
    scoring: str,
    dist_func=None,
) -> np.ndarray:
    """Score a block of queries vs gallery under one of three modes.

    ``scoring='dE_full'``      — Euclidean distance on L2-normalised full vectors
                                 (Euclidean BAU resnet50 baseline, 2048-d features).
    ``scoring='dE_identity'``  — Euclidean distance on L2-normalised identity slice
                                 of Finsler checkpoint features (direction-only control).
    ``scoring='dF_midpoint'``  — Randers midpoint-drift distance (paper eq. 4) on
                                 full 4096-d Finsler features via ``dist_func``.
    """
    if scoring == "dE_full":
        q = F.normalize(q_feats, dim=1)
        g = F.normalize(g_feats, dim=1)
        return euclidean_dist(q, g).cpu().numpy()
    if scoring == "dE_identity":
        q = F.normalize(q_feats[:, :identity_dim], dim=1)
        g = F.normalize(g_feats[:, :identity_dim], dim=1)
        return euclidean_dist(q, g).cpu().numpy()
    if scoring == "dF_midpoint":
        if dist_func is None:
            raise ValueError("scoring='dF_midpoint' requires dist_func (finsler_drift_dist partial)")
        return dist_func(q_feats, g_feats, alpha=None).cpu().numpy()
    raise ValueError(f"Unknown scoring: {scoring}")


def _cross_source_cell(
    features,
    meta,
    *,
    identity_dim: int,
    scoring: str,
    dist_func,
    s_q: int,
    sigma_q: int,
    s_g: int,
    sigma_g: int,
) -> dict:
    """Compute mAP / Rank-1 on queries (s_q, sigma_q) vs gallery (s_g, sigma_g).

    Uses ``camid`` = source_idx so that the cmc/mean_ap same-(pid,cam) filter does NOT
    drop true matches when ``s_q != s_g`` and DOES drop them when ``s_q == s_g``
    (standard ReID cross-camera evaluation).
    """
    q_items, g_items = [], []
    for fname, feat in features.items():
        m = meta[fname]
        src = int(m.get("source_idx", 1))
        sev = int(m["severity"])
        if src == s_q and sev == sigma_q:
            q_items.append((fname, m["pid"], src))
        if src == s_g and sev == sigma_g:
            g_items.append((fname, m["pid"], src))
    if not q_items or not g_items:
        return {"mAP": float("nan"), "Rank1": float("nan"), "n_queries": 0, "n_gallery": 0}

    q_feats = torch.stack([features[f] for f, _, _ in q_items])
    g_feats = torch.stack([features[f] for f, _, _ in g_items])
    distmat = _score_distances(
        q_feats, g_feats, identity_dim=identity_dim, scoring=scoring, dist_func=dist_func
    )

    q_pids = np.array([pid for _, pid, _ in q_items])
    g_pids = np.array([pid for _, pid, _ in g_items])
    q_cams = np.array([c for _, _, c in q_items])
    g_cams = np.array([c for _, _, c in g_items])

    mAP = mean_ap(distmat, q_pids, g_pids, q_cams, g_cams)
    cmc_scores = cmc(
        distmat, q_pids, g_pids, q_cams, g_cams,
        separate_camera_set=False, single_gallery_shot=False, first_match_break=True,
    )
    return {
        "mAP": float(mAP),
        "Rank1": float(cmc_scores[0] if len(cmc_scores) > 0 else 0.0),
        "n_queries": int(len(q_items)),
        "n_gallery": int(len(g_items)),
    }


def compute_cross_source_severity_retrieval(
    features,
    meta,
    *,
    identity_dim: int,
    scoring: str,
    dist_func=None,
    sources=(1, 2),
):
    """Point 1a: cross-source, cross-severity retrieval (5x5 per direction).

    Returns a dict::
        {
          "mAP":  {(s_q, s_g): 5x5 list of lists (row sigma_q, col sigma_g)},
          "Rank1":{(s_q, s_g): 5x5 list of lists},
          "mAP_mean":  5x5 averaged over both source directions,
          "Rank1_mean":5x5 averaged over both source directions,
          "asymmetry_A": mAP_mean[sigma_q, sigma_g] - mAP_mean[sigma_g, sigma_q],
          "scoring": scoring,
        }

    The asymmetry matrix A is the H1b signature of the Randers midpoint-drift term:
    A ≡ 0 for symmetric metrics (d_E) up to sampling noise, non-trivial for d_F.
    """
    pair_directions = [(a, b) for a in sources for b in sources if a != b]
    map_by_dir: dict = {}
    rank_by_dir: dict = {}
    nq_by_dir: dict = {}

    for (s_q, s_g) in pair_directions:
        mAP_mat = np.full((len(_SEVERITIES), len(_SEVERITIES)), np.nan, dtype=np.float64)
        R1_mat = np.full_like(mAP_mat, np.nan)
        nq_mat = np.zeros_like(mAP_mat, dtype=np.int64)
        for i, sigma_q in enumerate(_SEVERITIES):
            for j, sigma_g in enumerate(_SEVERITIES):
                cell = _cross_source_cell(
                    features, meta,
                    identity_dim=identity_dim, scoring=scoring, dist_func=dist_func,
                    s_q=s_q, sigma_q=sigma_q, s_g=s_g, sigma_g=sigma_g,
                )
                mAP_mat[i, j] = cell["mAP"]
                R1_mat[i, j] = cell["Rank1"]
                nq_mat[i, j] = cell["n_queries"]
        map_by_dir[(s_q, s_g)] = mAP_mat
        rank_by_dir[(s_q, s_g)] = R1_mat
        nq_by_dir[(s_q, s_g)] = nq_mat

    mAP_stack = np.stack(list(map_by_dir.values()), axis=0)
    R1_stack = np.stack(list(rank_by_dir.values()), axis=0)
    mAP_mean = np.nanmean(mAP_stack, axis=0)
    R1_mean = np.nanmean(R1_stack, axis=0)
    A = mAP_mean - mAP_mean.T

    return {
        "severities": list(_SEVERITIES),
        "sources": list(sources),
        "scoring": scoring,
        "mAP_per_direction": {f"{a}->{b}": map_by_dir[(a, b)].tolist() for (a, b) in pair_directions},
        "Rank1_per_direction": {f"{a}->{b}": rank_by_dir[(a, b)].tolist() for (a, b) in pair_directions},
        "n_queries_per_direction": {f"{a}->{b}": nq_by_dir[(a, b)].tolist() for (a, b) in pair_directions},
        "mAP_mean": mAP_mean.tolist(),
        "Rank1_mean": R1_mean.tolist(),
        "asymmetry_A": A.tolist(),
        "asymmetry_max_abs": float(np.nanmax(np.abs(A))),
    }


def compute_same_source_severity_retrieval(
    features,
    meta,
    *,
    identity_dim: int,
    scoring: str,
    dist_func=None,
    sources=(1, 2),
):
    """Point 1b (ablation): same-source, cross-severity retrieval.

    Queries (s=s, sigma_q) and gallery (s=s, sigma_g) share source index; the mean_ap
    same-(pid, cam) filter then forces the ranker to find the PID via the OPPOSITE
    source — which isn't in the gallery, so every correct match is removed. To avoid
    that degenerate case we build the gallery as the union over BOTH sources at sigma_g
    but tag the query's source as a distinct cam so the same-source copy still gets
    filtered, matching the mathematical contract in §3.1 of the design doc.
    """
    mAP_mat = np.full((len(_SEVERITIES), len(_SEVERITIES)), np.nan, dtype=np.float64)
    R1_mat = np.full_like(mAP_mat, np.nan)
    for i, sigma_q in enumerate(_SEVERITIES):
        for j, sigma_g in enumerate(_SEVERITIES):
            if sigma_q == sigma_g:
                # Diagonal is degenerate: query equals one gallery entry (same source)
                # and the same-cam filter removes it; skipping keeps the figure honest.
                continue
            q_items, g_items = [], []
            for fname, feat in features.items():
                m = meta[fname]
                src = int(m.get("source_idx", 1))
                sev = int(m["severity"])
                if sev == sigma_q:
                    q_items.append((fname, m["pid"], src))
                if sev == sigma_g:
                    g_items.append((fname, m["pid"], src))
            if not q_items or not g_items:
                continue
            q_feats = torch.stack([features[f] for f, _, _ in q_items])
            g_feats = torch.stack([features[f] for f, _, _ in g_items])
            distmat = _score_distances(
                q_feats, g_feats, identity_dim=identity_dim, scoring=scoring, dist_func=dist_func
            )
            q_pids = np.array([pid for _, pid, _ in q_items])
            g_pids = np.array([pid for _, pid, _ in g_items])
            q_cams = np.array([c for _, _, c in q_items])
            g_cams = np.array([c for _, _, c in g_items])
            mAP_mat[i, j] = mean_ap(distmat, q_pids, g_pids, q_cams, g_cams)
            cmc_scores = cmc(
                distmat, q_pids, g_pids, q_cams, g_cams,
                separate_camera_set=False, single_gallery_shot=False, first_match_break=True,
            )
            R1_mat[i, j] = float(cmc_scores[0] if len(cmc_scores) > 0 else 0.0)

    A = np.where(np.isnan(mAP_mat) | np.isnan(mAP_mat.T), np.nan, mAP_mat - mAP_mat.T)

    return {
        "severities": list(_SEVERITIES),
        "scoring": scoring,
        "mAP": mAP_mat.tolist(),
        "Rank1": R1_mat.tolist(),
        "asymmetry_A": A.tolist(),
        "asymmetry_max_abs": float(np.nanmax(np.abs(A))) if np.any(~np.isnan(A)) else float("nan"),
        "note": "Diagonal sigma_q=sigma_g skipped (same-source same-severity collapses).",
    }


def _parallel_transport_omega(
    omega_j: np.ndarray, n_j: np.ndarray, n_i: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """Parallel transport of omega_j from tangent space at n_j to tangent space at n_i.

    Closed-form on the unit sphere (Pennec 2006, §3). Both n_i, n_j must be unit-norm.
    Returns the transported vector; falls back to omega_j if n_i, n_j are antipodal.
    """
    dot = float(np.dot(n_i, n_j))
    if dot <= -1.0 + eps:
        return omega_j.copy()
    coeff = float(np.dot(n_j, omega_j)) / (1.0 + dot)
    return omega_j - coeff * (n_i + n_j)


def compute_drift_cosine_matrix(
    features,
    meta,
    *,
    identity_dim: int,
    transport: str = "ambient",
    max_entries: int = 250000,
):
    """Point 2: all-pairs cosine-similarity matrix on drift vectors.

    ``transport='ambient'`` — cosine in R^{d_id} (free-vector similarity, primary).
    ``transport='parallel'`` — parallel-transport omega_j to the tangent at z^id_i on
    the unit sphere via the closed-form Pennec (2006) formula, then cosine. Because
    parallel transport is an isometry, ||P omega_j|| = ||omega_j||.

    Reports the full N x N matrix (JSON-truncated if >max_entries) plus 8-way block
    means over (same/diff pid) x (same/diff source) x (same/diff severity).
    """
    keys = sorted(features.keys())
    N = len(keys)
    omega = np.stack(
        [features[k][identity_dim:].cpu().numpy().astype(np.float64) for k in keys], axis=0
    )
    z_id_raw = np.stack(
        [features[k][:identity_dim].cpu().numpy().astype(np.float64) for k in keys], axis=0
    )
    z_id_unit = z_id_raw / np.maximum(np.linalg.norm(z_id_raw, axis=1, keepdims=True), 1e-12)
    om_norm = np.linalg.norm(omega, axis=1)
    om_safe = np.maximum(om_norm, 1e-12)

    if transport == "ambient":
        omega_unit = omega / om_safe[:, None]
        C = omega_unit @ omega_unit.T
    elif transport == "parallel":
        C = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            om_i_norm = om_safe[i]
            for j in range(N):
                if i == j:
                    C[i, j] = 1.0
                    continue
                transported = _parallel_transport_omega(omega[j], z_id_unit[j], z_id_unit[i])
                C[i, j] = float(omega[i] @ transported) / (om_i_norm * om_safe[j])
    else:
        raise ValueError(f"Unknown transport: {transport}")

    pids = np.array([meta[k]["pid"] for k in keys])
    srcs = np.array([int(meta[k].get("source_idx", 1)) for k in keys])
    sevs = np.array([int(meta[k]["severity"]) for k in keys])

    same_pid = pids[:, None] == pids[None, :]
    same_src = srcs[:, None] == srcs[None, :]
    same_sev = sevs[:, None] == sevs[None, :]
    triu = np.triu(np.ones_like(same_pid, dtype=bool), k=1)

    blocks = {}
    for P_label, P_mask in [("same_pid", same_pid), ("diff_pid", ~same_pid)]:
        for S_label, S_mask in [("same_src", same_src), ("diff_src", ~same_src)]:
            for Sig_label, Sig_mask in [("same_sev", same_sev), ("diff_sev", ~same_sev)]:
                mask = P_mask & S_mask & Sig_mask & triu
                n = int(mask.sum())
                if n == 0:
                    blocks[f"{P_label}__{S_label}__{Sig_label}"] = {
                        "n_pairs": 0, "mean": float("nan"), "std": float("nan"),
                    }
                else:
                    vals = C[mask]
                    blocks[f"{P_label}__{S_label}__{Sig_label}"] = {
                        "n_pairs": n,
                        "mean": float(vals.mean()),
                        "std": float(vals.std()),
                    }

    out: dict = {
        "transport": transport,
        "n_samples": N,
        "keys": keys,
        "pids": pids.tolist(),
        "source_idx": srcs.tolist(),
        "severity": sevs.tolist(),
        "block_means": blocks,
    }
    if N * N <= max_entries:
        out["C_full"] = C.tolist()
    else:
        out["C_full"] = None
        out["C_full_dropped_reason"] = f"N*N={N*N} exceeds max_entries={max_entries}"
    return out, C, keys, pids, srcs, sevs


def _orthogonal_projection_stats(delta: np.ndarray, w: np.ndarray) -> dict:
    d_norm = float(np.linalg.norm(delta))
    if d_norm < 1e-12:
        return {"delta_norm": 0.0, "along": 0.0, "perp_norm": 0.0, "eta": float("nan")}
    along = float(np.dot(w, delta))
    eta = (along * along) / (d_norm * d_norm)
    perp_norm = float(np.sqrt(max(d_norm * d_norm - along * along, 0.0)))
    return {"delta_norm": d_norm, "along": along, "perp_norm": perp_norm, "eta": float(eta)}


def _projector_C_subspace(omega_0: np.ndarray, omega_k: np.ndarray) -> np.ndarray | None:
    """Return U in R^{d x 2} with orthonormal columns spanning {omega_0, omega_k}, or None."""
    n0 = np.linalg.norm(omega_0)
    nk = np.linalg.norm(omega_k)
    if n0 < 1e-12 or nk < 1e-12:
        return None
    u1 = omega_0 / n0
    tilde_k = omega_k / nk
    proj = float(u1 @ tilde_k) * u1
    rem = tilde_k - proj
    rem_norm = np.linalg.norm(rem)
    if rem_norm < 1e-10:
        return u1[:, None]  # degenerate: 1-D subspace
    u2 = rem / rem_norm
    return np.stack([u1, u2], axis=1)


def compute_drift_orthogonal_projection(
    features,
    meta,
    *,
    identity_dim: int,
    source_idx: int = 1,
    shuffle_seed: int = 0,
    shuffle_reps: int = 200,
):
    """Point 3: drift-orthogonal projection of identity drift Delta.

    For every PID p with severities 0 and k=1..4 at the given source_idx, compute
      - Delta     = z^id_k - z^id_0
      - w_B       = (omega_0 + omega_k) / ||omega_0 + omega_k||   (midpoint, primary; paper eq. 4)
      - w_A       = omega_k / ||omega_k||                          (per-sample, legacy; §5.1)
      - w_ref     = omega_0 / ||omega_0||                          (reference-point; supplement §2)
      - U         = GS orthonormalisation of (omega_0, omega_k)    (2-D, supporting)
      - eta_B     = (w_B^T Delta)^2 / ||Delta||^2
      - eta_A     = (w_A^T Delta)^2 / ||Delta||^2
      - eta_ref   = (w_ref^T Delta)^2 / ||Delta||^2
      - eta_C     = ||U^T Delta||^2 / ||Delta||^2 = cos^2 theta_1
      - cos_omega_0_k  = (omega_0 . omega_k) / (||omega_0|| ||omega_k||)
      - finsler_distance = ||Delta|| + 0.5 ||omega_0 + omega_k|| * (w_B^T Delta)

    See changelogs/toy_dataset_reference_projector_supplement.md for the w_ref
    projector, its conditional ordering relative to w_B, and the validity gauge
    role of cos_omega_0_k.

    Shuffle null: recompute eta_B, eta_C, eta_ref after permuting the
    (omega_0, omega_k) pairing against Delta (preserves empirical marginals;
    tests conditional coupling between drift subspace and identity drift).
    """
    groups = _group_by_pid_source_severity(features, meta)
    per_pair = []  # list of dicts

    # Collect per (pid, k)
    for pid, by_src in groups.items():
        by_sev = by_src.get(source_idx)
        if by_sev is None or 0 not in by_sev:
            continue
        fname0, feat0 = by_sev[0]
        z0 = feat0[:identity_dim].cpu().numpy().astype(np.float64)
        om0 = feat0[identity_dim:].cpu().numpy().astype(np.float64)
        for k in (1, 2, 3, 4):
            if k not in by_sev:
                continue
            fnamek, featk = by_sev[k]
            zk = featk[:identity_dim].cpu().numpy().astype(np.float64)
            omk = featk[identity_dim:].cpu().numpy().astype(np.float64)
            delta = zk - z0

            # B: midpoint projector (primary)
            mid = om0 + omk
            mid_n = np.linalg.norm(mid)
            if mid_n < 1e-12:
                continue
            w_B = mid / mid_n
            stats_B = _orthogonal_projection_stats(delta, w_B)

            # A: per-sample projector (legacy reference; uses omega_k)
            omk_n = np.linalg.norm(omk)
            stats_A = {"eta": float("nan"), "delta_norm": stats_B["delta_norm"], "perp_norm": float("nan")}
            if omk_n >= 1e-12:
                w_A = omk / omk_n
                stats_A = _orthogonal_projection_stats(delta, w_A)

            # ref: reference-point projector (uses omega_0; supplement §2)
            om0_n = float(np.linalg.norm(om0))
            stats_ref = {"eta": float("nan"), "delta_norm": stats_B["delta_norm"],
                         "perp_norm": float("nan"), "along": float("nan")}
            if om0_n >= 1e-12:
                w_ref = om0 / om0_n
                stats_ref = _orthogonal_projection_stats(delta, w_ref)

            # cosine similarity between the clean and perturbed drift vectors
            if om0_n < 1e-12 or omk_n < 1e-12:
                cos_0k = float("nan")
            else:
                cos_0k = float(np.dot(om0, omk) / (om0_n * omk_n))

            # C: 2-D GS projector (supporting)
            U = _projector_C_subspace(om0, omk)
            if U is None:
                eta_C = float("nan"); perp_C = float("nan")
            else:
                UtD = U.T @ delta
                cap = float(UtD @ UtD)
                d2 = float(delta @ delta)
                eta_C = cap / d2 if d2 > 0 else float("nan")
                perp_C = float(np.sqrt(max(d2 - cap, 0.0)))

            # Randers coupling scalar: 1/2 * ||om0+omk|| * (w_B^T Delta)
            randers_term = 0.5 * mid_n * stats_B["along"]
            # Mean Finsler distance between the full embeddings (identity term + Randers term)
            finsler_d = float(stats_B["delta_norm"] + randers_term)

            per_pair.append({
                "pid": int(pid),
                "source_idx": int(source_idx),
                "severity_k": int(k),
                "delta_norm": stats_B["delta_norm"],
                "eta_A": stats_A["eta"],
                "eta_B": stats_B["eta"],
                "eta_C": float(eta_C),
                "eta_ref": stats_ref["eta"],
                "perp_B_norm": stats_B["perp_norm"],
                "perp_C_norm": float(perp_C),
                "perp_ref_norm": stats_ref["perp_norm"],
                "cos_omega_0_k": cos_0k,
                "randers_asymmetry_term": float(randers_term),
                "finsler_distance": finsler_d,
                "omega0_norm": om0_n,
                "omegak_norm": float(omk_n),
                "omega_midpoint_norm": float(mid_n),
            })

    # Aggregate by severity k
    per_k: dict = {}
    for k in (1, 2, 3, 4):
        rows = [r for r in per_pair if r["severity_k"] == k]
        if not rows:
            per_k[k] = {"n": 0}
            continue
        per_k[k] = {
            "n": len(rows),
            "eta_A_mean": float(np.nanmean([r["eta_A"] for r in rows])),
            "eta_B_mean": float(np.nanmean([r["eta_B"] for r in rows])),
            "eta_C_mean": float(np.nanmean([r["eta_C"] for r in rows])),
            "eta_ref_mean": float(np.nanmean([r["eta_ref"] for r in rows])),
            "eta_B_std": float(np.nanstd([r["eta_B"] for r in rows])),
            "eta_C_std": float(np.nanstd([r["eta_C"] for r in rows])),
            "eta_ref_std": float(np.nanstd([r["eta_ref"] for r in rows])),
            "delta_norm_mean": float(np.mean([r["delta_norm"] for r in rows])),
            "randers_term_mean": float(np.mean([r["randers_asymmetry_term"] for r in rows])),
            "cos_omega_0_k_mean": float(np.nanmean([r["cos_omega_0_k"] for r in rows])),
            "cos_omega_0_k_std": float(np.nanstd([r["cos_omega_0_k"] for r in rows])),
            "perp_ref_norm_mean": float(np.nanmean([r["perp_ref_norm"] for r in rows])),
            "perp_ref_norm_std": float(np.nanstd([r["perp_ref_norm"] for r in rows])),
            "finsler_distance_mean": float(np.mean([r["finsler_distance"] for r in rows])),
            "finsler_distance_std": float(np.std([r["finsler_distance"] for r in rows])),
        }

    # Shuffle null: permute the (om0, omk) pairing across PIDs within each k.
    rng = np.random.default_rng(int(shuffle_seed))
    shuffle_nulls = {k: {"eta_B": [], "eta_C": [], "eta_ref": []} for k in (1, 2, 3, 4)}
    # Pre-collect per-k arrays for a vectorised shuffle
    per_k_arrays: dict = {}
    for k in (1, 2, 3, 4):
        rows = [r for r in per_pair if r["severity_k"] == k]
        if not rows:
            continue
        # Reconstruct arrays
        deltas, mids, Us, om0s = [], [], [], []
        for pid, by_src in groups.items():
            by_sev = by_src.get(source_idx)
            if by_sev is None or 0 not in by_sev or k not in by_sev:
                continue
            _, f0 = by_sev[0]
            _, fk = by_sev[k]
            z0 = f0[:identity_dim].cpu().numpy().astype(np.float64)
            zk = fk[:identity_dim].cpu().numpy().astype(np.float64)
            om0 = f0[identity_dim:].cpu().numpy().astype(np.float64)
            omk = fk[identity_dim:].cpu().numpy().astype(np.float64)
            deltas.append(zk - z0)
            mids.append(om0 + omk)
            Us.append(_projector_C_subspace(om0, omk))
            om0s.append(om0)
        per_k_arrays[k] = (deltas, mids, Us, om0s)

    for k, (deltas, mids, Us, om0s) in per_k_arrays.items():
        n = len(deltas)
        if n < 2:
            continue
        for _ in range(int(shuffle_reps)):
            perm = rng.permutation(n)
            etaB_vals, etaC_vals, etaref_vals = [], [], []
            for i in range(n):
                d = deltas[i]
                mid = mids[perm[i]]
                U = Us[perm[i]]
                om0_sh = om0s[perm[i]]
                d2 = float(d @ d)
                if d2 < 1e-12:
                    continue
                mid_n = np.linalg.norm(mid)
                if mid_n < 1e-12:
                    continue
                w_B = mid / mid_n
                along = float(w_B @ d)
                etaB_vals.append((along * along) / d2)
                if U is not None:
                    UtD = U.T @ d
                    etaC_vals.append(float(UtD @ UtD) / d2)
                om0_sh_n = np.linalg.norm(om0_sh)
                if om0_sh_n >= 1e-12:
                    w_ref_sh = om0_sh / om0_sh_n
                    along_ref = float(w_ref_sh @ d)
                    etaref_vals.append((along_ref * along_ref) / d2)
            if etaB_vals:
                shuffle_nulls[k]["eta_B"].append(float(np.mean(etaB_vals)))
            if etaC_vals:
                shuffle_nulls[k]["eta_C"].append(float(np.mean(etaC_vals)))
            if etaref_vals:
                shuffle_nulls[k]["eta_ref"].append(float(np.mean(etaref_vals)))

    shuffle_summary = {}
    for k, d in shuffle_nulls.items():
        shuffle_summary[k] = {
            "n_reps": len(d["eta_B"]),
            "eta_B_null_mean": float(np.mean(d["eta_B"])) if d["eta_B"] else float("nan"),
            "eta_B_null_std":  float(np.std(d["eta_B"]))  if d["eta_B"] else float("nan"),
            "eta_C_null_mean": float(np.mean(d["eta_C"])) if d["eta_C"] else float("nan"),
            "eta_C_null_std":  float(np.std(d["eta_C"]))  if d["eta_C"] else float("nan"),
            "eta_ref_null_mean": float(np.mean(d["eta_ref"])) if d["eta_ref"] else float("nan"),
            "eta_ref_null_std":  float(np.std(d["eta_ref"]))  if d["eta_ref"] else float("nan"),
        }

    return {
        "source_idx": int(source_idx),
        "identity_dim": int(identity_dim),
        "analytic_null_B": 1.0 / float(identity_dim),
        "analytic_null_C": 2.0 / float(identity_dim),
        "per_pair": per_pair,
        "per_severity_k": per_k,
        "shuffle_null": shuffle_summary,
    }


def compute_pairwise_drift_shift_spearman(features, meta, identity_dim: int):
    """H1c: Spearman(|sigma_i - sigma_j|, ||omega_i - omega_j||) over within-PID pairs."""
    groups = _group_by_pid_source_severity(features, meta)
    sev_diffs, om_diffs = [], []
    for pid, by_src in groups.items():
        # pool all (source, severity) samples for this PID
        items = []
        for src, by_sev in by_src.items():
            for sev, (fname, feat) in by_sev.items():
                items.append((src, sev, feat[identity_dim:].cpu().numpy().astype(np.float64)))
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                s_i, sev_i, om_i = items[i]
                s_j, sev_j, om_j = items[j]
                sev_diffs.append(abs(sev_i - sev_j))
                om_diffs.append(float(np.linalg.norm(om_i - om_j)))
    if not sev_diffs:
        return {"rho": float("nan"), "pval": float("nan"), "n_pairs": 0}
    rho, pval = stats.spearmanr(sev_diffs, om_diffs)
    return {"rho": float(rho), "pval": float(pval), "n_pairs": int(len(sev_diffs))}


def plot_omega_identity_pca_panels(
    z_id,
    omega,
    severities,
    pids,
    output_path,
    selected_pids,
    tsne_path=None,
    tsne_perplexity=30.0,
    tsne_seed=0,
):
    """Two-panel figure: PCA on ω (polylines s=0→4 per identity) and PCA on z_id (C1 sanity).

    Toy dataset: synthetic ``camera'' IDs equal corruption severity + 1; not physical viewpoints.
    PCA is a linear map: projected differences match projection of Δω in the same basis.
    """
    # --- Drift (ω) PCA ---
    pca_o = PCA(n_components=2)
    Wo = pca_o.fit_transform(omega)

    fig, (ax_o, ax_i) = plt.subplots(1, 2, figsize=(10, 4.2))

    try:
        cmap = colormaps["viridis"].resampled(5)
    except (AttributeError, TypeError, ValueError):
        cmap = plt.cm.get_cmap("viridis", 5)
    norm = plt.Normalize(vmin=0, vmax=4)

    for pid in selected_pids:
        mask = pids == pid
        idx = np.where(mask)[0]
        order = np.argsort(severities[mask])
        idx = idx[order]
        ax_o.plot(Wo[idx, 0], Wo[idx, 1], "-", color="0.75", linewidth=1.0, zorder=1)
        sc = ax_o.scatter(
            Wo[idx, 0],
            Wo[idx, 1],
            c=severities[idx],
            cmap=cmap,
            norm=norm,
            s=36,
            zorder=2,
            edgecolors="white",
            linewidths=0.4,
        )

    ax_o.set_title(r"Drift $\hat{\mathbf{z}}^{\omega}$ (PCA-2)", fontsize=11)
    ax_o.set_xlabel(r"PC1 ({:.0f}\% var.)".format(100 * pca_o.explained_variance_ratio_[0]))
    ax_o.set_ylabel(r"PC2 ({:.0f}\% var.)".format(100 * pca_o.explained_variance_ratio_[1]))
    cbar = fig.colorbar(sc, ax=ax_o, fraction=0.046, pad=0.04, ticks=range(5))
    cbar.set_label(r"Severity $s$ (toy $\mathrm{cam}=s{+}1$)")

    # --- Identity PCA (separate basis) ---
    pca_z = PCA(n_components=2)
    Wz = pca_z.fit_transform(z_id)

    for pid in selected_pids:
        mask = pids == pid
        idx = np.where(mask)[0]
        order = np.argsort(severities[mask])
        idx = idx[order]
        ax_i.plot(Wz[idx, 0], Wz[idx, 1], "-", color="0.75", linewidth=1.0, zorder=1)
        sc2 = ax_i.scatter(
            Wz[idx, 0],
            Wz[idx, 1],
            c=severities[idx],
            cmap=cmap,
            norm=norm,
            s=36,
            zorder=2,
            edgecolors="white",
            linewidths=0.4,
        )

    ax_i.set_title(r"Identity $\mathbf{z}^{\mathrm{id}}$ (PCA-2)", fontsize=11)
    ax_i.set_xlabel(r"PC1 ({:.0f}\% var.)".format(100 * pca_z.explained_variance_ratio_[0]))
    ax_i.set_ylabel(r"PC2 ({:.0f}\% var.)".format(100 * pca_z.explained_variance_ratio_[1]))
    fig.colorbar(sc2, ax=ax_i, fraction=0.046, pad=0.04, ticks=range(5))

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved ω / identity PCA figure to {output_path}")

    provenance = {
        "sklearn_version": sklearn.__version__,
        "selected_pids": [int(x) for x in selected_pids],
        "n_points": int(len(severities)),
        "omega_pca": {
            "explained_variance_ratio": pca_o.explained_variance_ratio_.tolist(),
            "mean_abs_omega": float(np.mean(np.abs(omega))),
        },
        "identity_pca": {
            "explained_variance_ratio": pca_z.explained_variance_ratio_.tolist(),
        },
    }

    if tsne_path:
        n = omega.shape[0]
        perp = min(float(tsne_perplexity), max(5.0, (n - 1) / 3.0))
        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            learning_rate="auto",
            init="pca",
            random_state=int(tsne_seed),
        )
        Wo_tsne = tsne.fit_transform(omega)
        fig_t, ax_t = plt.subplots(figsize=(5.5, 4.2))
        sc_t = ax_t.scatter(
            Wo_tsne[:, 0],
            Wo_tsne[:, 1],
            c=severities,
            cmap=cmap,
            norm=norm,
            s=28,
            edgecolors="white",
            linewidths=0.35,
        )
        ax_t.set_title(r"$t$-SNE on $\hat{\mathbf{z}}^{\omega}$")
        ax_t.set_xlabel("dim 1")
        ax_t.set_ylabel("dim 2")
        fig_t.colorbar(sc_t, ax=ax_t, fraction=0.046, pad=0.04, ticks=range(5))
        fig_t.tight_layout()
        fig_t.savefig(tsne_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig_t)
        print(f"Saved ω t-SNE figure to {tsne_path}")
        provenance["tsne"] = {
            "perplexity_used": float(perp),
            "random_state": int(tsne_seed),
            "note": "Scatter only; distances/angles not interpretable as ω geometry.",
        }

    return provenance


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_quantitative(metrics, output_path):
    """Plot drift magnitude and identity distance vs. corruption severity."""
    severities = sorted(k for k in metrics.keys() if isinstance(k, int))
    drift_means = [metrics[s]["drift_norm_mean"] for s in severities]
    drift_stds = [metrics[s]["drift_norm_std"] for s in severities]
    id_means = [metrics[s]["identity_dist_mean"] for s in severities]
    id_stds = [metrics[s]["identity_dist_std"] for s in severities]

    fig, ax1 = plt.subplots(figsize=(5.5, 4))

    color_drift = "#1f77b4"
    color_id = "#d62728"

    ax1.set_xlabel("Corruption severity level", fontsize=12)
    ax1.set_ylabel(r"Drift magnitude $\|\hat{\mathbf{z}}^{\omega}\|_2$", color=color_drift, fontsize=11)
    line1 = ax1.errorbar(severities, drift_means, yerr=drift_stds, color=color_drift,
                         marker="o", capsize=4, linewidth=2, label=r"$\|\hat{\mathbf{z}}^{\omega}\|_2$")
    ax1.tick_params(axis="y", labelcolor=color_drift)

    # Annotate values
    for s, m in zip(severities, drift_means):
        ax1.annotate(f"{m:.3f}", (s, m), textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=8, color=color_drift)

    ax2 = ax1.twinx()
    ax2.set_ylabel(r"Identity distance $\|\mathbf{z}_k^{\mathrm{id}} - \mathbf{z}_0^{\mathrm{id}}\|_2$",
                   color=color_id, fontsize=11)
    line2 = ax2.errorbar(severities, id_means, yerr=id_stds, color=color_id,
                         marker="s", capsize=4, linewidth=2, linestyle="--",
                         label=r"$\|\mathbf{z}_k^{\mathrm{id}} - \mathbf{z}_0^{\mathrm{id}}\|_2$")
    ax2.tick_params(axis="y", labelcolor=color_id)

    rho = metrics.get("spearman_rho", float("nan"))
    pval = metrics.get("spearman_pval", float("nan"))
    ax1.text(
        0.02,
        0.98,
        f"$\\rho_s={rho:.3f}$, $p={pval:.1e}$",
        transform=ax1.transAxes,
        fontsize=8,
        verticalalignment="top",
        color="0.35",
    )

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=9)

    ax1.set_xticks(severities)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved quantitative figure to {output_path}")


def plot_qualitative_retrieval(eucl_topk, finsler_topk, dataset_dir, output_path,
                               num_queries=4, top_k=5, seed=42):
    """Plot qualitative rank-list comparison: Euclidean vs. Finsler."""
    rng = np.random.RandomState(seed)

    # Find queries where Finsler gets rank-1 correct but Euclidean does not
    diff_queries = []
    for qf in eucl_topk:
        if qf not in finsler_topk:
            continue
        e_correct = eucl_topk[qf]["topk"][0]["correct"] if eucl_topk[qf]["topk"] else False
        f_correct = finsler_topk[qf]["topk"][0]["correct"] if finsler_topk[qf]["topk"] else False
        if f_correct and not e_correct:
            diff_queries.append(qf)

    # If not enough differential queries, fall back to random
    if len(diff_queries) < num_queries:
        all_queries = list(eucl_topk.keys())
        diff_queries = rng.choice(all_queries, size=min(num_queries, len(all_queries)), replace=False).tolist()
    else:
        diff_queries = rng.choice(diff_queries, size=min(num_queries, len(diff_queries)), replace=False).tolist()

    nrows = num_queries
    ncols = 1 + top_k  # query + top-k
    fig, axes = plt.subplots(nrows * 2, ncols, figsize=(1.8 * ncols, 2.8 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    gallery_dir = osp.join(dataset_dir, "gallery")
    query_s1_dir = osp.join(dataset_dir, "query_s1")
    query_s2_dir = osp.join(dataset_dir, "query_s2")
    all_dir = osp.join(dataset_dir, "bounding_box_test")

    def load_img(fname):
        for d in [query_s1_dir, query_s2_dir, gallery_dir, all_dir]:
            p = osp.join(d, fname)
            if osp.exists(p):
                return Image.open(p).convert("RGB")
        return Image.new("RGB", (64, 128), (200, 200, 200))

    for row_idx, qf in enumerate(diff_queries[:num_queries]):
        # Row pair: Euclidean (top), Finsler (bottom)
        for method_idx, (method_name, topk_data) in enumerate([
            ("Euclidean BAU", eucl_topk), ("Finsler BAU", finsler_topk)
        ]):
            ax_row = row_idx * 2 + method_idx
            qpid = topk_data[qf]["query_pid"]

            # Query image
            ax = axes[ax_row, 0]
            qimg = load_img(qf)
            ax.imshow(qimg)
            ax.set_title(f"Query\nPID={qpid}", fontsize=7)
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("blue")
                spine.set_linewidth(2)

            # Top-k gallery
            for k_idx, item in enumerate(topk_data[qf]["topk"][:top_k]):
                ax = axes[ax_row, 1 + k_idx]
                gimg = load_img(item["fname"])
                ax.imshow(gimg)
                color = "green" if item["correct"] else "red"
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color(color)
                    spine.set_linewidth(3)
                sev = int(item.get("severity", int(item["camid"]) - 1))
                ax.set_title(f"s={sev}", fontsize=7)
                ax.axis("off")

            # Label method on the left
            axes[ax_row, 0].set_ylabel(method_name, fontsize=9, rotation=0, labelpad=40, va="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved qualitative retrieval figure to {output_path}")


def plot_paper_corruption_strip(features, meta, dataset_dir, output_path, identity_dim=2048):
    """Five-panel strip with $\\|\\omega\\|$ per crop (optional; e.g. draft\\_3).  No suptitle."""
    pid_features = defaultdict(dict)
    for fname, feat in features.items():
        m = meta[fname]
        pid_features[m["pid"]][m["severity"]] = (fname, feat)

    example_pid = None
    for pid in sorted(pid_features.keys()):
        if len(pid_features[pid]) == 5:
            example_pid = pid
            break

    if example_pid is None:
        print(f"Warning: no PID with all 5 severities; skip {output_path}")
        return

    all_dir = osp.join(dataset_dir, "bounding_box_test")
    n_levels = 5
    fig, axes = plt.subplots(1, n_levels, figsize=(10, 2.6))
    if n_levels == 1:
        axes = [axes]

    for sev_idx in range(n_levels):
        fname, feat = pid_features[example_pid][sev_idx]
        drift = feat[identity_dim:]
        drift_norm = torch.norm(drift, p=2).item()
        img_path = osp.join(all_dir, fname)
        img = Image.open(img_path).convert("RGB")
        ax = axes[sev_idx]
        ax.imshow(img)
        ax.set_title(f"$s={sev_idx}$\n$\\|\\omega\\|={drift_norm:.3f}$", fontsize=7)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved optional ω-strip to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_DEFAULT_EUCLIDEAN_CKPT = (
    "/home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification/"
    "logs/AGReIDv2_sweep/job_1502418_dt_cuhk03_EucBaseline/best.pth"
)


def main():
    parser = argparse.ArgumentParser(description="Toy dataset drift vector analysis")
    parser.add_argument("--finsler-checkpoint", type=str, required=True,
                        help="Path to best Finsler (resnet50_finsler) checkpoint")
    parser.add_argument(
        "--euclidean-checkpoint",
        type=str,
        default=_DEFAULT_EUCLIDEAN_CKPT,
        help="Path to Euclidean BAU (resnet50) baseline checkpoint",
    )
    parser.add_argument("--dataset-dir", type=str,
                        default="/home/stud/leez/storage/user/reid/data/ToyCorruption",
                        help="Path to ToyCorruption dataset")
    parser.add_argument("--output-dir", type=str,
                        default="results/toy_analysis",
                        help="Output directory for figures and metrics")
    parser.add_argument("--num-classes", type=int, default=13726)
    parser.add_argument("--num-domains", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument(
        "--viz-num-identities",
        type=int,
        default=15,
        help="Subsample this many identities (must have severities 0--4) for ω / identity PCA plots",
    )
    parser.add_argument(
        "--viz-seed",
        type=int,
        default=42,
        help="RNG seed for deterministic PID subsample",
    )
    parser.add_argument(
        "--fig-omega-identity-pca",
        type=str,
        default="fig_omega_identity_pca.pdf",
        help="Output filename for combined ω + identity PCA figure (under --output-dir)",
    )
    parser.add_argument(
        "--viz-tsne",
        action="store_true",
        help="Also write exploratory t-SNE scatter on ω only (non-metric; no vector semantics)",
    )
    parser.add_argument(
        "--fig-omega-tsne",
        type=str,
        default="fig_omega_tsne.pdf",
        help="Output filename for optional t-SNE figure (under --output-dir)",
    )
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-seed", type=int, default=0)
    parser.add_argument(
        "--paper-fig-dir",
        type=str,
        default="paper/draft_4/fig",
        help="Copy fig_corruption_strip.pdf and fig_retrieval_mAP.pdf here (LaTeX hooks)",
    )
    parser.add_argument(
        "--paper-mirror-dir",
        type=str,
        default="paper/draft_4/toy_analysis",
        help="Mirror all files from --output-dir here; set to empty string to disable",
    )
    parser.add_argument(
        "--omega-cos-transport",
        type=str,
        default="ambient",
        choices=("ambient", "parallel"),
        help="Drift cosine matrix variant (Point 2). ambient = free-vector cosine (primary); "
             "parallel = Pennec (2006) closed-form parallel transport on the identity sphere.",
    )
    parser.add_argument(
        "--projection-report",
        type=str,
        default="B_primary",
        choices=("B_primary", "all"),
        help="Point 3 projector reporting. B_primary = paper-facing midpoint figure only; "
             "all = also include legacy per-sample P_A and 2-D P_C in the PDF.",
    )
    parser.add_argument(
        "--projection-shuffle-reps",
        type=int,
        default=200,
        help="Number of permutations for the Point 3 shuffle null (default: 200).",
    )
    parser.add_argument(
        "--skip-v3-diagnostics",
        action="store_true",
        help="Skip Points 1a/1b/2/3 even on v3.0 data (legacy v2.0 output only).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset_version = _detect_dataset_version(args.dataset_dir)
    is_v3 = dataset_version.startswith("3") and not args.skip_v3_diagnostics
    print(f"Detected ToyCorruption dataset version: {dataset_version} "
          f"(v3 diagnostics {'enabled' if is_v3 else 'disabled'})")

    # -----------------------------------------------------------------------
    # 1. Load models
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Loading Finsler model (full embedding: identity + drift)...")
    finsler_model = load_finsler_model(
        args.finsler_checkpoint,
        num_classes=args.num_classes,
        num_domains=args.num_domains,
    )

    print("Loading Euclidean BAU baseline (resnet50)...")
    print(f"  checkpoint: {args.euclidean_checkpoint}")
    eucl_model = load_bau_euclidean_resnet50(args.euclidean_checkpoint, num_classes=args.num_classes)

    # -----------------------------------------------------------------------
    # 2. Extract features on the full toy dataset
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Extracting features on toy dataset...")
    transform = get_test_transform(args.height, args.width)
    all_dataset = ToyDataset(
        osp.join(args.dataset_dir, "bounding_box_test"),
        transform=transform,
        dataset_version=dataset_version,
    )
    all_loader = DataLoader(all_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    print(f"  Dataset size: {len(all_dataset)} images")
    finsler_features, meta = extract_features(finsler_model, all_loader)
    eucl_features, _ = extract_features(eucl_model, all_loader)
    print(f"  Extracted {len(finsler_features)} Finsler embeddings (dim={next(iter(finsler_features.values())).shape[0]})")
    print(f"  Extracted {len(eucl_features)} Euclidean BAU embeddings (dim={next(iter(eucl_features.values())).shape[0]})")

    identity_dim = 2048

    # -----------------------------------------------------------------------
    # 2b. Low-dimensional ω / identity PCA (subsampled)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Drift / identity PCA visualization (subsampled identities)...")
    selected_pids = subsample_pids_with_full_severity(meta, args.viz_num_identities, args.viz_seed)
    if len(selected_pids) == 0:
        print("  Skipped: no PIDs with all severity levels 0--4 in metadata.")
    else:
        print(f"  Selected {len(selected_pids)} identities (seed={args.viz_seed}): {selected_pids}")
        z_id_np, omega_np, sev_np, pid_np = gather_identity_drift_arrays(
            finsler_features, meta, selected_pids, identity_dim=identity_dim
        )
        pca_path = osp.join(args.output_dir, args.fig_omega_identity_pca)
        tsne_path = osp.join(args.output_dir, args.fig_omega_tsne) if args.viz_tsne else None
        prov = plot_omega_identity_pca_panels(
            z_id_np,
            omega_np,
            sev_np,
            pid_np,
            pca_path,
            selected_pids,
            tsne_path=tsne_path,
            tsne_perplexity=args.tsne_perplexity,
            tsne_seed=args.tsne_seed,
        )
        prov["viz_num_identities_requested"] = args.viz_num_identities
        prov["viz_seed"] = args.viz_seed
        prov["fig_omega_identity_pca"] = args.fig_omega_identity_pca
        if args.viz_tsne:
            prov["fig_omega_tsne"] = args.fig_omega_tsne
        prov_path = osp.join(args.output_dir, "drift_projection_provenance.json")
        with open(prov_path, "w") as f:
            json.dump(
                {
                    **prov,
                    "finsler_checkpoint": args.finsler_checkpoint,
                    "note": "Toy cam_id = severity + 1 (corruption axis); not physical cameras.",
                },
                f,
                indent=2,
            )
        print(f"  Wrote provenance to {prov_path}")

    # -----------------------------------------------------------------------
    # 3. C1 + C2: Drift magnitude and identity distance vs. severity
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Computing drift/identity metrics per severity level...")
    drift_metrics = compute_drift_identity_metrics(finsler_features, meta, identity_dim=identity_dim)

    print("\nResults:")
    print(f"  Spearman rho = {drift_metrics['spearman_rho']:.4f}, p = {drift_metrics['spearman_pval']:.2e}")
    for sev in sorted(k for k in drift_metrics if isinstance(k, int)):
        m = drift_metrics[sev]
        print(f"  Severity {sev}: drift_norm={m['drift_norm_mean']:.4f}±{m['drift_norm_std']:.4f}, "
              f"id_dist={m['identity_dist_mean']:.4f}±{m['identity_dist_std']:.4f}")

    # Save metrics
    metrics_path = osp.join(args.output_dir, "drift_identity_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(drift_metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    # Plot quantitative figure
    quant_path = osp.join(args.output_dir, "fig_drift_monotonicity.pdf")
    plot_quantitative(drift_metrics, quant_path)

    # -----------------------------------------------------------------------
    # 4. C3: Retrieval experiment (clean query -> corrupted gallery)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Computing retrieval metrics (clean query -> corrupted gallery)...")
    print("  Euclidean: BAU resnet50 checkpoint, ranking = Euclidean on 2048-d embeddings")
    print("  Finsler:   resnet50_finsler checkpoint, ranking = finsler_drift_dist on 4096-d")

    dist_func = partial(finsler_drift_dist, identity_dim=identity_dim, method="symmetric_trapezoidal")

    eucl_retrieval = compute_retrieval_metrics(
        eucl_features,
        meta,
        identity_dim=identity_dim,
        baseline_euclidean_resnet50=True,
    )
    finsler_retrieval = compute_retrieval_metrics(
        finsler_features,
        meta,
        identity_dim=identity_dim,
        dist_func=dist_func,
        use_full_embedding=True,
    )

    print("\nRetrieval results (Euclidean BAU, resnet50, $d_E$):")
    for sev in sorted(eucl_retrieval.keys()):
        r = eucl_retrieval[sev]
        print(f"  Gallery severity {sev}: mAP={r['mAP']:.4f}, Rank-1={r['Rank1']:.4f}")

    print("\nRetrieval results (Finsler BAU, $d_F$ on full embedding):")
    for sev in sorted(finsler_retrieval.keys()):
        r = finsler_retrieval[sev]
        print(f"  Gallery severity {sev}: mAP={r['mAP']:.4f}, Rank-1={r['Rank1']:.4f}")

    # Save retrieval metrics
    retrieval_path = osp.join(args.output_dir, "retrieval_metrics.json")
    with open(retrieval_path, "w") as f:
        json.dump({
            "euclidean_bau_resnet50_dE": eucl_retrieval,
            "finsler_bau_dF": finsler_retrieval,
            "euclidean_checkpoint": args.euclidean_checkpoint,
            "finsler_checkpoint": args.finsler_checkpoint,
        }, f, indent=2)
    print(f"\nSaved retrieval metrics to {retrieval_path}")

    # Paper hooks: mAP curve + corruption strip (shared styling with regenerate_toy_figures.py)
    retrieval_fig_path = osp.join(args.output_dir, "fig_retrieval_mAP.pdf")
    make_retrieval_mAP_plot(eucl_retrieval, finsler_retrieval, retrieval_fig_path)
    print(f"Saved retrieval mAP figure to {retrieval_fig_path}")

    strip_path = osp.join(args.output_dir, "fig_corruption_strip.pdf")
    make_corruption_strip(args.dataset_dir, strip_path)
    print(f"Saved corruption strip to {strip_path}")

    omega_strip_path = osp.join(args.output_dir, "fig_paper_corruption_strip.pdf")
    plot_paper_corruption_strip(
        finsler_features,
        meta,
        args.dataset_dir,
        omega_strip_path,
        identity_dim=identity_dim,
    )

    # -----------------------------------------------------------------------
    # 4b. v3.0 asymmetry diagnostics (Points 1a / 1b / 2 / 3)
    # -----------------------------------------------------------------------
    if is_v3:
        print("\n" + "=" * 60)
        print("v3.0 diagnostics: Points 1a, 1b, 2, 3 (see changelogs/toy_dataset_asymmetry_diagnostics.md)")

        # --- Point 1a: cross-source, cross-severity retrieval -----------------
        print("  [1a] Cross-source, cross-severity 5x5 mAP matrix (d_E and d_F)...")
        p1a_dE = compute_cross_source_severity_retrieval(
            eucl_features, meta, identity_dim=identity_dim, scoring="dE_full",
        )
        p1a_dF = compute_cross_source_severity_retrieval(
            finsler_features, meta, identity_dim=identity_dim, scoring="dF_midpoint",
            dist_func=dist_func,
        )

        # Paper figures: mean mAP per direction, plus d_F asymmetry
        make_cross_severity_heatmap(
            p1a_dE["mAP_mean"],
            osp.join(args.output_dir, "fig_cross_severity_mAP_dE.pdf"),
            cbar_label="mAP (%)", cmap="viridis",
        )
        make_cross_severity_heatmap(
            p1a_dF["mAP_mean"],
            osp.join(args.output_dir, "fig_cross_severity_mAP_dF.pdf"),
            cbar_label="mAP (%)", cmap="viridis",
        )
        make_cross_severity_heatmap(
            p1a_dF["asymmetry_A"],
            osp.join(args.output_dir, "fig_cross_severity_asymmetry_A.pdf"),
            cbar_label=r"$A_{\sigma_q,\sigma_g}$ (pp)", diverging=True,
        )

        # Per-direction asymmetry matrices (source-averaging removed) so the
        # task-difficulty baseline in d_E and the Randers correction in d_F can
        # be read off without cross-source cancellation. Four PDFs total.
        for tag, p1a_block in (("dE", p1a_dE), ("dF", p1a_dF)):
            per_dir = p1a_block["mAP_per_direction"]
            for dir_key, mat in per_dir.items():
                arr = np.array(mat, dtype=np.float64)
                A_dir = arr - arr.T
                src = dir_key.replace("->", "to")  # "1to2" / "2to1"
                out_path = osp.join(
                    args.output_dir,
                    f"fig_cross_severity_asymmetry_A_{tag}_{src}.pdf",
                )
                make_cross_severity_heatmap(
                    A_dir.tolist(),
                    out_path,
                    cbar_label=r"$A_{\sigma_q,\sigma_g}$ (pp)",
                    diverging=True,
                )

        # --- Point 1b: same-source cross-severity retrieval (ablation) -------
        print("  [1b] Same-source cross-severity retrieval (ablation)...")
        p1b_dE = compute_same_source_severity_retrieval(
            eucl_features, meta, identity_dim=identity_dim, scoring="dE_full",
        )
        p1b_dF = compute_same_source_severity_retrieval(
            finsler_features, meta, identity_dim=identity_dim, scoring="dF_midpoint",
            dist_func=dist_func,
        )

        # --- H1c: pair-wise drift-shift Spearman ------------------------------
        print("  [H1c] Pair-wise drift-shift Spearman...")
        h1c = compute_pairwise_drift_shift_spearman(
            finsler_features, meta, identity_dim=identity_dim,
        )
        print(f"        rho = {h1c['rho']:.4f}, p = {h1c['pval']:.2e}, n = {h1c['n_pairs']}")

        # Extend retrieval_metrics.json with the v3 block and H1c
        with open(retrieval_path) as f:
            existing = json.load(f)
        existing.update({
            "cross_source_severity_dE": p1a_dE,
            "cross_source_severity_dF": p1a_dF,
            "same_source_severity_dE": p1b_dE,
            "same_source_severity_dF": p1b_dF,
            "drift_shift_spearman_H1c": h1c,
            "toy_dataset_version": dataset_version,
        })
        with open(retrieval_path, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"  Appended v3 retrieval tensors to {retrieval_path}")
        print(f"  |A|_max (d_F) = {p1a_dF['asymmetry_max_abs']:.4f} "
              f"(H1b falsifier threshold: 0.03)")

        # --- Point 2: drift cosine matrix ------------------------------------
        print(f"  [2] Drift cosine matrix (transport={args.omega_cos_transport})...")
        cos_result, _C, _keys, _pids, _srcs, _sevs = compute_drift_cosine_matrix(
            finsler_features, meta, identity_dim=identity_dim,
            transport=args.omega_cos_transport,
        )
        cos_path = osp.join(args.output_dir, "omega_cosine_blocks.json")
        with open(cos_path, "w") as f:
            json.dump(cos_result, f, indent=2)
        print(f"  Saved cosine block means + full matrix to {cos_path}")
        make_cosine_block_bar(
            cos_result["block_means"],
            osp.join(args.output_dir, "fig_omega_cosine_blocks.pdf"),
        )

        # --- Point 3: drift-orthogonal projection ---------------------------
        print("  [3] Drift-orthogonal projection of identity drift "
              f"(source_idx=1, shuffle_reps={args.projection_shuffle_reps})...")
        proj = compute_drift_orthogonal_projection(
            finsler_features, meta, identity_dim=identity_dim, source_idx=1,
            shuffle_reps=args.projection_shuffle_reps,
        )
        proj_path = osp.join(args.output_dir, "drift_orthogonal_projection.json")
        with open(proj_path, "w") as f:
            json.dump(proj, f, indent=2)
        print(f"  Saved per-pair projection + shuffle null to {proj_path}")
        # Figures
        make_drift_absorption_plot(
            proj["per_severity_k"],
            proj["analytic_null_B"],
            proj["analytic_null_C"],
            proj["shuffle_null"],
            osp.join(args.output_dir, "fig_drift_orthogonal_absorption.pdf"),
        )
        make_drift_alignment_plot(
            proj["per_severity_k"],
            proj["shuffle_null"],
            osp.join(args.output_dir, "fig_drift_alignment.pdf"),
        )
        # Report H3 + H4/H5 summaries
        for k in (1, 2, 3, 4):
            pk = proj["per_severity_k"].get(k) or proj["per_severity_k"].get(str(k))
            if pk and pk.get("n", 0) > 0:
                print(f"    k={k}: eta_B={pk['eta_B_mean']:.4f} (analytic null {proj['analytic_null_B']:.2e}), "
                      f"eta_C={pk['eta_C_mean']:.4f} (analytic null {proj['analytic_null_C']:.2e}), "
                      f"eta_ref={pk.get('eta_ref_mean', float('nan')):.4f}, "
                      f"cos(om0,omk)={pk.get('cos_omega_0_k_mean', float('nan')):.3f}, "
                      f"dF={pk.get('finsler_distance_mean', float('nan')):.4f}")

    # -----------------------------------------------------------------------
    # 5. Qualitative rank-list comparison
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Computing qualitative rank-list panels...")

    eucl_topk = compute_retrieval_topk(
        eucl_features,
        meta,
        identity_dim=identity_dim,
        top_k=5,
        baseline_euclidean_resnet50=True,
    )
    finsler_topk = compute_retrieval_topk(
        finsler_features,
        meta,
        identity_dim=identity_dim,
        dist_func=dist_func,
        use_full_embedding=True,
        top_k=5,
    )

    qual_path = osp.join(args.output_dir, "fig_retrieval_qualitative.pdf")
    plot_qualitative_retrieval(
        eucl_topk, finsler_topk, args.dataset_dir, qual_path,
        num_queries=4, top_k=5,
    )

    # -----------------------------------------------------------------------
    # 6. Copy hooks + optional full output mirror for paper/draft_4
    # -----------------------------------------------------------------------
    mirror = args.paper_mirror_dir.strip() if args.paper_mirror_dir else ""
    copy_paper_outputs(
        args.output_dir,
        args.paper_fig_dir,
        mirror_dir=mirror or None,
    )
    print(f"Copied hook PDFs → {args.paper_fig_dir}")
    if mirror:
        print(f"Mirrored output dir → {mirror}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    rho = drift_metrics["spearman_rho"]
    pval = drift_metrics["spearman_pval"]
    print(f"C1 (Identity invariance): identity distance from clean ranges "
          f"{drift_metrics[0]['identity_dist_mean']:.4f} to {drift_metrics[4]['identity_dist_mean']:.4f}")
    print(f"C2 (Drift monotonicity): Spearman rho={rho:.4f}, p={pval:.2e}")
    if rho > 0.5 and pval < 0.05:
        print("   -> POSITIVE: Drift magnitude increases monotonically with severity")
    elif rho < -0.5:
        print("   -> NEGATIVE: Drift magnitude decreases with severity (unexpected)")
    else:
        print("   -> INCONCLUSIVE: No strong monotonic relationship")

    # Compare Finsler BAU vs Euclidean BAU (independent checkpoints)
    fins_maps = [finsler_retrieval[s]["mAP"] for s in sorted(finsler_retrieval.keys())]
    eucl_maps = [eucl_retrieval[s]["mAP"] for s in sorted(eucl_retrieval.keys())]
    delta = np.mean(fins_maps) - np.mean(eucl_maps)
    print(f"C3 (Retrieval): Finsler BAU ($d_F$) avg mAP = {np.mean(fins_maps)*100:.1f}%, "
          f"Euclidean BAU ($d_E$) avg mAP = {np.mean(eucl_maps)*100:.1f}%, delta = {delta*100:+.1f}%")

    print(f"\nAll outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
