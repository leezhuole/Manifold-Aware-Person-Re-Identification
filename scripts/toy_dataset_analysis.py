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

Usage:
    python scripts/toy_dataset_analysis.py \
        --finsler-checkpoint /home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification/logs/finsler_primary/job_1521403_primary_unified_1c_w0.1_driftInst/best.pth \
        --euclidean-checkpoint logs/AGReIDv2_sweep/job_1502418_dt_cuhk03_EucBaseline/best.pth \
        --dataset-dir /home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification/examples/data/ToyCorruption \
        --output-dir results/toy_analysis \
        --viz-tsne

Low-dimensional drift plots (optional):
    --viz-num-identities 15 --viz-seed 42 \\
    --fig-omega-identity-pca fig_omega_identity_pca.pdf \\
    --viz-tsne --fig-omega-tsne fig_omega_tsne.pdf

Toy ``camera'' IDs follow ``scripts/generate_toy_dataset.py``: cam_id = severity + 1 (corruption level).
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

from toy_paper_figures import copy_paper_outputs, make_corruption_strip, make_retrieval_mAP_plot


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

class ToyDataset(torch.utils.data.Dataset):
    """Minimal dataset that returns (img_tensor, fname, pid, camid)."""

    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        pattern = re.compile(r"(\d+)_c(\d)")
        self.samples = []
        for fname in sorted(os.listdir(image_dir)):
            if not fname.endswith(".jpg"):
                continue
            m = pattern.search(fname)
            if not m:
                continue
            pid = int(m.group(1))
            camid = int(m.group(2))
            self.samples.append((fname, pid, camid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, pid, camid = self.samples[idx]
        img = Image.open(osp.join(self.image_dir, fname)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, fname, pid, camid


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
    """Extract features and metadata from a data loader."""
    model.eval()
    features = OrderedDict()
    meta = OrderedDict()  # fname -> (pid, camid, severity)

    with torch.no_grad():
        for imgs, fnames, pids, camids in data_loader:
            imgs = imgs.cuda()
            outputs = model(imgs)
            outputs = outputs.data.cpu()
            for fname, output, pid, camid in zip(fnames, outputs, pids, camids):
                features[fname] = output
                severity = int(camid) - 1  # camid 1-5 -> severity 0-4
                meta[fname] = {"pid": int(pid), "camid": int(camid), "severity": severity}

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
                "dist": float(dists[j]),
                "correct": int(gpid) == int(qpid),
            })
        results[qf] = {"query_pid": int(qpid), "topk": topk}

    return results


# ---------------------------------------------------------------------------
# Low-dimensional drift / identity visualization (PCA, optional t-SNE)
# ---------------------------------------------------------------------------

def subsample_pids_with_full_severity(meta, num_identities, seed):
    """Choose PIDs that have all severity levels 0..4 (toy: cam_id = severity + 1)."""
    pid_sev = defaultdict(set)
    for m in meta.values():
        pid_sev[m["pid"]].add(int(m["severity"]))
    eligible = []
    need = {0, 1, 2, 3, 4}
    for pid in sorted(pid_sev.keys()):
        if need.issubset(pid_sev[pid]):
            eligible.append(pid)
    if not eligible:
        return []
    rng = np.random.RandomState(int(seed))
    k = min(int(num_identities), len(eligible))
    chosen = rng.choice(eligible, size=k, replace=False).tolist()
    return sorted(chosen)


def gather_identity_drift_arrays(features, meta, pids, identity_dim=2048):
    """Stack identity and drift slices for selected PIDs × severities 0..4 (row order: pid, then s)."""
    pid_feats = defaultdict(dict)
    for fname, feat in features.items():
        m = meta[fname]
        pid_feats[m["pid"]][m["severity"]] = feat

    z_id_list, omega_list, sev_list, pid_list = [], [], [], []
    for pid in pids:
        for s in range(5):
            if s not in pid_feats[pid]:
                raise KeyError(f"Missing severity {s} for pid {pid}")
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
    query_dir = osp.join(dataset_dir, "query")
    all_dir = osp.join(dataset_dir, "bounding_box_test")

    def load_img(fname):
        for d in [query_dir, gallery_dir, all_dir]:
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
                sev = int(item["camid"]) - 1
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
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

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
    all_dataset = ToyDataset(osp.join(args.dataset_dir, "bounding_box_test"), transform=transform)
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
