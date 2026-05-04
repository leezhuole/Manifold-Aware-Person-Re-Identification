from __future__ import print_function, absolute_import
import os.path as osp
import re
import time
import collections
from collections import OrderedDict
import numpy as np
import torch
import random
import copy

from .evaluation_metrics import cmc, mean_ap
from .utils.meters import AverageMeter
from .utils.rerank import re_ranking
from .utils import to_torch
from .utils.randers import randers_distance_matrix


def toy_severity_from_reid_fname(fname):
    """Parse toy severity from ``{{pid}}_c{{src}}s{{severity+1}}_000001_01.jpg`` basename."""
    base = osp.basename(fname)
    m = re.search(r"_c\d+s(\d+)_", base)
    if m is None:
        raise ValueError("Cannot parse severity from toy filename: {}".format(base))
    return int(m.group(1)) - 1


def spearman_rho_theta_severity(theta_dict, paths_in_order):
    """Spearman ρ between θ (from ``theta_dict``) and filename-derived severity."""
    try:
        from scipy.stats import spearmanr
    except ImportError:
        spearmanr = None
    thetas = []
    sevs = []
    for p in paths_in_order:
        k = _resolve_feature_key(theta_dict, p)
        thetas.append(float(theta_dict[k]))
        sevs.append(toy_severity_from_reid_fname(p))
    x = np.asarray(thetas, dtype=np.float64)
    y = np.asarray(sevs, dtype=np.float64)
    if spearmanr is not None:
        rho, _ = spearmanr(x, y)
        return float(rho)
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _resolve_feature_key(features_or_theta, fpath):
    if fpath in features_or_theta:
        return fpath
    n = osp.normpath(fpath)
    if n in features_or_theta:
        return n
    base = osp.basename(fpath)
    for k in features_or_theta:
        if osp.basename(k) == base:
            return k
    raise KeyError("No feature entry for path: {}".format(fpath))


def extract_cnn_feature(model, inputs, return_theta=False):
    params = list(model.parameters())
    device = params[0].device if params else torch.device("cpu")
    inputs = to_torch(inputs).to(device)
    outputs = model(inputs)
    if return_theta:
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
            raise TypeError(
                "return_theta=True requires eval forward to return (f_norm, theta); "
                "got {!r}".format(type(outputs))
            )
        f_out, theta_out = outputs
        f_out = f_out.detach().cpu()
        theta_out = theta_out.detach().cpu().view(theta_out.size(0), -1).squeeze(-1)
        return f_out, theta_out
    if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
        # Eval with θ head: store identity slice only unless caller asked for θ above.
        outputs = outputs[0]
    return outputs.detach().cpu()


def extract_features(
    model, data_loader, print_freq=50, return_theta=False, pre_extract_sleep=0.0
):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()
    thetas = OrderedDict() if return_theta else None
    end = time.time()
    with torch.no_grad():
        if pre_extract_sleep and float(pre_extract_sleep) > 0:
            time.sleep(float(pre_extract_sleep))
        print("feature extraction ...")
        for i, batch in enumerate(data_loader):
            data_time.update(time.time() - end)
            if len(batch) == 5:
                imgs, fnames, pids, _, _ = batch
            else:
                imgs, fnames, pids, _ = batch

            if return_theta:
                feats_b, theta_b = extract_cnn_feature(model, imgs, return_theta=True)
                for fname, output, pid, th in zip(
                    fnames, feats_b, pids, theta_b
                ):
                    features[fname] = output
                    labels[fname] = pid
                    thetas[fname] = float(th)
            else:
                outputs = extract_cnn_feature(model, imgs, return_theta=False)
                for fname, output, pid in zip(fnames, outputs, pids):
                    features[fname] = output
                    labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    if return_theta:
        return features, labels, thetas
    return features, labels


def _stack_identity_features(features, items):
    """Stack per-image identity vectors for ``items`` = list of (fname, pid, cam)."""
    if len(items) == 0:
        return torch.empty(0, 0)
    vecs = []
    for f, _, _ in items:
        k = _resolve_feature_key(features, f)
        vecs.append(features[k].view(1, -1))
    return torch.cat(vecs, dim=0)


def _stack_theta_for_items(theta_dict, items, ref_tensor):
    """Scalar θ per row; zeros if ``theta_dict`` is None (Euclidean-only path)."""
    if len(items) == 0:
        return torch.empty(0, dtype=ref_tensor.dtype, device=ref_tensor.device)
    if theta_dict is None:
        return torch.zeros(len(items), dtype=ref_tensor.dtype, device=ref_tensor.device)
    vals = [float(theta_dict[_resolve_feature_key(theta_dict, f)]) for f, _, _ in items]
    return torch.tensor(vals, dtype=ref_tensor.dtype, device=ref_tensor.device)


def _collect_d8_pairs(gallery_clean, gallery_corrupted, severity_levels=(1, 2, 3, 4)):
    """Collect (z^0, z^k) paths: same PID; z^0 from ``gallery_clean`` at severity 0; z^k from
    ``gallery_corrupted`` at severity ``k``. Camera IDs need not match (cross-view pairs
    allowed); all other matching rules unchanged.
    """
    pairs = []
    clean_rows = []
    for row in gallery_clean:
        path0, _pid0, _cam0 = row
        try:
            if toy_severity_from_reid_fname(path0) != 0:
                continue
        except ValueError:
            continue
        clean_rows.append(row)

    for path0, pid0, _cam0 in clean_rows:
        for k in severity_levels:
            matches = []
            for row in gallery_corrupted:
                pathk, pidk, _camk = row
                if pidk != pid0:
                    continue
                try:
                    sk = toy_severity_from_reid_fname(pathk)
                except ValueError:
                    continue
                if sk == k:
                    matches.append(pathk)
            for pathk in matches:
                pairs.append((path0, pathk, k))
    return pairs


def _scalar_randers_pair(f_q, theta_q, f_g, theta_g, alpha):
    """Single query→gallery Randers distance (same formula as ``randers_distance_matrix``)."""
    dist_m = randers_distance_matrix(
        f_q.view(1, -1),
        theta_q.view(-1),
        f_g.view(1, -1),
        theta_g.view(-1),
        alpha,
    )
    return dist_m[0, 0]


def compute_mean_asymmetric_gap_d8(
    features,
    theta_dict,
    gallery_clean,
    gallery_corrupted,
    alpha,
    severity_levels=(1, 2, 3, 4),
):
    """Mean of ``d_R(z^0, z^k) - d_R(z^k, z^0)`` over matched clean↔severity pairs.

    Pairs are built from ``gallery_clean`` (severity-0 rows) and ``gallery_corrupted`` with
    the same ``pid`` and filename-derived severity ``k ∈ severity_levels`` (any camera).

    When ``theta_dict`` is ``None``, θ is treated as zero and every gap is exactly zero
    (squared Euclidean term is symmetric for a fixed pair).

    Parameters
    ----------
    features : mapping
        Path → L2-normalized identity tensor (same as ``bidirectional_evaluate``).
    theta_dict : mapping or None
        Path → scalar θ; keys resolved like ``extract_features`` output.
    gallery_clean, gallery_corrupted : list of (path, pid, cam)
        Toy protocol: ``gallery_clean`` is typically the clean query set (e.g. ``query_s1``).

    Returns
    -------
    dict
        ``mean_gap`` — mean over all valid pairs; ``mean_gap_by_severity`` — ``k → mean``;
        ``n_pairs``, ``n_pairs_by_severity``; ``gaps`` omitted (large); empty input → NaN means.
    """
    pairs = _collect_d8_pairs(gallery_clean, gallery_corrupted, severity_levels=severity_levels)
    if not pairs:
        nan = float("nan")
        return {
            "mean_gap": nan,
            "mean_gap_by_severity": {int(k): nan for k in severity_levels},
            "n_pairs": 0,
            "n_pairs_by_severity": {int(k): 0 for k in severity_levels},
        }

    ref = features[_resolve_feature_key(features, pairs[0][0])]
    device, dtype = ref.device, ref.dtype

    by_k = collections.OrderedDict((int(k), []) for k in severity_levels)
    all_gaps = []

    for path0, pathk, k in pairs:
        k0 = _resolve_feature_key(features, path0)
        kk = _resolve_feature_key(features, pathk)
        f0 = features[k0].to(device=device, dtype=dtype).view(-1)
        fk = features[kk].to(device=device, dtype=dtype).view(-1)
        if theta_dict is None:
            t0 = torch.zeros(1, device=device, dtype=dtype)
            tk = torch.zeros(1, device=device, dtype=dtype)
        else:
            t0 = torch.tensor(
                [float(theta_dict[_resolve_feature_key(theta_dict, path0)])],
                device=device,
                dtype=dtype,
            )
            tk = torch.tensor(
                [float(theta_dict[_resolve_feature_key(theta_dict, pathk)])],
                device=device,
                dtype=dtype,
            )

        d_fwd = _scalar_randers_pair(f0, t0, fk, tk, alpha)
        d_rev = _scalar_randers_pair(fk, tk, f0, t0, alpha)
        gap = float((d_fwd - d_rev).item())
        all_gaps.append(gap)
        by_k[int(k)].append(gap)

    mean_by_sev = collections.OrderedDict()
    n_by_sev = collections.OrderedDict()
    for k in severity_levels:
        kk = int(k)
        lst = by_k.get(kk, [])
        n_by_sev[kk] = len(lst)
        mean_by_sev[kk] = float(np.mean(lst)) if lst else float("nan")

    return {
        "mean_gap": float(np.mean(all_gaps)),
        "mean_gap_by_severity": mean_by_sev,
        "n_pairs": len(all_gaps),
        "n_pairs_by_severity": n_by_sev,
    }


def _ranking_metrics(distmat, query, gallery, cmc_topk=(1, 5, 10)):
    query_ids = [pid for _, pid, _ in query]
    gallery_ids = [pid for _, pid, _ in gallery]
    query_cams = [cam for _, _, cam in query]
    gallery_cams = [cam for _, _, cam in gallery]
    m_ap = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    topk = max(cmc_topk) if len(cmc_topk) else 1
    cmc_vec = cmc(
        distmat,
        query_ids,
        gallery_ids,
        query_cams,
        gallery_cams,
        topk=topk,
        separate_camera_set=False,
        single_gallery_shot=False,
        first_match_break=True,
    )
    cmc_dict = {k: float(cmc_vec[k - 1]) for k in cmc_topk}
    return {"mAP": float(m_ap), "cmc": cmc_dict}


def bidirectional_evaluate(
    model,
    data_loader,
    query_clean,
    gallery_corrupted,
    query_corrupted,
    gallery_clean,
    alpha_values,
    return_theta=False,
    cmc_topk=(1, 5, 10),
    print_freq=50,
    verbose=True,
):
    """Bidirectional toy-protocol eval: clean↔corrupted with Randers distance.

    **Direction A** — clean queries, corrupted gallery: ``query_clean`` ×
    ``gallery_corrupted``. **Direction B** — corrupted queries, clean gallery:
    ``query_corrupted`` × ``gallery_clean``.

    ``data_loader`` must cover every image path appearing in the four lists
    (typically one ``DataLoader`` over the union of paths).

    Parameters
    ----------
    alpha_values : iterable of float
        Randers weights (include ``0.0`` for Euclidean via the Randers path;
        ordering matches squared L2 from ``pairwise_distance`` when θ is flat).

    return_theta : bool
        If True, ``model`` eval forward must return ``(f_norm, theta)`` and θ
        is passed into ``randers_distance_matrix``. If False, θ is treated as
        zero (Randers reduces to Euclidean for all α for ranking purposes).

    Returns
    -------
    dict
        ``euclidean`` — mAP / CMC for both directions using ``pairwise_distance``.
        ``randers`` — per-α dict with ``direction_A``, ``direction_B``,
        ``delta_mAP`` (= mAP_A − mAP_B).
        If ``return_theta`` is True, ``theta_by_fname`` maps each resolved image
        key to its scalar θ (same keys as in feature extraction).
    """
    if verbose:
        print("Bidirectional evaluation (clean ↔ corrupted) ...")
    if return_theta:
        features, _labels, theta_dict = extract_features(
            model, data_loader, print_freq=print_freq, return_theta=True
        )
    else:
        features, _labels = extract_features(
            model, data_loader, print_freq=print_freq, return_theta=False
        )
        theta_dict = None

    # Euclidean baseline (symmetric squared L2, same as standard BAU eval)
    dist_a, _, _ = pairwise_distance(features, query_clean, gallery_corrupted)
    dist_b, _, _ = pairwise_distance(features, query_corrupted, gallery_clean)
    euclid_a = _ranking_metrics(dist_a, query_clean, gallery_corrupted, cmc_topk)
    euclid_b = _ranking_metrics(dist_b, query_corrupted, gallery_clean, cmc_topk)
    out = {
        "euclidean": {
            "direction_A": euclid_a,
            "direction_B": euclid_b,
            "delta_mAP": euclid_a["mAP"] - euclid_b["mAP"],
        },
        "randers": {},
    }

    f_q_a = _stack_identity_features(features, query_clean)
    f_g_a = _stack_identity_features(features, gallery_corrupted)
    f_q_b = _stack_identity_features(features, query_corrupted)
    f_g_b = _stack_identity_features(features, gallery_clean)

    t_q_a = _stack_theta_for_items(theta_dict, query_clean, f_q_a)
    t_g_a = _stack_theta_for_items(theta_dict, gallery_corrupted, f_g_a)
    t_q_b = _stack_theta_for_items(theta_dict, query_corrupted, f_q_b)
    t_g_b = _stack_theta_for_items(theta_dict, gallery_clean, f_g_b)

    for alpha in alpha_values:
        alpha = float(alpha)
        r_a = randers_distance_matrix(f_q_a, t_q_a, f_g_a, t_g_a, alpha)
        r_b = randers_distance_matrix(f_q_b, t_q_b, f_g_b, t_g_b, alpha)
        ma = _ranking_metrics(r_a, query_clean, gallery_corrupted, cmc_topk)
        mb = _ranking_metrics(r_b, query_corrupted, gallery_clean, cmc_topk)
        d8_stats = compute_mean_asymmetric_gap_d8(
            features,
            theta_dict,
            gallery_clean,
            gallery_corrupted,
            alpha,
        )
        out["randers"][alpha] = {
            "direction_A": ma,
            "direction_B": mb,
            "delta_mAP": ma["mAP"] - mb["mAP"],
            "d8_mean_asymmetric_gap": d8_stats,
        }
    if return_theta and theta_dict is not None:
        out["theta_by_fname"] = theta_dict
    return out


def pairwise_distance(features, query=None, gallery=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist_m

    x = torch.cat(
        [features[_resolve_feature_key(features, f)].unsqueeze(0) for f, _, _ in query],
        0,
    )
    y = torch.cat(
        [features[_resolve_feature_key(features, f)].unsqueeze(0) for f, _, _ in gallery],
        0,
    )
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(x, y.t(), beta=1.0, alpha=-2.0)
    return dist_m, x.numpy(), y.numpy()


def evaluate_all(query_features, gallery_features, distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), cmc_flag=False):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    if (not cmc_flag):
        return mAP

    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True),}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores:')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'.format(k, cmc_scores['market1501'][k-1]))
    return cmc_scores['market1501'], mAP


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, cmc_flag=False, rerank=False, cmc_topk=(1, 5, 10)):
        print('Start evaluation ...')
        features, _ = extract_features(self.model, data_loader)
        distmat, query_features, gallery_features = pairwise_distance(features, query, gallery)
        results = evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag, cmc_topk=cmc_topk)

        if (not rerank):
            return results

        print('Applying person re-ranking ...')
        distmat_qq, _, _ = pairwise_distance(features, query, query)
        distmat_gg, _, _ = pairwise_distance(features, gallery, gallery)
        distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())
        return evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag, cmc_topk=cmc_topk)