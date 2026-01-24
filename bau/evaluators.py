from __future__ import print_function, absolute_import
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
from .utils.visualisation import visualize_embeddings, visualize_hyperbolic_embeddings
import geoopt
# from pykeops.torch import LazyTensor
import wandb
from .loss.triplet import euclidean_dist

# def pairwise_distance_keops(x, y, manifold):
#     """
#     Computes distance matrix using Symbolic KeOps.
#     Memory Usage: O(M + N) instead of O(M * N)
#     """
#     # 1. Wrap tensors as LazyTensors
#     # 'i' = indexed by query (rows), 'j' = indexed by gallery (cols)
#     X_i = LazyTensor(x.unsqueeze(1))  # Shape (M, 1, D)
#     Y_j = LazyTensor(y.unsqueeze(0))  # Shape (1, N, D)

#     # 2. Define the Manifold Distance Symbolically
#     # This does NOT compute anything yet. It just builds a formula.
    
#     if isinstance(manifold, geoopt.PoincareBall):
#         # TODO
#         pass

#     else:
#         raise NotImplementedError("Need to write formula for this manifold")

#     # 3. Trigger Computation
#     # The .solve() or .sum() or simply converting to torch triggers the CUDA kernel
#     # This computes the M*N matrix directly.
#     return dist_m # Returns a LazyTensor, call .cpu() or .numpy() to materialize


def extract_cnn_feature(model, inputs):
    inputs = to_torch(inputs).cuda()
    outputs = model(inputs)
    outputs = outputs.data.cpu()
    return outputs


def extract_features(model, data_loader, print_freq=50):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()
    end = time.time()
    with torch.no_grad():
        time.sleep(2)
        print("feature extraction ...")
        for i, (imgs, fnames, pids, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            outputs = extract_cnn_feature(model, imgs)
            # Move outputs to CPU immediately to save VRAM
            outputs = outputs.cpu()
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

    return features, labels


def pairwise_distance(features, query=None, gallery=None, manifold=None, alpha=None):
    """
    Manifold-aware pairwise distance computation.

    
    Note:
    1. We chunk the query set (block-wise computation) to avoid memory explosion when 
    computing distances on large datasets in hyperbolic space. If we have m queries, 
    n gallery items, and dimension d, broadcasting creates a tensor of shape (m, n, d). 

    """
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        if manifold is None:
            # if alpha is None:
            #     dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
            #     dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
            # else:
            #     dist_m = euclidean_dist(x, x, alpha=alpha)
            dist_m = euclidean_dist(x, x, alpha=alpha)
        else:
            dist_m = manifold.dist(x.unsqueeze(1), x.unsqueeze(0), dim=-1)
        return dist_m

    assert query is not None and gallery is not None, "query and gallery must be provided together"
    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)

    # 1. Handle Device Mismatch safely: Use a copy to avoid side-effects on the training model
    # We want to use the GPU if available, so we ensure the manifold is on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if manifold is not None and hasattr(manifold, 'k') and isinstance(manifold.k, torch.Tensor):
        if manifold.k.device != device:
            manifold = copy.deepcopy(manifold).to(device)

    if manifold is None:
        # if alpha is None:
        #     dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #              torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        #     dist_m.addmm_(mat1=x, mat2=y.t(), beta=1, alpha=-2)
        # else:
        #     dist_m = euclidean_dist(x, y, alpha=alpha)
        dist_m = euclidean_dist(x, y, alpha=alpha)
    else:
        # 2. Handle Memory Explosion: Compute distances in blocks ON GPU
        # We chunk BOTH query and gallery to keep intermediate tensors small.
        # Block size: 100 queries x 1000 gallery items
        q_batch_size = 100
        g_batch_size = 1000
        
        # The final result matrix stays on CPU to save VRAM
        dist_m = torch.zeros((m, n), device='cpu', dtype=x.dtype)
        print('Computing hyperbolic distance matrix in blocks ({}x{}) on {}...'.format(q_batch_size, g_batch_size, device))
        
        for i in range(0, m, q_batch_size):
            q_end = min(i + q_batch_size, m)
            # Move only the current query batch to GPU
            x_batch = x[i:q_end].to(device)
            
            # Verbose logging: Print progress every 500 queries
            if i % 500 == 0:
                 print('  Processing query rows {:>5}-{:<5} / {}'.format(i, q_end, m))

            for j in range(0, n, g_batch_size):
                g_end = min(j + g_batch_size, n)
                # Move only the current gallery batch to GPU
                y_batch = y[j:g_end].to(device)
                
                # Compute distance on GPU
                dist_batch = manifold.dist(x_batch.unsqueeze(1), y_batch.unsqueeze(0), dim=-1)
                
                # Move result back to CPU immediately to free VRAM
                dist_m[i:q_end, j:g_end] = dist_batch.cpu()

                # Explicitly delete intermediate tensors and free GPU memory 
                del dist_batch, y_batch
                torch.cuda.empty_cache()

            # Explicitly delete intermediate tensors and free GPU memory
            del x_batch
            torch.cuda.empty_cache()

    return dist_m, x.detach().cpu().numpy(), y.detach().cpu().numpy()


def evaluate_all(query_features, gallery_features, distmat, visfig, query=None, gallery=None,
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
        return mAP, visfig

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
    return cmc_scores['market1501'], mAP, visfig


class Evaluator(object):
    def __init__(self, model, alpha_module=None):
        super(Evaluator, self).__init__()
        self.model = model
        module = getattr(model, 'module', model)
        self.manifold = getattr(module, 'manifold', None)
        self.alpha_module = alpha_module

    def evaluate(self, data_loader, query, gallery, cmc_flag=False, rerank=False, cmc_topk=(1, 5, 10)):
        print('Start evaluation ...')
        features, labels = extract_features(self.model, data_loader)
        alpha_value = self.alpha_module.value() if self.alpha_module is not None else None
        distmat, query_features, gallery_features = pairwise_distance(
            features, query, gallery, manifold=self.manifold, alpha=alpha_value
        )
        distmat = distmat.cpu()
        
        # Visualize the embeddings (Note that we do not differentiate query and gallery for now)
        if self.manifold is not None: 
            vis_fig = visualize_hyperbolic_embeddings(
                np.vstack([query_features, gallery_features]),
                np.array([pid for _, pid, _ in query] + [pid for _, pid, _ in gallery]),
                manifold=self.manifold,
                is_query=np.concatenate([
                    np.ones(len(query_features), dtype=bool), 
                    np.zeros(len(gallery_features), dtype=bool)
                ]),
                k=None,
                n=1000,
                seed=42
            )

        else:
            vis_fig = visualize_embeddings(
                np.vstack([query_features, gallery_features]),
                np.array([pid for _, pid, _ in query] + [pid for _, pid, _ in gallery]),
                is_query=np.concatenate([
                    np.ones(len(query_features), dtype=bool), 
                    np.zeros(len(gallery_features), dtype=bool)
                ]),
                k=None,
                n=1000,
                seed=42
            )

        results = evaluate_all(query_features, gallery_features, distmat, vis_fig, query=query, gallery=gallery, cmc_flag=cmc_flag, cmc_topk=cmc_topk)

        if (not rerank):
            return results

        if self.manifold is not None:
            print('Re-ranking is disabled for hyperbolic embeddings.')
            return results

        print('Applying person re-ranking ...')
        distmat_qq, _, _ = pairwise_distance(features, query, query, manifold=self.manifold, alpha=alpha_value)
        distmat_gg, _, _ = pairwise_distance(features, gallery, gallery, manifold=self.manifold, alpha=alpha_value)
        distmat = re_ranking(distmat.cpu().numpy(), distmat_qq.cpu().numpy(), distmat_gg.cpu().numpy())
        
        return evaluate_all(query_features, gallery_features, distmat, vis_fig, query=query, gallery=gallery, cmc_flag=cmc_flag, cmc_topk=cmc_topk)