import numpy as np
import matplotlib
matplotlib.use('Agg')           # non-interactive backend (safe for headless / cluster)
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE
import umap
from PIL import Image
import os.path as osp

def visualization_sampler(embeddings, labels, k=None, n=100, is_query=None, seed=42):
    """
    Sample embeddings and labels for visualization.

    Parameters:
    - embeddings: The feature vectors to sample from. [shape: (num_samples, feature_dim)]
    - labels: The corresponding labels for the embeddings. (i.e., person IDs) [shape: (num_samples,)]
    - k: Maximum number of samples per identity. If None, all samples are used.
    - n: Minimum number of identities to sample.
    - is_query: Boolean array indicating if the sample is a query. [shape: (num_samples,)]
    - seed: Seed for the private RNG used for sampling. Ensures identical
      selections across runs independent of the global RNG state.
    
    Outputs:
    - selected_embeddings: Sampled feature vectors. [shape: (k*n, feature_dim)]
    - selected_labels: Corresponding labels for the sampled feature vectors. [shape: (k*n,)]
    - selected_is_query: (Optional) Boolean array indicating if the sampled vector is a query. [shape: (k*n,)]
    """
    rng = np.random.default_rng(seed)
    unique_labels = np.unique(labels)
    selected_labels = rng.choice(unique_labels, size=min(n, len(unique_labels)), replace=False)
    
    selected_embeddings = []
    selected_labels_list = []
    selected_is_query_list = []
    
    for label in selected_labels:
        indices = np.where(labels == label)[0]
        if k is not None:
            selected_indices = rng.choice(indices, size=min(k, len(indices)), replace=False)
        else:
            selected_indices = indices
        selected_embeddings.append(embeddings[selected_indices])
        selected_labels_list.extend([label] * len(selected_indices))
        if is_query is not None:
            selected_is_query_list.extend(is_query[selected_indices])
    
    selected_embeddings = np.vstack(selected_embeddings)
    selected_labels_list = np.array(selected_labels_list)

    if is_query is not None:
        selected_is_query_list = np.array(selected_is_query_list, dtype=bool)
        return selected_embeddings, selected_labels_list, selected_is_query_list

    return selected_embeddings, selected_labels_list, None


def visualize_embeddings(embeddings, labels, k=10, n=50, seed=42, is_query=None,
                         n_neighbors=30, min_dist=0.05, spread=1.0, metric='cosine',
                         normalize=True, point_size=18, color_by='pid', normalize_pid_colors=True):
    """
    Visualize the embeddings in the latent space using UMAP.
    
    Parameters:
    - embeddings: The feature vectors to visualize. [shape: (num_samples, feature_dim)]
    - labels: The corresponding labels for the embeddings. [shape: (num_samples,)]
    - k: Number of samples per identity to visualize.
    - n: Number of identities to randomly select for visualization.
        - is_query: Boolean array indicating if the sample is a query. [shape: (num_samples,)]
        - color_by: 'pid' (color by identity) or 'split' (color by query/gallery).
        - normalize_pid_colors: If True and color_by='pid', remap the sampled PIDs to
            a contiguous range [0..K-1] so the colormap uses its full dynamic range.
    
    Outputs:
    - A UMAP plot of the selected embeddings.

    """
    # Sample embeddings and labels
    selected_embeddings, selected_labels_list, selected_is_query = visualization_sampler(embeddings, labels, k, n, is_query, seed=seed)

    # Optional L2 normalization to emphasize angular structure
    if normalize:
        norms = np.linalg.norm(selected_embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        selected_embeddings = selected_embeddings / norms

    # Apply UMAP with clustering-friendly settings
    plane_mapper = umap.UMAP(
        random_state=seed,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        metric=metric,
        init='spectral',
    ).fit(selected_embeddings)

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    if is_query is not None and selected_is_query is not None:
        gallery_mask = ~selected_is_query
        query_mask = selected_is_query

        if color_by == 'pid':
            if normalize_pid_colors:
                unique_pids = np.unique(selected_labels_list)
                pid_to_idx = {pid: idx for idx, pid in enumerate(unique_pids)}
                color_vals = np.vectorize(pid_to_idx.get)(selected_labels_list)
                vmin, vmax = 0, max(len(unique_pids) - 1, 0)
            else:
                color_vals = selected_labels_list
                vmin, vmax = selected_labels_list.min(), selected_labels_list.max()

            # Plot gallery (colored by PID)
            ax.scatter(
                plane_mapper.embedding_[:, 0][gallery_mask],
                plane_mapper.embedding_[:, 1][gallery_mask],
                c=color_vals[gallery_mask],
                cmap='Spectral',
                marker='o',
                label='Gallery',
                vmin=vmin, vmax=vmax,
                alpha=0.45,
                s=point_size,
                linewidths=0.0,
                zorder=2,
            )

            # Plot query as crosses but with the SAME PID colour as gallery
            ax.scatter(
                plane_mapper.embedding_[:, 0][query_mask],
                plane_mapper.embedding_[:, 1][query_mask],
                c=color_vals[query_mask],
                cmap='Spectral',
                marker='x',
                label='Query',
                vmin=vmin, vmax=vmax,
                alpha=0.7,
                s=point_size * 0.9,
                linewidths=0.8,
                zorder=3,
            )
        else:
            # Color by split (query vs gallery)
            ax.scatter(
                plane_mapper.embedding_[:, 0][gallery_mask],
                plane_mapper.embedding_[:, 1][gallery_mask],
                c='#1f77b4',
                marker='o',
                label='Gallery',
                alpha=0.5,
                s=point_size,
                linewidths=0.0,
                zorder=2,
            )
            ax.scatter(
                plane_mapper.embedding_[:, 0][query_mask],
                plane_mapper.embedding_[:, 1][query_mask],
                c='#d62728',
                marker='x',
                label='Query',
                alpha=0.7,
                s=point_size * 0.9,
                linewidths=0.8,
                zorder=3,
            )

        ax.legend()
    else:
        if color_by == 'pid' and normalize_pid_colors:
            unique_pids = np.unique(selected_labels_list)
            pid_to_idx = {pid: idx for idx, pid in enumerate(unique_pids)}
            color_vals = np.vectorize(pid_to_idx.get)(selected_labels_list)
            vmin, vmax = 0, max(len(unique_pids) - 1, 0)
        else:
            color_vals = selected_labels_list
            vmin, vmax = None, None

        ax.scatter(
            plane_mapper.embedding_.T[0],
            plane_mapper.embedding_.T[1],
            c=color_vals,
            cmap='Spectral',
            s=point_size,
            alpha=0.7,
            linewidths=0.0,
            vmin=vmin,
            vmax=vmax,
        )

    # plt.colorbar(scatter, label='Identity')
    ax.set_title('UMAP Visualization of Latent Embeddings')
    ax.set_xlabel('UMAP Component 1')
    ax.set_ylabel('UMAP Component 2')
    ax.grid(True)
    
    return ax


def visualize_hyperbolic_embeddings(embeddings, labels, manifold, k=None, n=1000, seed=42, is_query=None):
    """
    Visualize hyperbolic embeddings in the Poincaré ball model via UMAP.
    
    Parameters:
    - embeddings: The hyperbolic feature vectors to visualize. [shape: (num_samples, feature_dim)]
    - labels: The corresponding labels for the embeddings. [shape: (num_samples,)]
    - k: Number of samples per identity to visualize.
    - n: Number of identities to randomly select for visualization.
    - manifold: The manifold object used for hyperbolic geometry.
    - is_query: Boolean array indicating if the sample is a query. [shape: (num_samples,)]
    
    Outputs:
    - A plot of the selected hyperbolic embeddings in the Poincaré ball colored by their labels.
    
    References:
    - UMAP for non-Euclidean embeddings: https://umap-learn.readthedocs.io/en/latest/embedding_space.html
    """
    # Sample embeddings and labels
    selected_embeddings, selected_labels_list, selected_is_query = visualization_sampler(embeddings, labels, k, n, is_query, seed=seed)

    hyperbolic_mapper = umap.UMAP(output_metric='hyperboloid',random_state=seed).fit(selected_embeddings)
    
    # Extract x and y coordinates + solve for z coordinate
    x = hyperbolic_mapper.embedding_[:, 0]
    y = hyperbolic_mapper.embedding_[:, 1]
    z = np.sqrt(1 + np.sum(hyperbolic_mapper.embedding_**2, axis=1))

    # Map the data into the Poincare model
    disk_x = x / (1 + z)
    disk_y = y / (1 + z)

    # Plotting 
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    if is_query is not None and selected_is_query is not None:
        vmin, vmax = selected_labels_list.min(), selected_labels_list.max()
        
        # Plot gallery (is_query == False)
        gallery_mask = ~selected_is_query
        ax.scatter(disk_x[gallery_mask], disk_y[gallery_mask], c=selected_labels_list[gallery_mask], 
                   cmap='Spectral', marker='o', label='Gallery', vmin=vmin, vmax=vmax, alpha=0.3)
        
        # Plot query (is_query == True)
        query_mask = selected_is_query
        ax.scatter(disk_x[query_mask], disk_y[query_mask], c=selected_labels_list[query_mask], 
                   cmap='Spectral', marker='x', label='Query', vmin=vmin, vmax=vmax, alpha=1.0)
        
        ax.legend()
    else:
        ax.scatter(disk_x, disk_y, c=selected_labels_list, cmap='Spectral')
    
    # Draw the boundary of the Poincaré disk
    boundary = Circle((0,0), 1, fc='none', ec='k')
    ax.add_artist(boundary)

    # Set limits to ensure the whole disk is visible regardless of data distribution
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axis('off')

    ax.set_title('Hyperbolic Embedding Visualization (Poincaré Disk)')
    ax.set_xlabel('UMAP Component 1')
    ax.set_ylabel('UMAPComponent 2')
    ax.grid(True)

    return ax


# ---------------------------------------------------------------------------
#  Qualitative retrieval visualisation
# ---------------------------------------------------------------------------

def _load_image(img_path, root=None, size=(128, 256)):
    """Load an image from *img_path* and resize to (W, H) = *size*.

    Parameters
    ----------
    img_path : str
        Absolute or relative path to the image file.
    root : str or None
        Optional directory prefix (joined when *img_path* is relative).
    size : tuple[int, int]
        Target ``(width, height)`` in pixels.

    Returns
    -------
    np.ndarray
        RGB image array of shape ``(H, W, 3)`` in ``[0, 255]``.
    """
    fpath = img_path if root is None else osp.join(root, img_path)
    img = Image.open(fpath).convert('RGB')
    img = img.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(img)


def _add_border(img, color, border_width=4):
    """Return a copy of *img* with a solid colour border.

    Parameters
    ----------
    img : np.ndarray
        ``(H, W, 3)`` uint8 image.
    color : tuple[int, int, int]
        RGB colour for the border.
    border_width : int
        Border thickness in pixels.

    Returns
    -------
    np.ndarray
        Bordered image with the same dtype.
    """
    bordered = img.copy()
    bw = border_width
    bordered[:bw, :] = color          # top
    bordered[-bw:, :] = color         # bottom
    bordered[:, :bw] = color          # left
    bordered[:, -bw:] = color         # right
    return bordered


def visualize_retrieval(distmat, query, gallery, images_dir=None,
                        num_queries=4, top_k=5, seed=1):
    """Generate a qualitative retrieval figure.

    For each of *num_queries* randomly (but deterministically) selected query
    images, show the query on the left and its *top_k* nearest gallery
    retrievals on the right.  Correct matches (same person ID) get a **green**
    border; incorrect ones get **red**.  Each retrieved image is annotated with
    a confidence score defined as:

    .. math::

        \\text{conf}(q, g) = \\frac{1}{1 + d(q, g)}

    This is a monotonically decreasing function of the distance ``d`` that
    the evaluation pipeline uses for ranking, so it faithfully reflects the
    model's inference-time decision criterion.  The value is normalised to
    ``[0, 1]`` and printed as a percentage.

    Parameters
    ----------
    distmat : np.ndarray
        Distance matrix of shape ``(num_query, num_gallery)`` — already
        computed by the evaluation pipeline.
    query : list[tuple]
        Query set as ``[(img_path, pid, camid), ...]``.
    gallery : list[tuple]
        Gallery set as ``[(img_path, pid, camid), ...]``.
    images_dir : str or None
        Root directory prepended to image paths (pass ``dataset.images_dir``).
    num_queries : int
        Number of query probes to visualise.
    top_k : int
        Number of gallery retrievals per query.
    seed : int
        Seed for the dedicated RNG that selects query indices.  Using a
        private ``numpy.random.Generator`` guarantees that the *same* query
        images are chosen across different runs and different epochs,
        independent of the global RNG state.

    Returns
    -------
    matplotlib.figure.Figure
        The retrieval figure (caller is responsible for saving / logging and
        closing it via ``plt.close(fig)``).
    """
    distmat = np.asarray(distmat)
    num_q, num_g = distmat.shape

    query_ids  = np.array([pid for _, pid, _ in query])
    query_cams = np.array([cam for _, _, cam in query])
    gallery_ids  = np.array([pid for _, pid, _ in gallery])
    gallery_cams = np.array([cam for _, _, cam in gallery])

    # --- deterministic query selection with a *private* RNG ----------------
    rng = np.random.default_rng(seed)
    selected = rng.choice(num_q, size=min(num_queries, num_q), replace=False)
    selected.sort()                     # canonical order for reproducibility

    nrows = len(selected)
    ncols = 1 + top_k                   # query + top-k gallery
    img_w, img_h = 128, 256             # width, height per thumbnail
    border_w = 4

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(2.0 * ncols, 4.0 * nrows),
                             gridspec_kw={'wspace': 0.05, 'hspace': 0.15})
    if nrows == 1:
        axes = axes[np.newaxis, :]      # ensure 2-D indexing

    for row, qi in enumerate(selected):
        q_path, q_pid, q_cam = query[qi]

        # ---- rank gallery for this query, respecting re-id protocol ------
        # Exclude gallery items with SAME pid AND SAME camera (standard
        # evaluation protocol used by cmc / mean_ap).
        dists = distmat[qi].copy()
        invalid = (gallery_ids == q_pid) & (gallery_cams == q_cam)
        dists[invalid] = np.inf
        rank_indices = np.argsort(dists)[:top_k]

        # ---- draw query image -------------------------------------------
        q_img = _load_image(q_path, root=images_dir, size=(img_w, img_h))
        q_img = _add_border(q_img, color=(0, 120, 255), border_width=border_w)   # blue
        axes[row, 0].imshow(q_img)
        axes[row, 0].set_title('Query', fontsize=9, fontweight='bold')
        axes[row, 0].set_ylabel(f'PID {q_pid}', fontsize=8)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        # ---- draw retrieved gallery images --------------------------------
        for col, gi in enumerate(rank_indices, start=1):
            g_path, g_pid, _ = gallery[gi]
            correct = (g_pid == q_pid)

            g_img = _load_image(g_path, root=images_dir, size=(img_w, img_h))
            border_colour = (0, 200, 0) if correct else (220, 0, 0)
            g_img = _add_border(g_img, color=border_colour, border_width=border_w)

            # Confidence: 1 / (1 + d).  Monotonically decreasing with d,
            # bounded in (0, 1], directly tied to the ranking criterion.
            d = float(dists[gi])
            conf = 1.0 / (1.0 + d)

            axes[row, col].imshow(g_img)
            axes[row, col].set_title(f'Conf: {conf:.0%}', fontsize=8)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    fig.suptitle('Qualitative Retrieval (Top-{})'.format(top_k),
                 fontsize=12, fontweight='bold', y=1.0)
    fig.subplots_adjust(wspace=0.05, hspace=0.15)
    return fig