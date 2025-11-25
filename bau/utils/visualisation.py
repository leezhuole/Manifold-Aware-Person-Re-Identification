import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.manifold import TSNE
import umap

def visualization_sampler(embeddings, labels, k=None, n=1000, is_query=None):
    """
    Sample embeddings and labels for visualization.

    Parameters:
    - embeddings: The feature vectors to sample from. [shape: (num_samples, feature_dim)]
    - labels: The corresponding labels for the embeddings. (i.e., person IDs) [shape: (num_samples,)]
    - k: Maximum number of samples per identity. If None, all samples are used.
    - n: Minimum number of identities to sample.
    - is_query: Boolean array indicating if the sample is a query. [shape: (num_samples,)]
    
    Outputs:
    - selected_embeddings: Sampled feature vectors. [shape: (k*n, feature_dim)]
    - selected_labels: Corresponding labels for the sampled feature vectors. [shape: (k*n,)]
    - selected_is_query: (Optional) Boolean array indicating if the sampled vector is a query. [shape: (k*n,)]
    """
    unique_labels = np.unique(labels)
    selected_labels = np.random.choice(unique_labels, size=min(n, len(unique_labels)), replace=False)
    
    selected_embeddings = []
    selected_labels_list = []
    selected_is_query_list = []
    
    for label in selected_labels:
        indices = np.where(labels == label)[0]
        if k is not None:
            selected_indices = np.random.choice(indices, size=min(k, len(indices)), replace=False)
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


def visualize_embeddings(embeddings, labels, k=None, n=1000, seed=42, is_query=None):
    """
    Visualize the embeddings in the latent space using t-SNE.
    
    Parameters:
    - embeddings: The feature vectors to visualize. [shape: (num_samples, feature_dim)]
    - labels: The corresponding labels for the embeddings. [shape: (num_samples,)]
    - k: Number of samples per identity to visualize.
    - n: Number of identities to randomly select for visualization.
    - is_query: Boolean array indicating if the sample is a query. [shape: (num_samples,)]
    
    Outputs:
    - A t-SNE plot of the selected embeddings colored by their labels.

    """
    # Sample embeddings and labels
    selected_embeddings, selected_labels_list, selected_is_query = visualization_sampler(embeddings, labels, k, n, is_query)

    # Apply UMAP
    plane_mapper = umap.UMAP(random_state=seed).fit(selected_embeddings)

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    if is_query is not None and selected_is_query is not None:
        vmin, vmax = selected_labels_list.min(), selected_labels_list.max()
        
        # Plot gallery (is_query == False)
        gallery_mask = ~selected_is_query
        ax.scatter(plane_mapper.embedding_.T[0][gallery_mask], plane_mapper.embedding_.T[1][gallery_mask], 
                   c=selected_labels_list[gallery_mask], cmap='Spectral', marker='o', label='Gallery', vmin=vmin, vmax=vmax, alpha=0.3)
        
        # Plot query (is_query == True)
        query_mask = selected_is_query
        ax.scatter(plane_mapper.embedding_.T[0][query_mask], plane_mapper.embedding_.T[1][query_mask], 
                   c=selected_labels_list[query_mask], cmap='Spectral', marker='x', label='Query', vmin=vmin, vmax=vmax, alpha=1.0)
        
        ax.legend()
    else:
        ax.scatter(plane_mapper.embedding_.T[0], plane_mapper.embedding_.T[1], c=selected_labels_list, cmap='Spectral')

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
    selected_embeddings, selected_labels_list, selected_is_query = visualization_sampler(embeddings, labels, k, n, is_query)

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