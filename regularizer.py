# regularizer.py

import torch


def compute_lw_from_batch(embeddings, cluster_labels):
    """
    【旧方法，保留备用】
    Compute Lw = sum_k [ (sum of squared distances to mean center in class k) / N_k ]

    Args:
        embeddings: [B, D] tensor, sample embeddings in current batch
        cluster_labels: [B] long tensor, pre-assigned cluster ID for each sample

    Returns:
        Lw: scalar tensor, differentiable
    """
    unique_labels = torch.unique(cluster_labels)
    Lw = torch.tensor(0.0, device=embeddings.device)

    for label in unique_labels:
        mask = (cluster_labels == label)
        group_embs = embeddings[mask]  # [N_k, D]
        N_k = group_embs.size(0)
        if N_k <= 1:
            continue  # skip singleton groups

        center = group_embs.mean(dim=0)  # [D]
        dist_sq_sum = (group_embs - center).pow(2).sum()  # sum of squared distances
        Lw_k = dist_sq_sum / N_k  # normalize by count
        Lw += Lw_k

    return Lw


# =================================================================
# MODIFIED L_B FUNCTION (Simplex Equiangular Loss)
# =================================================================

def compute_LB_differentiable(centroids):
    """
    Compute inter-cluster dispersion loss using Simplex Equiangular (ETF) loss.
    This loss encourages the centroids to form a simplex structure, where
    cos(c_i, c_j) -> -1/(K-1) for all i != j.

    L_B = mean( (cos(c_i, c_j) - (-1/(K-1)))^2 ) for all i < j

    Args:
        centroids: [K, D] tensor, cluster centers (differentiable path)

    Returns:
        L_B: scalar tensor
    """
    K = centroids.size(0)
    # If K=1, there is no inter-cluster loss.
    if K <= 1:
        return torch.tensor(0.0, device=centroids.device)

    # Target cosine similarity for a simplex structure
    # This is the target for all off-diagonal pairs
    target_cos = -1.0 / (K - 1)

    # Normalize centroids to unit vectors for easy cosine similarity calculation
    # [K, D]
    centroids_norm = torch.nn.functional.normalize(centroids, p=2, dim=1)

    # Compute pairwise cosine similarity matrix: [K, K]
    # sim_matrix[i, j] = cos(c_i_norm, c_j_norm)
    sim_matrix = torch.matmul(centroids_norm, centroids_norm.t())

    # Create a mask to select only the upper triangle (pairs where i < j)
    # This avoids double counting and self-comparisons (diagonal)
    mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()

    # Get all unique pairwise similarities (i < j)
    off_diagonal_sims = sim_matrix[mask]

    # Compute the mean squared error from the target cosine
    losses = (off_diagonal_sims - target_cos).pow(2)
    L_B = losses.mean()

    # Handle potential NaN if K was 1 (though we checked)
    if torch.isnan(L_B):
        return torch.tensor(0.0, device=centroids.device)

    return L_B


# =================================================================
# THIS FUNCTION REMAINS THE SAME
# =================================================================

def compute_LB_from_embs(embeddings, labels, K):
    """
    Build centroids from embeddings and fixed labels, then compute differentiable L_B.

    Args:
        embeddings: [N, D] tensor, e.g., net.student_emb.weight
        labels:     [N]    long tensor, cluster assignment (fixed per epoch)
        K:          int, number of clusters

    Returns:
        L_B: scalar tensor
    """
    device = embeddings.device
    centroids = []

    for k in range(K):
        mask = (labels == k)
        if mask.sum() == 0:
            # Handle empty cluster: use small noise
            # Note: This might slightly affect the simplex loss,
            # but is necessary to avoid errors.
            centroid = torch.randn(embeddings.size(1), device=device) * 0.01
        else:
            centroid = embeddings[mask].mean(dim=0)  # [D]
        centroids.append(centroid)

    centroids = torch.stack(centroids, dim=0)  # [K, D]

    # Call the NEW L_B function
    return compute_LB_differentiable(centroids)