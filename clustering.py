# clustering.py

from sklearn.cluster import KMeans
import torch
import pandas as pd
import os

def run_kmeans_and_get_labels(embeddings, n_clusters, device='cpu', random_state=0):
    """
    Run KMeans clustering on embeddings and return cluster labels as tensor.

    Args:
        embeddings: [N, D] numpy array or tensor
        n_clusters: int
        device: str or torch.device

    Returns:
        labels: [N] long tensor on specified device
    """
    if isinstance(embeddings, torch.Tensor):
        embs_np = embeddings.detach().cpu().numpy()
    else:
        embs_np = embeddings

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    labels_np = kmeans.fit_predict(embs_np)
    labels = torch.tensor(labels_np).long().to(device)

    return labels

