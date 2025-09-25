import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import matplotlib.pyplot as plt


def as_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def l2_normalize(x, eps=1e-12):
    x = as_numpy(x).astype(np.float32, copy=False)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


def find_optimal_clusters_root(root_embeddings, max_k=15, random_state=33):
    """
    Choose K for root embeddings using elbow + cosine silhouette.
    Expects embeddings of shape [N, D] (torch.Tensor or np.ndarray).
    Returns (optimal_k, inertias, sils).
    """
    x = l2_normalize(root_embeddings)
    n = x.shape[0]
    if n < 3:
        print("Not enough roots to cluster (N < 3). Returning K=2.")
        return 2, [], [], [], None

    k_max = max(2, min(max_k, n - 1))
    k_range = list(range(2, k_max + 1))

    inertias = []
    silhouettes = []

    for k in tqdm(k_range, desc="Finding optimal K"):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(x)
        inertias.append(float(km.inertia_))
        # silhouette can fail if clusters collapse/singleton
        try:
            s = float(silhouette_score(x, km.labels_, metric="cosine"))
        except Exception:
            s = -1.0
        silhouettes.append(s)

    best_idx = int(np.argmax(silhouettes))
    optimal_k = k_range[best_idx]

    return optimal_k, inertias, silhouettes, k_range, silhouettes[best_idx]


def plot_find_optimal_clusters_root(optimal_k, inertias, silhouettes, k_range):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(k_range, inertias, "o-")
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Inertia (WCSS)")
    ax1.set_title("Elbow (L2-normalized)")

    ax2.plot(k_range, silhouettes, "o-")
    ax2.set_xlabel("Number of Clusters (K)")
    ax2.set_ylabel("Silhouette (cosine)")
    ax2.set_title("Silhouette vs K")
    plt.show()
