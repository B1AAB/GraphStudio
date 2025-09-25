import numpy as np
import torch
import torch.nn.functional as F
from script_classification.utilities.graph_ops import reverse_edges


@torch.no_grad()
def evaluate_root_knn_consistency(encoder, loader, device, augmenter, k=10):
    encoder.eval()
    Z1, Z2 = [], []
    for batch in loader:
        batch = batch.to(device)
        v1, v2 = augmenter(batch)

        ei1 = reverse_edges(v1.edge_index)
        n1, _ = encoder(v1.x, ei1, v1.batch, edge_attr=v1.edge_attr)

        ei2 = reverse_edges(v2.edge_index)
        n2, _ = encoder(v2.x, ei2, v2.batch, edge_attr=v2.edge_attr)

        idx = batch.ptr[:-1] + batch.seed_root.view(-1).long()
        Z1.append(F.normalize(n1[idx], dim=1).cpu())
        Z2.append(F.normalize(n2[idx], dim=1).cpu())

    Z1 = torch.cat(Z1, 0).numpy()
    Z2 = torch.cat(Z2, 0).numpy()

    S1 = Z1 @ Z1.T  # cosine on unit vectors
    S2 = Z2 @ Z2.T
    np.fill_diagonal(S1, -np.inf)
    np.fill_diagonal(S2, -np.inf)
    N = Z1.shape[0]; k = min(k, max(1, N-1))
    nn1 = np.argpartition(-S1, kth=k, axis=1)[:, :k]
    nn2 = np.argpartition(-S2, kth=k, axis=1)[:, :k]
    overlap = np.array([len(set(nn1[i]).intersection(nn2[i]))/k for i in range(N)], dtype=float)
    return float(overlap.mean())
