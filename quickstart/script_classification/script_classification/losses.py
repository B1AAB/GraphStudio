import torch
import torch.nn.functional as F


def info_nce_global(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1, symmetric: bool = True):
    """
    Global InfoNCE over all nodes in the batch.
    Assumes node i in z1 corresponds to node i in z2 (no reindexing between views).
    Uses all other nodes as negatives (across all graphs).
    """
    assert z1.shape == z2.shape, "z1 and z2 must have same shape [N, D]"
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    logits = (z1 @ z2.T) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)

    loss12 = F.cross_entropy(logits, labels)
    if not symmetric:
        return loss12

    # z2 -> z1 direction
    logits_t = (z2 @ z1.T) / temperature
    loss21 = F.cross_entropy(logits_t, labels)
    return 0.5 * (loss12 + loss21)

def info_nce_root_only(g1: torch.Tensor, g2: torch.Tensor, T: float = 0.2, symmetric: bool = True):
    g1 = F.normalize(g1, dim=1)
    g2 = F.normalize(g2, dim=1)
    logits = (g1 @ g2.T) / T  # [G, G]
    labels = torch.arange(g1.size(0), device=g1.device)
    loss12 = F.cross_entropy(logits, labels)
    if not symmetric:
        return loss12
    loss21 = F.cross_entropy((g2 @ g1.T) / T, labels)
    return 0.5 * (loss12 + loss21)
