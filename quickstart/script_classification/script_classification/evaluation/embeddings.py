import torch
import torch.nn.functional as F
from script_classification.utilities.graph_ops import reverse_edges


@torch.no_grad()
def embed_roots(encoder, loader, device):
    encoder.eval()
    chunks = []
    for batch in loader:
        batch = batch.to(device)
        edges = reverse_edges(batch.edge_index)
        node_z, _ = encoder(batch.x, edges, batch.batch, edge_attr=batch.edge_attr)
        idx = batch.ptr[:-1] + batch.seed_root.view(-1).long()
        chunks.append(F.normalize(node_z[idx], dim=1).cpu())
    return torch.cat(chunks, dim=0).numpy()


@torch.no_grad()
def embed_roots_proj(encoder, proj_head, loader, device):
    encoder.eval()
    zs = []
    for batch in loader:
        batch = batch.to(device)
        edges = reverse_edges(batch.edge_index)
        node_z, _ = encoder(batch.x, edges, batch.batch, edge_attr=batch.edge_attr)
        idx = batch.ptr[:-1] + batch.seed_root.view(-1).long()
        zs.append(F.normalize(proj_head(node_z[idx]), dim=1).cpu())
    return torch.cat(zs, dim=0).numpy()
