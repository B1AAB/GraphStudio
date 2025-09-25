import torch
import torch.nn.functional as F
from script_classification.utilities.graph_ops import reverse_edges


def train_one_epoch(
    encoder,
    loader,
    optimizer,
    device,
    epoch,
    writer,
    augmenter,
    proj_head,
    temp_root=0.2
):
    encoder.train()

    total = 0.0
    tot_root = 0.0

    for step, batch in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)
        batch = batch.to(device, non_blocking=True)
        idx = batch.ptr[:-1] + batch.seed_root.view(-1).long()
        
        v1, v2 = augmenter(batch)

        # ---- View 1 ----
        ei1 = reverse_edges(v1.edge_index)
        n1, _ = encoder(v1.x, ei1, v1.batch, edge_attr=v1.edge_attr)        
        r1 = n1[idx]                            # [G, D_enc]
        p1 = F.normalize(proj_head(r1), dim=1)  # [G, D_proj]
        del v1, n1

        # ---- View 2 ----
        ei2 = reverse_edges(v2.edge_index)
        n2, _ = encoder(v2.x, ei2, v2.batch, edge_attr=v2.edge_attr)
        r2 = n2[idx]                            # [G, D_enc]
        p2 = F.normalize(proj_head(r2), dim=1)  # [G, D_proj]
        
        # symmetric InfoNCE        
        # anchors=p1, positives=p2
        logits = (p1 @ p2.T) / temp_root
        labels = torch.arange(p1.size(0), device=p1.device)
        loss12 = F.cross_entropy(logits, labels)        
        # swap roles
        # anchors=p2, positives=p1
        logits_t = (p2 @ p1.T) / temp_root
        loss21 = F.cross_entropy(logits_t, labels)
        loss_root = 0.5 * (loss12 + loss21)

        loss = loss_root

        loss.backward()
        optimizer.step()

        total += loss.item()
        tot_root += loss_root.item()
        if writer is not None:
            gs = epoch * len(loader) + step
            writer.add_scalar("train/step_total", float(loss.item()), gs)

        # free tensors early
        del v2, n2, r1, r2, p1, p2

    nb = max(1, len(loader))
    return {"loss": total / nb, "root_loss": tot_root / nb}