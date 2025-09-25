import torch

def reverse_edges(ei):
    return torch.stack((ei[1], ei[0]), dim=0)