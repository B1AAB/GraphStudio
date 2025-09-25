import torch
import torch.nn as nn
from torch_geometric.data import Data


# method for creating augmented views for contrastive learning.
#
# define a transformer that creates two augmented views of a graph
# the method is generic and works by randomly dropping edges. 
# also, note that most of the sampled graphs are DAG, so dropping 
# one edge, may make the entire subgraph disconnected. 
# this should be taken into account in the following method to avoid 
# turning a single-connected graph to two disconnected communities with 
# no edges between them.
# randomly dropping edges can be slow (maybe run on cpu then move to gpu?!)
# and another drawback is that it can severely alter the graphs structure
# so it needs a decent amount of careful considerations to avoid breaking
# the graph.
#
# the following method is slightly different, it does not drop edges,
# instead it alters blockheight (edge feature) and
# original in/out degrees (node features), so it can be vectorized and
# run in gpu during model training.
#
# ********************* important note!!
# this augmentation does not change node/edge order 

class ViewAugmenter(nn.Module):
    """
    Two on-GPU structure-preserving views:
        - BlockHeight monotone warp: h' = a*h + b  (a>0 keeps order)
        - Original degree jitter: multiplicative +/-percentage on selected columns (never zeroed)
        - Optional Value jitter (disabled by default)
            Assumes edge_attr = [Value, BlockHeight] and
            x = [..., OriginalIndegree, OriginalOutdegree] at the given cols.
    """
    def __init__(
        self,
        block_height_col: int = 1,
        block_height_scale_range=(0.99, 1.01),   # small scale
        block_height_shift_range=(-3.0, 3.0),    # small shift in block height
        degree_cols=(2, 3),              # indices of OriginalIn/OutDegree in x
        degree_jitter=0.10,              # +/-10% multiplicative jitter
        value_col: int | None = None,
        value_jitter=0.0,                # e.g., 0.05 for +/-5%
    ):
        super().__init__()
        self.block_height_col = block_height_col
        self.block_height_scale_range = block_height_scale_range
        self.block_height_shift_range = block_height_shift_range
        self.degree_cols = degree_cols
        self.degree_jitter = float(degree_jitter)
        self.value_col = value_col
        self.value_jitter = float(value_jitter)

    @torch.no_grad()
    def _one_view(self, data: Data) -> Data:
        out = Data()
        out.edge_index = data.edge_index
    
        # Augment edge attributes
        edge_attribute = data.edge_attr.clone()
        
        # BlockHeight monotone warp
        if self.block_height_col is not None:
            a = torch.empty((), device=edge_attribute.device).uniform_(*self.block_height_scale_range)
            b = torch.empty((), device=edge_attribute.device).uniform_(*self.block_height_shift_range)
            edge_attribute[:, self.block_height_col] = edge_attribute[:, self.block_height_col] * a + b
        
        # Value jitter
        if self.value_col is not None:
            btc_value = edge_attribute[:, self.value_col]
            noise = 1.0 + (2*torch.rand_like(btc_value) - 1.0) * self.value_jitter
            edge_attribute[:, self.value_col] = (btc_value * noise).clamp_min(0)
        
        out.edge_attr = edge_attribute

        # Augment node attributes
        x = data.x.clone()
        if self.degree_cols is not None and len(self.degree_cols) > 0:
            cols = torch.tensor(self.degree_cols, device=x.device)
            # multiplicative ±pct jitter (never set to zero)
            noise = 1.0 + (2*torch.rand(x.size(0), len(cols), device=x.device) - 1.0) * self.degree_jitter
            x[:, cols] = (x[:, cols] * noise).clamp_min(0)
        out.x = x
        
        # carry-over misc fields you need
        for attr in ("y", "batch", "seed_root", "graph_id"):
            if hasattr(data, attr):
                setattr(out, attr, getattr(data, attr))
        out.num_nodes = data.num_nodes
        return out

    def forward(self, data: Data):
        return self._one_view(data), self._one_view(data)