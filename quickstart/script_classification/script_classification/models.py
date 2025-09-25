import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import AttentionalAggregation
from torch_geometric.nn import GINEConv, APPNP


class GraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim=2, appnp_K=16, appnp_alpha=0.1):
        super().__init__()
        self.out_dim = out_channels

        mlp1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.conv1 = GINEConv(mlp1, edge_dim=edge_dim)
        self.norm1 = nn.LayerNorm(hidden_channels)

        mlp2 = nn.Sequential(
            nn.Linear(hidden_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )
        self.conv2 = GINEConv(mlp2, edge_dim=edge_dim)
        self.norm2 = nn.LayerNorm(out_channels)

        self.appnp = APPNP(K=appnp_K, alpha=appnp_alpha, dropout=0.0)

        self.dropout = nn.Dropout(0.1)
        gate_nn = nn.Sequential(
            nn.Linear(out_channels, out_channels // 2),
            nn.GELU(),
            nn.Linear(out_channels // 2, 1)
        )
        self.pool = AttentionalAggregation(gate_nn=gate_nn)

    def forward(self, x, edge_index, batch, edge_attr):
        h = self.conv1(x, edge_index, edge_attr)
        h = self.norm1(F.gelu(h))
        h = self.dropout(h)

        h = self.conv2(h, edge_index, edge_attr)
        h = self.norm2(h)
        h = self.appnp(h, edge_index)

        g = self.pool(h, batch)
        return h, g