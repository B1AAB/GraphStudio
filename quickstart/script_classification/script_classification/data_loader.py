import os
import os.path as osp

import numpy as np
import pandas as pd
import torch
from torch.serialization import add_safe_globals
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import Data as PyGData, DataEdgeAttr
from torch_geometric.data.storage import GlobalStorage
from tqdm.notebook import tqdm


class BitcoinScriptsDataset(InMemoryDataset):
    NODE_FEATURE_NAMES = [
        "InDegree", "OutDegree", "OriginalInDegree", "OriginalOutDegree",
        "OutHopsFromRoot", 
        "IncomingValueSumNormalized", "IncomingValueMeanNormalized", "IncomingBlockHeightSpanNormalized",
        "OutgoingValueSumNormalized", "OutgoingValueMeanNormalized", "OutgoingBlockHeightSpanNormalized",
        "DegreeRatioNormalized"
    ]
    
    EDGE_FEATURE_NAMES = ["Value", "BlockHeight"]
    
    def __init__(self, root, transform=None, pre_transform=None, min_nodes_per_graph=500, max_nodes_per_graph=5000):
        self.min_nodes_per_graph = min_nodes_per_graph
        self.max_nodes_per_graph = max_nodes_per_graph
        
        super().__init__(root, transform, pre_transform)
        processed_paths = self.processed_paths[0]
        # white-list PyG classes for safe unpickling
        add_safe_globals([PyGData, DataEdgeAttr, GlobalStorage])
        
        self.data, self.slices, metadata = torch.load(processed_paths, map_location="cpu", weights_only=False)

        self.per_graph_node_count = metadata.get("per_graph_node_count", [])
        self.per_graph_edge_count = metadata.get("per_graph_edge_count", [])
        self.per_graph_stats = metadata.get("per_graph_stats", {})
        self.node_feature_names = metadata.get("node_feature_names", [])
        self.edge_feature_names = metadata.get("edge_feature_names", [])

    @property
    def raw_file_names(self):
        return [d for d in os.listdir(self.raw_dir) if osp.isdir(osp.join(self.raw_dir, d))]

    @property
    def processed_file_names(self):
        return "bitcoin_script_to_script_communities.pt"

    def download(self):
        pass
    
    def _get_filenames(self, graph_id):
            return \
                osp.join(self.raw_dir, str(graph_id), "BitcoinScriptNode.tsv"), \
                osp.join(self.raw_dir, str(graph_id), "BitcoinS2S.tsv"), \
                osp.join(self.raw_dir, str(graph_id), "metadata.tsv")

    def _get_root_node_index(self, labels_filename):
        labels_df = pd.read_csv(labels_filename, sep="\t")
        return int(labels_df["RootNodeIdx"].values[0])
    
    def _get_nodes(self, filename):
        return pd.read_csv(filename, sep="\t")
    
    def _get_normalized_nodes_features(self, node_df):        
        # hops_min = node_df["OutHopsFromRoot"].min()
        # hops_max = node_df["OutHopsFromRoot"].max()
        # node_df["OutHopsFromRootNormalized"] = (node_df["OutHopsFromRoot"] - hops_min) / max(hops_max - hops_min, 1e-8)

        # o_in_degree =np.log1p(node_df["OriginalInDegree"])
        # o_in_degree_min = o_in_degree.min()
        # o_in_degree_max = o_in_degree.max()
        # node_df["OriginalInDegreeNormalized"] = (o_in_degree - o_in_degree_min) / max(o_in_degree_max - o_in_degree_min, 1e-8)

        # o_out_degree =np.log1p(node_df["OriginalOutDegree"])
        # o_out_degree_min = o_out_degree.min()
        # o_out_degree_max = o_out_degree.max()
        # node_df["OriginalOutDegreeNormalized"] = (o_out_degree - o_out_degree_min) / max(o_out_degree_max - o_out_degree_min, 1e-8)

        return node_df

    def _get_edges(self, filename):
        return pd.read_csv(filename, sep="\t")

    def _get_min_max_normalized(self, values):
        feature_min, feature_max = float(values.min()), float(values.max())
        values_range = max(feature_max - feature_min, 1e-8)
        return (values - feature_min) / values_range

    def _get_normalized_edge_features(self, edge_df):
        source_node_index = edge_df["Source"].to_numpy(dtype=np.int64)
        target_node_index = edge_df["Target"].to_numpy(dtype=np.int64)

        values = edge_df["Value"].to_numpy(dtype=np.float32)
        # values_scaled = np.log1p(values)
        # values_normalized = self._get_min_max_normalized(values_scaled)
        values_normalized = values

        heights = edge_df["BlockHeight"].to_numpy(dtype=np.float32)
        # heights_normalized = self._get_min_max_normalized(heights)
        heights_normalized = heights

        edge_attr = torch.from_numpy(np.column_stack([values_normalized, heights_normalized]).astype(np.float32))
        edge_index = torch.stack([torch.from_numpy(source_node_index).long(), torch.from_numpy(target_node_index).long()], dim=0)   
        
        edge_df_norm = pd.DataFrame({
            "Source": source_node_index,
            "Target": target_node_index,
            "Value": values,
            "BlockHeight": heights,
            "ValueNormalized": values_normalized,
            "HeightNormalized": heights_normalized
        })

        return edge_df_norm, edge_attr, edge_index

    def _fill_from_group(self, df: pd.DataFrame, col: str, nodes_count: int, default: float = 0.0) -> np.ndarray:
        """_summary_
        This method achieves the following: 
        
        nodes_count = 5 (nodes 0..4)
        df (groupby on incoming):
        index  val_sum
        1      3.5
        4      1.2

        _fill_from_group(df, "val_sum", 5, 0) -> [0.0, 3.5, 0.0, 0.0, 1.2]
        """
        return (
            df[col]
            .reindex(range(nodes_count), fill_value=default)
            .to_numpy(dtype=np.float32)
        )

    def _get_neighborhood_features(self, edge_df_normalized, nodes_count):
        incoming_edges = edge_df_normalized.groupby("Target").agg(
            val_sum=("Value", "sum"),
            val_mean=("Value", "mean"),
            h_min=("BlockHeight", "min"),
            h_max=("BlockHeight", "max"),
            deg=("Target", "count"),
        )
        outgoing_edges = edge_df_normalized.groupby("Source").agg(
            val_sum=("Value", "sum"),
            val_mean=("Value", "mean"),
            h_min=("BlockHeight", "min"),
            h_max=("BlockHeight", "max"),
            deg=("Source", "count"),
        )
        
        inc_val_sum  = self._fill_from_group(incoming_edges, "val_sum", nodes_count)
        inc_val_mean = self._fill_from_group(incoming_edges, "val_mean", nodes_count)
        inc_h_min    = self._fill_from_group(incoming_edges, "h_min", nodes_count)
        inc_h_max    = self._fill_from_group(incoming_edges, "h_max", nodes_count)
        inc_deg      = self._fill_from_group(incoming_edges, "deg", nodes_count)
        inc_span     = np.maximum(inc_h_max - inc_h_min, 0.0)

        out_val_sum  = self._fill_from_group(outgoing_edges, "val_sum", nodes_count)
        out_val_mean = self._fill_from_group(outgoing_edges, "val_mean", nodes_count)
        out_h_min    = self._fill_from_group(outgoing_edges, "h_min", nodes_count)
        out_h_max    = self._fill_from_group(outgoing_edges, "h_max", nodes_count)
        out_deg      = self._fill_from_group(outgoing_edges, "deg", nodes_count)
        out_span     = np.maximum(out_h_max - out_h_min, 0.0)

        deg_ratio    = out_deg / (inc_deg + 1e-6)

        x_extra_np = np.stack([
            self._get_min_max_normalized(np.log1p(inc_val_sum)), 
            self._get_min_max_normalized(np.log1p(inc_val_mean)), 
            self._get_min_max_normalized(np.log1p(inc_span)),
            self._get_min_max_normalized(np.log1p(out_val_sum)), 
            self._get_min_max_normalized(np.log1p(out_val_mean)), 
            self._get_min_max_normalized(np.log1p(out_span)),
            self._get_min_max_normalized(np.log1p(deg_ratio))
        ], axis=1).astype(np.float32)
        
        return torch.from_numpy(x_extra_np)

    def _get_hop_level_stats(self, nodes_df: pd.DataFrame, edge_df: pd.DataFrame) -> dict:
        """
        returns min/max/avg of edge features, grouped by the hop level of the source node.
        """
        node_hops = nodes_df[["OriginalInDegree", "OriginalOutDegree", "OutHopsFromRoot"]].copy()
        merged_df = pd.merge(
            edge_df, 
            node_hops, 
            left_on="Source", 
            right_index=True, 
            how="inner"
        )

        stats_df = merged_df.groupby("OutHopsFromRoot").agg(
            Value_min=("Value", "min"),
            Value_max=("Value", "max"),
            Value_avg=("Value", "mean"),
            BlockHeight_min=("BlockHeight", "min"),
            BlockHeight_max=("BlockHeight", "max"),
            BlockHeight_avg=("BlockHeight", "mean"),
            OriginalInDegree_min=("OriginalInDegree", "min"),
            OriginalInDegree_max=("OriginalInDegree", "max"),
            OriginalInDegree_avg=("OriginalInDegree", "mean"),
            OriginalOutDegree_min=("OriginalOutDegree", "min"),
            OriginalOutDegree_max=("OriginalOutDegree", "max"),
            OriginalOutDegree_avg=("OriginalOutDegree", "mean"),
        )
        
        # for easier serialization
        # {hop_level: {'Value_min': ..., 'Value_max': ...}}
        return stats_df.to_dict(orient="index")

    def process(self):
        data_list = []
        per_graph_node_count = []
        per_graph_edge_count = []
        per_graph_stats = {}

        graph_dirs = sorted(d for d in os.listdir(self.raw_dir) if osp.isdir(osp.join(self.raw_dir, d)))
        for graph_id in tqdm(graph_dirs, desc="Processing Graphs"):
            node_path, edge_path, labels_path = self._get_filenames(graph_id)
            if any([not osp.exists(node_path), not osp.exists(edge_path), not osp.exists(labels_path)]):
                continue

            nodes_df = self._get_nodes(node_path)
            nodes_count = len(nodes_df)
            if nodes_count == 0 or nodes_count > self.max_nodes_per_graph or nodes_count < self.min_nodes_per_graph:
                continue
            # nodes_df = self._get_normalized_nodes_features(nodes_df)
            nodes_base_features_tensor = torch.tensor(
                nodes_df[[
                    "InDegree", 
                    "OutDegree", 
                    "OriginalInDegree", 
                    "OriginalOutDegree", 
                    "OutHopsFromRoot"]].values,
                dtype=torch.float
            )

            edge_df = self._get_edges(edge_path)
            edge_df_norm, edge_attr, edge_index = self._get_normalized_edge_features(edge_df)
            
            nodes_neighborhood_features_tensor = self._get_neighborhood_features(edge_df_norm, nodes_count)

            x = torch.cat([nodes_base_features_tensor, nodes_neighborhood_features_tensor], dim=1)

            stats = self._get_hop_level_stats(nodes_df, edge_df)
            per_graph_stats[graph_id] = stats

            data = Data(
                graph_id=graph_id,
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                seed_root=torch.tensor([self._get_root_node_index(labels_path)], dtype=torch.long)
            )
            data_list.append(data)

            per_graph_node_count.append(nodes_count)
            per_graph_edge_count.append(edge_index.size(1))
            # per_graph_stats.append(get_graph_stats(edge_index, edge_attr))

        if len(data_list) == 0:
            raise RuntimeError(f"No graphs were processed from {self.raw_dir}")

        data, slices = self.collate(data_list)
        metadata = {
            "per_graph_node_count": per_graph_node_count,
            "per_graph_edge_count": per_graph_edge_count,
            "per_graph_stats": per_graph_stats,
            "node_feature_names": self.NODE_FEATURE_NAMES,
            "edge_feature_names": self.EDGE_FEATURE_NAMES
        }
        torch.save((data, slices, metadata), self.processed_paths[0])
        print(f"Successfully processed and saved {len(data_list)} graphs.")
