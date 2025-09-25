# TODO: These stats are not representative enough, 
# because it is generic and does not include
# bitcoin specific features.


import torch, numpy as np, networkx as nx
from collections import Counter, defaultdict


def get_graph_stats(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    *,
    root_node: int,
    value_col: int = 0,
    block_height_col: int = 1) -> dict:
    
    graph = nx.DiGraph()
    graph.add_edges_from(zip(
        edge_index[0].tolist(),  # source scripts
        edge_index[1].tolist()))  # target scripts

    # in this dict, the key is node id, and value is its distance from the root
    node_distance_from_root_dict = nx.single_source_shortest_path_length(
        graph, root_node)
    
    hop_counts = Counter(d for d in node_distance_from_root_dict.values() if d > 0)
    # the key is hop level, and value is its frequency, 
    # so basically is the count (value) of nodes at n-hop (key) distance from root
    hop_counts_dict = {int(k): int(v) for k, v in hop_counts.items()}
    
    
    edge_hops = defaultdict(lambda: defaultdict(list))
    for i in range(edge_index.size(1)):
        u = int(edge_index[0, i])
        v = int(edge_index[1, i])
        du = node_distance_from_root_dict.get(u)
        dv = node_distance_from_root_dict.get(v)
        
        key = f"hop{du} -> hop{dv}"
        edge_hops[key]["Value"].append(float(edge_attr[i, value_col]))
        edge_hops[key]["BlockHeight"].append(float(edge_attr[i, block_height_col]))

    edge_hop_stats = {}
    for key, gathered_values in sorted(edge_hops.items()):
        stats = {}
        for name, vals in gathered_values.items():
            vals = np.asarray(vals, dtype=float)
            stats[name] = {
                "min": vals.min(),
                "max": vals.max(),
                "mean": vals.mean(),
                "count": vals.size,
            }
            
        edge_hop_stats[key] = stats

    return {
        "root_node_index": int(root_node),
        "hop_counts": hop_counts_dict,
        "edge_hop_stats": edge_hop_stats,
    }
