from typing import Union
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from tgnnexplainer import ROOT_DIR

from tgnnexplainer.xgraph.models.ext.tgat.graph import NeighborFinder
from tgnnexplainer.xgraph.dataset.tg_dataset import verify_dataframe_unify

class MarginalSubgraphDataset(Dataset):
    """ Collect pair-wise graph data to calculate marginal contribution. """
    def __init__(self, data, exclude_mask, include_mask, subgraph_build_func) -> object:
        self.num_nodes = data.num_nodes
        self.X = data.x
        self.edge_index = data.edge_index
        self.device = self.X.device

        self.label = data.y
        self.exclude_mask = torch.tensor(exclude_mask).type(torch.float32).to(self.device)
        self.include_mask = torch.tensor(include_mask).type(torch.float32).to(self.device)
        self.subgraph_build_func = subgraph_build_func

    def __len__(self):
        return self.exclude_mask.shape[0]

    def __getitem__(self, idx):
        exclude_graph_X, exclude_graph_edge_index = self.subgraph_build_func(self.X, self.edge_index, self.exclude_mask[idx])
        include_graph_X, include_graph_edge_index = self.subgraph_build_func(self.X, self.edge_index, self.include_mask[idx])
        exclude_data = Data(x=exclude_graph_X, edge_index=exclude_graph_edge_index)
        include_data = Data(x=include_graph_X, edge_index=include_graph_edge_index)
        return exclude_data, include_data


def k_hop_temporal_subgraph(df, num_hops, event_idx):
    """
    df: temporal graph, events stream. DataFrame. An user-item bipartite graph.
    node: center user node
    num_hops: number of hops of the subgraph
    event_idx: should start from 1. 1, 2, 3, ...
    return: a sub DataFrame

    """
    # verify_dataframe(df)
    verify_dataframe_unify(df)

    df_new = df.copy()
    df_new['u'] -= 1
    df_new['i'] -= 1
    df_new = df_new[df_new.e_idx <= event_idx] # ignore events latter than event_idx

    # center_node = df_new.iloc[event_idx-1, 0]
    center_node = df_new[df_new.e_idx == event_idx].u.values[0] # event_idx represents e_idx

    subsets = [[center_node], ]
    num_nodes = df_new.i.max() + 1

    # import ipdb; ipdb.set_trace()
    node_mask = np.zeros((num_nodes,), dtype=bool)
    source_nodes = np.array(df_new.iloc[:, 0], dtype=int) # user nodes, 0--k-1
    target_nodes = np.array(df_new.iloc[:, 1], dtype=int) # item nodes, k--N-1, N is the number of total users and items

    for _ in range(num_hops):
        node_mask.fill(False)
        node_mask[ np.array(subsets[-1]) ] = True
        edge_mask = node_mask[source_nodes]
        new_nodes = target_nodes[edge_mask] # new neighbors
        subsets.append(np.unique(new_nodes).tolist())

        source_nodes, target_nodes = target_nodes, source_nodes # regarded as undirected graph

    # import ipdb; ipdb.set_trace()
    subset = np.unique(np.concatenate([np.array(nodes) for nodes in subsets])) # selected temporal subgraph nodes
    
    assert center_node in subset

    source_nodes = np.array(df_new.iloc[:, 0], dtype=int)
    target_nodes = np.array(df_new.iloc[:, 1], dtype=int)

    node_mask.fill(False)
    node_mask[ subset ] = True

    user_mask = node_mask[source_nodes] # user mask for events
    item_mask = node_mask[target_nodes] # item mask for events

    # import ipdb; ipdb.set_trace()
    edge_mask = user_mask & item_mask # event mask
    # import ipdb; ipdb.set_trace()

    subgraph_df = df_new.iloc[edge_mask, :].copy()
    # subgraph_df.iloc[:, 1] -= base # recover user item naming indices
    # import ipdb; ipdb.set_trace()
    assert center_node in subgraph_df.iloc[:, 0].values

    subgraph_df['u'] += 1
    subgraph_df['i'] += 1

    return subgraph_df

# def tgat_node_reindex(u: Union[int, np.array], i: Union[int, np.array], num_users: int):
#     u = u + 1
#     i = i + 1 + num_users
#     return u, i

def construct_tgat_neighbor_finder(df):
    verify_dataframe_unify(df)

    num_nodes = df['i'].max()
    adj_list = [[] for _ in range(num_nodes + 1)]
    for i in range(len(df)):
        user, item, time, e_idx = df.u[i], df.i[i], df.ts[i], df.e_idx[i]
        adj_list[user].append((item, e_idx, time))
        adj_list[item].append((user, e_idx, time))
    neighbor_finder = NeighborFinder(adj_list, uniform=False) # default 'uniform' is False

    return neighbor_finder
