from typing import Union
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from dig import ROOT_DIR

from dig.xgraph.models.ext.tgat.graph import NeighborFinder
from dig.xgraph.dataset.tg_dataset import verify_dataframe

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


def k_hop_temporal_subgraph(t_graph, num_hops, event_idx):
    """
    t_graph: temporal graph, events stream. DataFrame. An user-item bipartite graph.
    node: center user node
    num_hops: number of hops of the subgraph

    return: a sub DataFrame

    """
    verify_dataframe(t_graph)

    center_node = t_graph.iloc[event_idx, 0]

    t_graph_new = t_graph.copy()
    base = t_graph.iloc[:, 0].max() + 1
    t_graph_new.iloc[:, 1] += base # unifying user item naming indices
    t_graph_new = t_graph_new.iloc[:event_idx+1, :] # ignore events latter than event_idx
    
    subsets = [[center_node], ]

    num_nodes = t_graph_new.iloc[:, 1].max() + 1

    # import ipdb; ipdb.set_trace()

    node_mask = np.zeros((num_nodes,), dtype=bool)
    source_nodes = np.array(t_graph_new.iloc[:, 0], dtype=int) # user nodes, 0--k-1
    target_nodes = np.array(t_graph_new.iloc[:, 1], dtype=int) # item nodes, k--N-1, N is the number of total users and items

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

    source_nodes = np.array(t_graph_new.iloc[:, 0], dtype=int)
    target_nodes = np.array(t_graph_new.iloc[:, 1], dtype=int)

    node_mask.fill(False)
    node_mask[ subset ] = True

    user_mask = node_mask[source_nodes] # user mask for events
    item_mask = node_mask[target_nodes] # item mask for events

    # import ipdb; ipdb.set_trace()
    edge_mask = user_mask & item_mask # event mask
    # import ipdb; ipdb.set_trace()

    subgraph_df = t_graph_new.iloc[edge_mask, :].copy()
    subgraph_df.iloc[:, 1] -= base # recover user item naming indices
    # import ipdb; ipdb.set_trace()
    assert center_node in subgraph_df.iloc[:, 0].values
    return subgraph_df

def tgat_node_reindex(u: Union[int, np.array], i: Union[int, np.array], num_users: int):
    u = u + 1
    i = i + 1 + num_users

    return u, i


def construct_tgat_model_data(df, dataset_name):
    verify_dataframe(df)

    num_users = df['u'].max() + 1
    num_nodes = df['u'].max() + 1 + df['i'].max() + 1

    adj_list = [[] for _ in range(num_nodes + 1)]
    for e_idx in df.index.values:
        user, item, time = df.iloc[e_idx, 0], df.iloc[e_idx, 1], df.iloc[e_idx, 2]
        user, item = tgat_node_reindex(user, item, num_users) # reindex user and item name
        e_idx = e_idx + 1 # NOTE: +1 for edge features, and edge indexs
        adj_list[user].append((item, e_idx, time))
        adj_list[item].append((user, e_idx, time))

    neighbor_finder = NeighborFinder(adj_list, uniform=False) # default 'uniform' is False
    edge_feats = np.load(ROOT_DIR/'xgraph'/'models'/'ext'/'tgat'/'processed'/f'ml_{dataset_name}.npy')
    node_feats = np.load(ROOT_DIR/'xgraph'/'models'/'ext'/'tgat'/'processed'/f'ml_{dataset_name}_node.npy')
    

    return neighbor_finder, node_feats, edge_feats



def construct_tgat_neighbor_finder(subgraph, n_users, uniform=True):
    """
    Construct neighbor finder, node feature array, and edge feature array

    Input: ? a subgraph dataframe, u, i, time, time_idx

    we should only:
    let user_idx += 1
    let item_idx += user_idx.max() + 1

    SHOULD NOT CHANGE THE 'subgraph' PARAMETER.

    """
    
    subgraph_new = subgraph.copy()
    subgraph_new.iloc[:, 0] += 1
    subgraph_new.iloc[:, 1] += (n_users + 1)

    old_new_mapping = {'user': {}, 'item': {}}
    for i in range(len(subgraph)):
        old_u = subgraph.iloc[i, 0]
        new_u = subgraph_new.iloc[i, 0]
        old_i = subgraph.iloc[i, 1]
        new_i = subgraph_new.iloc[i, 1]

        old_new_mapping['user'][old_u] = new_u
        old_new_mapping['item'][old_i] = new_i

    # new_target_u = target_u + 1
    # new_target_i = target_i + (n_users + 1)
    
    # import ipdb; ipdb.set_trace()

    adj_list = [[] for _ in range(subgraph_new.iloc[:, 1].max() + 1)]
    for e_idx in range(len(subgraph_new)):
        user, item, time = subgraph_new.iloc[e_idx, 0], subgraph_new.iloc[e_idx, 1], subgraph_new.iloc[e_idx, 2]
        adj_list[user].append((item, e_idx+1, time))
        adj_list[item].append((user, e_idx+1, time))


    neighbor_finder = NeighborFinder(adj_list, uniform=uniform)
    return neighbor_finder, old_new_mapping