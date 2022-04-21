import os
import copy
import math
import time
import itertools
import torch
import numpy as np
import networkx as nx
from rdkit import Chem
import pandas as pd
from pandas import DataFrame
from typing import Union
from torch import Tensor
from textwrap import wrap
from functools import partial
from typing import List, Tuple, Dict
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_networkx
from typing import Callable, Union, Optional
import matplotlib.pyplot as plt
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.datasets import MoleculeNet
from torch_geometric.utils import remove_self_loops
from .shapley import GnnNetsGC2valueFunc, GnnNetsNC2valueFunc, \
    gnn_score, mc_shapley, l_shapley, mc_l_shapley, NC_mc_l_shapley


from dig import ROOT_DIR
from dig.xgraph.method.tg_score import TGNNRewardWraper
from dig.xgraph.dataset.utils_dataset import k_hop_temporal_subgraph


def to_networkx_tg(events: DataFrame):
    base = events.iloc[:, 0].max() + 1
    g = nx.MultiGraph()
    g.add_nodes_from( events.iloc[:, 0] )
    g.add_nodes_from( events.iloc[:, 1] + base )
    t_edges = []
    for i in range(len(events)):
        user, item, t, e_idx = events.iloc[i, 0], events.iloc[i, 1], events.iloc[i, 2], events.index[i]
        t_edges.append((user, item, {'t': t, 'e_idx': i},))
    g.add_edges_from(t_edges)

    return g

def networkx_to_pd(tg: nx.MultiGraph):
    pass


def find_closest_node_result(results):
    """ return the highest reward tree_node with its subgraph is smaller than max_nodes """
    results = sorted(results, key=lambda x: len(x.coalition))

    result_node = results[0]
    for result_idx in range(len(results)):
        x = results[result_idx]
        # if len(x.coalition) <= max_nodes and x.P > result_node.P:
        if x.P > result_node.P:
            result_node = x
    return result_node


def reward_func(reward_method, value_func, node_idx=None,
                local_radius=4, sample_num=100,
                subgraph_building_method='zero_filling'):
    if reward_method.lower() == 'gnn_score':
        return partial(gnn_score,
                       value_func=value_func,
                       subgraph_building_method=subgraph_building_method)


    elif reward_method.lower() == 'mc_shapley':
        return partial(mc_shapley,
                       value_func=value_func,
                       subgraph_building_method=subgraph_building_method,
                       sample_num=sample_num)

    elif reward_method.lower() == 'l_shapley':
        return partial(l_shapley,
                       local_radius=local_radius,
                       value_func=value_func,
                       subgraph_building_method=subgraph_building_method)

    elif reward_method.lower() == 'mc_l_shapley':
        return partial(mc_l_shapley,
                       local_radius=local_radius,
                       value_func=value_func,
                       subgraph_building_method=subgraph_building_method,
                       sample_num=sample_num)

    elif reward_method.lower() == 'nc_mc_l_shapley':
        assert node_idx is not None, " Wrong node idx input "
        return partial(NC_mc_l_shapley,
                       node_idx=node_idx,
                       local_radius=local_radius,
                       value_func=value_func,
                       subgraph_building_method=subgraph_building_method,
                       sample_num=sample_num)

    else:
        raise NotImplementedError


def k_hop_subgraph_with_default_whole_graph(
        edge_index, node_idx=None, num_hops=3, relabel_nodes=False,
        num_nodes=None, flow='source_to_target'):
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.
    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
            node(s).
        num_hops: (int): The number of hops :math:`k`.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index  # edge_index 0 to 1, col: source, row: target

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    inv = None

    if node_idx is None:
        subsets = torch.tensor([0])
        cur_subsets = subsets
        while 1:
            node_mask.fill_(False)
            node_mask[subsets] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets = torch.cat([subsets, col[edge_mask]]).unique()
            if not cur_subsets.equal(subsets):
                cur_subsets = subsets
            else:
                subset = subsets
                break
    else:
        if isinstance(node_idx, (int, list, tuple)):
            node_idx = torch.tensor([node_idx], device=row.device, dtype=torch.int64).flatten()
        elif isinstance(node_idx, torch.Tensor) and len(node_idx.shape) == 0:
            node_idx = torch.tensor([node_idx])
        else:
            node_idx = node_idx.to(row.device)

        subsets = [node_idx]
        for _ in range(num_hops):
            node_mask.fill_(False)
            node_mask[subsets[-1]] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets.append(col[edge_mask])
        subset, inv = torch.cat(subsets).unique(return_inverse=True)
        inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes,), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask  # subset: key new node idx; value original node idx





class PlotUtils(object):
    def __init__(self, dataset_name, is_show=True):
        self.dataset_name = dataset_name
        self.is_show = is_show

    def plot(self, graph, nodelist, figname, title_sentence=None, **kwargs):
        """ plot function for different dataset """
        if self.dataset_name.lower() in ['ba_2motifs', 'ba_lrp']:
            self.plot_ba2motifs(graph, nodelist, title_sentence=title_sentence, figname=figname)
        elif self.dataset_name.lower() in ['mutag'] + list(MoleculeNet.names.keys()):
            x = kwargs.get('x')
            self.plot_molecule(graph, nodelist, x, title_sentence=title_sentence, figname=figname)
        elif self.dataset_name.lower() in ['ba_shapes', 'ba_community', 'tree_grid', 'tree_cycle']:
            y = kwargs.get('y')
            node_idx = kwargs.get('node_idx')
            self.plot_bashapes(graph, nodelist, y, node_idx, title_sentence=title_sentence, figname=figname)
        elif self.dataset_name.lower() in ['graph_sst2', 'graph_sst5', 'twitter']:
            words = kwargs.get('words')
            self.plot_sentence(graph, nodelist, words=words, title_sentence=title_sentence, figname=figname)
        else:
            raise NotImplementedError

    def plot_subgraph(self,
                      graph,
                      nodelist,
                      colors: Union[None, str, List[str]] = '#FFA500',
                      labels=None,
                      edge_color='gray',
                      edgelist=None,
                      subgraph_edge_color='black',
                      title_sentence=None,
                      figname=None):

        if edgelist is None:
            edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges()
                        if n_frm in nodelist and n_to in nodelist]
        pos = nx.kamada_kawai_layout(graph)
        pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}

        nx.draw_networkx_nodes(graph, pos,
                               nodelist=list(graph.nodes()),
                               node_color=colors,
                               node_size=300)

        nx.draw_networkx_edges(graph, pos, width=3, edge_color=edge_color, arrows=False)

        nx.draw_networkx_edges(graph, pos=pos_nodelist,
                               edgelist=edgelist, width=6,
                               edge_color=subgraph_edge_color,
                               arrows=False)

        if labels is not None:
            nx.draw_networkx_labels(graph, pos, labels)

        plt.axis('off')
        if title_sentence is not None:
            plt.title('\n'.join(wrap(title_sentence, width=60)))
        if figname is not None:
            plt.savefig(figname)
        if self.is_show:
            plt.show()
        if figname is not None:
            plt.close()

    def plot_subgraph_with_nodes(self,
                                 graph,
                                 nodelist,
                                 node_idx,
                                 colors='#FFA500',
                                 labels=None,
                                 edge_color='gray',
                                 edgelist=None,
                                 subgraph_edge_color='black',
                                 title_sentence=None,
                                 figname=None):
        node_idx = int(node_idx)
        if edgelist is None:
            edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges()
                        if n_frm in nodelist and n_to in nodelist]

        pos = nx.kamada_kawai_layout(graph)  # calculate according to graph.nodes()
        pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}

        nx.draw_networkx_nodes(graph, pos,
                               nodelist=list(graph.nodes()),
                               node_color=colors,
                               node_size=300)
        if isinstance(colors, list):
            list_indices = int(np.where(np.array(graph.nodes()) == node_idx)[0])
            node_idx_color = colors[list_indices]
        else:
            node_idx_color = colors

        nx.draw_networkx_nodes(graph, pos=pos,
                               nodelist=[node_idx],
                               node_color=node_idx_color,
                               node_size=600)

        nx.draw_networkx_edges(graph, pos, width=3, edge_color=edge_color, arrows=False)

        nx.draw_networkx_edges(graph, pos=pos_nodelist,
                               edgelist=edgelist, width=3,
                               edge_color=subgraph_edge_color,
                               arrows=False)

        if labels is not None:
            nx.draw_networkx_labels(graph, pos, labels)

        plt.axis('off')
        if title_sentence is not None:
            plt.title('\n'.join(wrap(title_sentence, width=60)))

        if figname is not None:
            plt.savefig(figname)
        if self.is_show:
            plt.show()
        if figname is not None:
            plt.close()

    def plot_sentence(self, graph, nodelist, words, edgelist=None, title_sentence=None, figname=None):
        pos = nx.kamada_kawai_layout(graph)
        words_dict = {i: words[i] for i in graph.nodes}
        if nodelist is not None:
            pos_coalition = {k: v for k, v in pos.items() if k in nodelist}
            nx.draw_networkx_nodes(graph, pos_coalition,
                                   nodelist=nodelist,
                                   node_color='yellow',
                                   node_shape='o',
                                   node_size=500)
            if edgelist is None:
                edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges()
                            if n_frm in nodelist and n_to in nodelist]
                nx.draw_networkx_edges(graph, pos=pos_coalition, edgelist=edgelist, width=5, edge_color='yellow')

        nx.draw_networkx_nodes(graph, pos, nodelist=list(graph.nodes()), node_size=300)

        nx.draw_networkx_edges(graph, pos, width=2, edge_color='grey')
        nx.draw_networkx_labels(graph, pos, words_dict)

        plt.axis('off')
        plt.title('\n'.join(wrap(' '.join(words), width=50)))
        if title_sentence is not None:
            string = '\n'.join(wrap(' '.join(words), width=50))
            string += '\n'.join(wrap(title_sentence, width=60))
            plt.title(string)
        if figname is not None:
            plt.savefig(figname)
        if self.is_show:
            plt.show()
        if figname is not None:
            plt.close()

    def plot_ba2motifs(self,
                       graph,
                       nodelist,
                       edgelist=None,
                       title_sentence=None,
                       figname=None):
        return self.plot_subgraph(graph, nodelist,
                                  edgelist=edgelist,
                                  title_sentence=title_sentence,
                                  figname=figname)

    def plot_molecule(self,
                      graph,
                      nodelist,
                      x,
                      edgelist=None,
                      title_sentence=None,
                      figname=None):
        # collect the text information and node color
        if self.dataset_name == 'mutag':
            node_dict = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}
            node_idxs = {k: int(v) for k, v in enumerate(np.where(x.cpu().numpy() == 1)[1])}
            node_labels = {k: node_dict[v] for k, v in node_idxs.items()}
            node_color = ['#E49D1C', '#4970C6', '#FF5357', '#29A329', 'brown', 'darkslategray', '#F0EA00']
            colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]

        elif self.dataset_name in MoleculeNet.names.keys():
            element_idxs = {k: int(v) for k, v in enumerate(x[:, 0])}
            node_idxs = element_idxs
            node_labels = {k: Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), int(v))
                           for k, v in element_idxs.items()}
            node_color = ['#29A329', 'lime', '#F0EA00',  'maroon', 'brown', '#E49D1C', '#4970C6', '#FF5357']
            colors = [node_color[(v - 1) % len(node_color)] for k, v in node_idxs.items()]
        else:
            raise NotImplementedError

        self.plot_subgraph(graph, nodelist,
                           colors=colors,
                           labels=node_labels,
                           edgelist=edgelist,
                           edge_color='gray',
                           subgraph_edge_color='black',
                           title_sentence=title_sentence,
                           figname=figname)

    def plot_bashapes(self,
                      graph,
                      nodelist,
                      y,
                      node_idx,
                      edgelist=None,
                      title_sentence=None,
                      figname=None):
        node_idxs = {k: int(v) for k, v in enumerate(y.reshape(-1).tolist())}
        node_color = ['#FFA500', '#4970C6', '#FE0000', 'green']
        colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]
        self.plot_subgraph_with_nodes(graph,
                                      nodelist,
                                      node_idx,
                                      colors,
                                      edgelist=edgelist,
                                      title_sentence=title_sentence,
                                      figname=figname,
                                      subgraph_edge_color='black')


class MCTSNode(object):
    def __init__(self, coalition: list = None, data: Data = None, ori_graph: nx.Graph = None,
                 c_puct: float = 10.0, W: float = 0, N: int = 0, P: float = 0,
                 load_dict: Optional[Dict] = None, device='cpu'):
        self.data = data # node data or edge data? or subgraph data? ignore now.
        self.coalition = coalition  # in our case, the coalition should be edge indices?
        self.ori_graph = ori_graph
        self.device = device
        self.c_puct = c_puct
        self.children = []
        self.W = W  # sum of node value
        self.N = N  # times of arrival
        self.P = P  # property score (reward)
        if load_dict is not None:
            self.load_info(load_dict)

    def Q(self):
        return self.W / self.N if self.N > 0 else 0

    def U(self, n):
        # return self.c_puct * self.P * math.sqrt(n) / (1 + self.N)
        return self.c_puct * math.sqrt(n) / (1 + self.N)

    @property
    def info(self):
        info_dict = {
            # 'data': self.data.to('cpu'),
            'coalition': self.coalition,
            # 'ori_graph': self.ori_graph,
            'W': self.W,
            'N': self.N,
            'P': self.P
        }
        return info_dict

    def load_info(self, info_dict):
        self.W = info_dict['W']
        self.N = info_dict['N']
        self.P = info_dict['P']
        self.coalition = info_dict['coalition']
        self.ori_graph = info_dict['ori_graph']
        self.data = info_dict['data'].to(self.device)
        self.children = []
        return self


# def preserved_candidates(coalition, source_idxs, candidates):
#     removed_idxs = obtain_removed_idxs(coalition, source_idxs)
#     preserved = [n for n in candidates if n not in removed_idxs]
#     return preserved

# def obtain_removed_idxs(coalition, source_idxs):
#     idx_set = set(coalition)
#     removed_idxs = list(filter(lambda x: x not in idx_set, source_idxs))
#     return removed_idxs


class MCTS(object):
    r"""
    Monte Carlo Tree Search Method.
    
    Args:
        X (:obj:`torch.Tensor`): Input node features
        edge_index (:obj:`torch.Tensor`): The edge indices.
        num_hops (:obj:`int`): The number of hops :math:`k`.
        n_rollout (:obj:`int`): The number of sequence to build the monte carlo tree.
        min_atoms (:obj:`int`): The number of atoms for the subgraph in the monte carlo tree leaf node. here is number of events preserved in the candidate events set.

        c_puct (:obj:`float`): The hyper-parameter to encourage exploration while searching.
        expand_atoms (:obj:`int`): The number of children to expand.
        high2low (:obj:`bool`): Whether to expand children tree node from high degree nodes to low degree nodes.
        node_idx (:obj:`int`): The target node index to extract the neighborhood.
        score_func (:obj:`Callable`): The reward function for tree node, such as mc_shapely and mc_l_shapely.
    """
    def __init__(self, events: DataFrame,
                 candidate_events = None,
                 num_users: int = -1,
                 n_rollout: int = 10, min_atoms: int = 5, c_puct: float = 10.0,
                 expand_atoms: int = 14,
                 node_idx: int = None, event_idx: int = None, score_func: Callable = None, device='cpu'):
        
        self.events = events # subgraph events or total events? subgraph events
        self.num_users = num_users
        self.graph = to_networkx_tg(events) # subgraph or total graph?
        # self.node_X = node_X # node features
        # self.event_X = event_X # event features
        self.node_idx = node_idx # node index to explain
        self.event_idx = event_idx # event index to explain

        # improve the strategy later
        # self.candidate_events = sorted(self.events.index.values.tolist())[-6:-1]
        # self.candidate_events = sorted(self.events.index.values.tolist())[-10:]
        # self.candidate_events = [10, 11, 12, 13, 14, 15, 19]
        self.candidate_events = candidate_events

        self.base_events_ = None

        # we only care these events, other events are preserved as is.
        # currently only take 10 temporal edges into consideration.
        
        self.device = device
        self.num_nodes = self.graph.number_of_nodes()


        self.score_func = score_func #! need to alter

        self.n_rollout = n_rollout
        self.min_atoms = min_atoms
        self.c_puct = c_puct
        self.expand_atoms = expand_atoms
        # self.high2low = high2low
        self.new_node_idx = None

        # extract the sub-graph and change the node indices.
        # if node_idx is not None:
        #     self.ori_node_idx = node_idx
        #     self.ori_graph = copy.copy(self.graph)
        #     x, edge_index, subset, edge_mask, kwargs = \
        #         self.__subgraph__(node_idx, self.X, self.edge_index, self.num_hops)
        #     self.data = Batch.from_data_list([Data(x=x, edge_index=edge_index)])
        #     self.graph = self.ori_graph.subgraph(subset.tolist())
        #     mapping = {int(v): k for k, v in enumerate(subset)}
        #     self.graph = nx.relabel_nodes(self.graph, mapping)
        #     self.new_node_idx = torch.where(subset == self.ori_node_idx)[0].item()
        #     self.num_nodes = self.graph.number_of_nodes()
        #     self.subset = subset
        self.data = None

        self.MCTSNodeClass = partial(MCTSNode, data=self.data, ori_graph=self.graph,
                                     c_puct=self.c_puct, device=self.device)

        # self.root_coalition = sorted([node for node in range(self.num_nodes)])
        self.initialize_tree()
    
    

    def set_score_func(self, score_func):
        self.score_func = score_func

    @staticmethod
    def __subgraph__(node_idx, x, edge_index, num_hops, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)
        subset, edge_index, _, edge_mask = k_hop_subgraph_with_default_whole_graph(
            edge_index, node_idx, num_hops, relabel_nodes=True, num_nodes=num_nodes)

        x = x[subset]
        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item

        return x, edge_index, subset, edge_mask, kwargs

    def compute_scores(self, score_func, children, target_event_idx):
        results = []
        for child in children:
            if child.P == 0:
                # score = score_func(child.coalition, child.data)
                score = score_func( self.base_events + child.coalition, target_event_idx)
            else:
                score = child.P
            results.append(score)
        return results

    def mcts_rollout(self, tree_node):
        """
        The tree_node now is a set of events
        """
        # import ipdb; ipdb.set_trace()
        if len( tree_node.coalition ) < self.min_atoms:
        # if len(current_graph_coalition) <= len(self.events) - len(self.candidate_events):
            return tree_node.P # its score
        
        # Expand if this node has never been visited
        if len(tree_node.children) == 0:
            # subgraph_all_events = tree_node.coalition
            candidate_set = set(self.candidate_events)
            
            # expand_events = [e_idx for e_idx in subgraph_all_events if e_idx in candidate_set] # only care events in the candidate list
            expand_events = tree_node.coalition
            for event in expand_events:
                important_events = [e_idx for e_idx in tree_node.coalition if e_idx != event ]

                # check the state map and merge the same sub-tg-graph (node in the tree)
                find_same = False
                subnode_coalition_key = self._node_key(important_events)
                for key in self.state_map.keys():
                    if key == subnode_coalition_key:
                        new_tree_node = self.state_map[key]
                        find_same = True
                        break
                
                if not find_same:
                    new_tree_node = self.MCTSNodeClass(important_events)
                    self.state_map[subnode_coalition_key] = new_tree_node
                
                # find same child ?
                find_same_child = False
                for child in tree_node.children:
                    if self._node_key(child.coalition) == self._node_key(new_tree_node.coalition):
                        find_same_child = True
                        break
                
                # expand new children
                if not find_same_child:
                    tree_node.children.append(new_tree_node)
            
            # compute scores of all children
            scores = self.compute_scores(self.score_func, tree_node.children, self.event_idx)
            for child, score in zip(tree_node.children, scores):
                child.P = score

        # If this node has children (it has been visited), then directly select one child
        sum_count = sum([c.N for c in tree_node.children])
        # import ipdb; ipdb.set_trace()
        selected_node = max(tree_node.children, key=lambda x: x.Q() + x.U(sum_count))
        v = self.mcts_rollout(selected_node)
        selected_node.W += v
        selected_node.N += 1
        return v


    def mcts(self, verbose=True):
        if verbose:
            print(f"The nodes in graph is {self.graph.number_of_nodes()}")
        start_time = time.time()
        for rollout_idx in range(self.n_rollout):
            self.mcts_rollout(self.root)
            if verbose:
                elapsed_time = time.time() - start_time
                print(f"At the {rollout_idx} rollout, {len(self.state_map)} states have been explored. Time: {elapsed_time:.2f} s")

        explanations = [node for _, node in self.state_map.items()]
        explanations = sorted(explanations, key=lambda x: x.P, reverse=True)
        return explanations
    
    def initialize_tree(self):
        # reset the search tree
        # self.root_coalition = self.events.index.values.tolist()
        self.root_coalition = copy.copy( self.candidate_events )
        self.root = self.MCTSNodeClass(self.root_coalition)
        self.root_key = self._node_key(self.root_coalition)
        self.state_map = {self.root_key: self.root}

    def _node_key(self, coalition):
        return "_".join(map(lambda x: str(x), sorted(coalition) ) )
    
    def set_candidate_events(self):
        if self.model_name == 'tgat':

            pass
        
        pass

    
    @property
    def base_events(self):
        if self.base_events_ is None:
            candidate_events_set_ = set(self.candidate_events)
            self.base_events_ = list(filter(lambda x: x not in candidate_events_set_, self.events.index.values.tolist()))
        return self.base_events_

    # def obtain_important_events(self, tree_node):
    #     coalition_set_ = set(tree_node.coalition)
    #     important_events = list(filter(lambda x: x in coalition_set_, self.candidate_events))
    #     return self.base_events + important_events

    def obtain_base_and_important_events(self, tree_node):
        return self.base_events + tree_node.coalition

    def obtain_base_and_unimportant_events(self, tree_node):
        important_ = set(tree_node.coalition)
        unimportant_events = list(filter(lambda x: x not in important_, self.candidate_events))
        return self.base_events + unimportant_events



class SubgraphXTG(object):
    r"""
    The implementation of paper
    `On Explainability of Graph Neural Networks via Subgraph Explorations <https://arxiv.org/abs/2102.05152>`_.
    
    Args:
        model (:obj:`torch.nn.Module`): The target model prepared to explain
        num_classes(:obj:`int`): Number of classes for the datasets
        num_hops(:obj:`int`, :obj:`None`): The number of hops to extract neighborhood of target node
          (default: :obj:`None`)
        explain_graph(:obj:`bool`): Whether to explain graph classification model (default: :obj:`True`)
        rollout(:obj:`int`): Number of iteration to get the prediction
        min_atoms(:obj:`int`): Number of atoms of the leaf node in search tree
        c_puct(:obj:`float`): The hyperparameter which encourages the exploration
        expand_atoms(:obj:`int`): The number of atoms to expand
          when extend the child nodes in the search tree
        high2low(:obj:`bool`): Whether to expand children nodes from high degree to low degree when
          extend the child nodes in the search tree (default: :obj:`False`)
        local_radius(:obj:`int`): Number of local radius to calculate :obj:`l_shapley`, :obj:`mc_l_shapley`
        sample_num(:obj:`int`): Sampling time of monte carlo sampling approximation for
          :obj:`mc_shapley`, :obj:`mc_l_shapley` (default: :obj:`mc_l_shapley`)
        reward_method(:obj:`str`): The command string to select the
        subgraph_building_method(:obj:`str`): The command string for different subgraph building method,
          such as :obj:`zero_filling`, :obj:`split` (default: :obj:`zero_filling`)
        save_dir(:obj:`str`, :obj:`None`): Root directory to save the explanation results (default: :obj:`None`)
        filename(:obj:`str`): The filename of results
        vis(:obj:`bool`): Whether to show the visualization (default: :obj:`True`)
    Example:
        >>> # For graph classification task
        >>> subgraphx = SubgraphX(model=model, num_classes=2)
        >>> _, explanation_results, related_preds = subgraphx(x, edge_index)
    """
    def __init__(self, model, model_name: str, all_events: DataFrame,  explanation_level: str, device, num_hops: Optional[int] = None, verbose: bool = True,
                 explain_graph: bool = True, rollout: int = 20, min_atoms: int = 1, c_puct: float = 150.0,
                 expand_atoms=14, local_radius=4, sample_num=100, reward_method='mc_l_shapley',
                 load_results=False, save_dir: Optional[str] = None,
                 filename: str = 'example', vis: bool = True):

        self.model = model
        self.model_name = model_name
        self.all_events = all_events
        self.num_users = all_events.iloc[:, 0].max() + 1
        self.model.eval()
        self.device = device
        self.model.to(self.device)
        # self.num_classes = num_classes
        self.num_hops = self.update_num_hops(num_hops)
        self.explain_graph = explain_graph
        self.explanation_level = explanation_level # `node`, `event`, or `graph`. set 'event' as the default
        # node: (node_idx, time); event: (event_idx); graph: nothing to be provided

        self.verbose = verbose

        # mcts hyper-parameters
        self.rollout = rollout
        self.min_atoms = min_atoms
        self.c_puct = c_puct
        self.expand_atoms = expand_atoms
        # self.high2low = high2low

        # reward function hyper-parameters
        self.local_radius = local_radius
        self.sample_num = sample_num
        self.reward_method = reward_method
        # self.subgraph_building_method = subgraph_building_method

        # saving and visualization
        self.vis = vis
        self.load_results = load_results
        self.save_dir = save_dir if save_dir is not None else str(ROOT_DIR/'xgraph'/'saved_mcts_results')
        self.filename = filename
        self.save = True if self.save_dir is not None else False


        # construct TGNN reward function
        self.tgnn_reward_wraper = TGNNRewardWraper(self.model, self.model_name, self.all_events, self.explanation_level)

    def update_num_hops(self, num_hops):
        if num_hops is not None:
            return num_hops

        k = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                k += 1
        return k

    # def get_reward_func(self, value_func, node_idx=None):
    #     if self.explain_graph:
    #         node_idx = None
    #     else:
    #         assert node_idx is not None
    #     return reward_func(reward_method=self.reward_method,
    #                        value_func=value_func,
    #                        node_idx=node_idx,
    #                        local_radius=self.local_radius,
    #                        sample_num=self.sample_num,
    #                        subgraph_building_method=self.subgraph_building_method)

    def get_mcts_class(self, events, event_idx: int = None, node_idx: int = None, score_func: Callable = None, candidate_events=None):
        # if self.explain_graph:
        #     node_idx = None
        # else:
        #     assert node_idx is not None
        if self.explanation_level == 'event':
            pass
        
        return MCTS(events,
                    num_users=self.num_users,
                    event_idx=event_idx,
                    node_idx=node_idx,
                    device=self.device,
                    score_func=score_func,
                    n_rollout=self.rollout,
                    min_atoms=self.min_atoms,
                    c_puct=self.c_puct,
                    expand_atoms=self.expand_atoms,
                    candidate_events=candidate_events
                    )

    def visualization(self, results: list,
                      max_nodes: int, plot_utils: PlotUtils, words: Optional[list] = None,
                      y: Optional[Tensor] = None, title_sentence: Optional[str] = None,
                      vis_name: Optional[str] = None):
        if self.save:
            if vis_name is None:
                vis_name = f"{self.filename}.png"
        else:
            vis_name = None
        tree_node_x = find_closest_node_result(results, max_nodes=max_nodes)
        if self.explain_graph:
            if words is not None:
                plot_utils.plot(tree_node_x.ori_graph,
                                tree_node_x.coalition,
                                words=words,
                                title_sentence=title_sentence,
                                figname=vis_name)
            else:
                plot_utils.plot(tree_node_x.ori_graph,
                                tree_node_x.coalition,
                                x=tree_node_x.data.x,
                                title_sentence=title_sentence,
                                figname=vis_name)
        else:
            subset = self.mcts_state_map.subset
            subgraph_y = y[subset].to('cpu')
            subgraph_y = torch.tensor([subgraph_y[node].item()
                                       for node in tree_node_x.ori_graph.nodes()])
            plot_utils.plot(tree_node_x.ori_graph,
                            tree_node_x.coalition,
                            node_idx=self.mcts_state_map.new_node_idx,
                            title_sentence=title_sentence,
                            y=subgraph_y,
                            figname=vis_name)

    def read_from_MCTSInfo_list(self, MCTSInfo_list):
        if isinstance(MCTSInfo_list[0], dict):
            ret_list = [MCTSNode(device=self.device).load_info(node_info) for node_info in MCTSInfo_list]
        elif isinstance(MCTSInfo_list[0][0], dict):
            ret_list = []
            for single_label_MCTSInfo_list in MCTSInfo_list:
                single_label_ret_list = [MCTSNode(device=self.device).load_info(node_info) for node_info in single_label_MCTSInfo_list]
                ret_list.append(single_label_ret_list)
        return ret_list

    def write_from_MCTSNode_list(self, MCTSNode_list):
        if isinstance(MCTSNode_list[0], MCTSNode):
            ret_list = [node.info for node in MCTSNode_list]
        elif isinstance(MCTSNode_list[0][0], MCTSNode):
            ret_list = []
            for single_label_MCTSNode_list in MCTSNode_list:
                single_label_ret_list = [node.info for node in single_label_MCTSNode_list]
                ret_list.append(single_label_ret_list)
        return ret_list
    
    def find_candidates(self, target_event_idx):
        # TODO: 
        from dig.xgraph.dataset.utils_dataset import tgat_node_reindex
        if self.model_name == 'tgat':
            ngh_finder = self.model.ngh_finder
            num_layers = self.model.num_layers
            u = self.all_events.iloc[target_event_idx, 0]
            i = self.all_events.iloc[target_event_idx, 1]
            ts = self.all_events.iloc[target_event_idx, 2]

            new_u, new_i = tgat_node_reindex(u, i, self.num_users)
            accu_e_idx = [ [target_event_idx+1, target_event_idx+1]] # NOTE: for subsequent '-1' operation
            accu_node = [ [new_u, new_i,] ]
            accu_ts = [ [ts, ts,] ]
            
            for i in range(num_layers):
                last_nodes = accu_node[-1]
                last_ts = accu_ts[-1]
                # import ipdb; ipdb.set_trace()

                out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch = ngh_finder.get_temporal_neighbor(last_nodes, last_ts)
                
                out_ngh_node_batch = out_ngh_node_batch.flatten()
                out_ngh_eidx_batch = out_ngh_eidx_batch.flatten()
                out_ngh_t_batch = out_ngh_t_batch.flatten()
                
                mask = out_ngh_node_batch != 0
                out_ngh_node_batch = out_ngh_node_batch[mask]
                out_ngh_eidx_batch = out_ngh_eidx_batch[mask]
                out_ngh_t_batch = out_ngh_t_batch[mask]

                # import ipdb; ipdb.set_trace()

                out_ngh_node_batch = out_ngh_node_batch.tolist()
                out_ngh_t_batch = out_ngh_t_batch.tolist()
                out_ngh_eidx_batch = (out_ngh_eidx_batch).tolist() 

                accu_node.append(out_ngh_node_batch)
                accu_ts.append(out_ngh_t_batch)
                accu_e_idx.append(out_ngh_eidx_batch)

            unique_e_idx = np.array(list(itertools.chain.from_iterable(accu_e_idx)))
            unique_e_idx = unique_e_idx[ unique_e_idx != 0 ] # NOTE: 0 are padded e_idxs
            unique_e_idx = unique_e_idx - 1 # NOTE: -1, because ngh_finder stored +1 e_idxs
            unique_e_idx = np.unique(unique_e_idx).tolist()
            
            return unique_e_idx
            
        else:
            raise NotImplementedError


    def explain(self,
                node_idx: Optional[int] = None,
                time: Optional[float] = None,
                event_idx: Optional[int] = None,
                saved_MCTSInfo_list: Optional[List[List]] = None):
        
        if saved_MCTSInfo_list is not None:
            results = self.read_from_MCTSInfo_list(saved_MCTSInfo_list)
            tree_node_x = find_closest_node_result(results)
            return tree_node_x
        
        # support event-level first
        if self.explanation_level == 'node':
            # node_idx, time
            pass

        elif self.explanation_level == 'event': # we now only care node/edge(event) level explanations, graph-level explanation is temporarily suspended
            # extract the subgraph that the search begins with
            assert event_idx is not None
            subgraph_df = k_hop_temporal_subgraph(self.all_events, num_hops=3, event_idx=event_idx)
            # subgraph_df = self.all_events.iloc[:event_idx+1, :].copy()
            # import ipdb; ipdb.set_trace()
            self.ori_subgraph_df = subgraph_df
            # set reward function
            # self.tgnn_reward_wraper.compute_original_score(self.all_events.index.values.tolist(), event_idx)
            self.tgnn_reward_wraper.compute_original_score(subgraph_df.index.values.tolist(), event_idx)
            # import ipdb; ipdb.set_trace()

            payoff_func = self.tgnn_reward_wraper

            # search
            candidate_events = self.find_candidates(event_idx)
            if len(candidate_events) > 20:
                candidate_events = candidate_events[-10:]
                print('more than 10 candidates, used 10 ones:')
                print(candidate_events)


            self.mcts_state_map = self.get_mcts_class(events=subgraph_df, event_idx=event_idx, candidate_events=candidate_events, score_func=payoff_func)

            print('search graph:')
            print(subgraph_df.to_string(max_rows=50))
            print(f'{len(candidate_events)} candicate events:', self.mcts_state_map.candidate_events)
            
            
            # import ipdb; ipdb.set_trace()

            results = self.mcts_state_map.mcts(verbose=self.verbose)

        elif self.explanation_level == 'graph':
            pass
        else: raise NotImplementedError('Wrong explanaion level')

        tree_node_x = find_closest_node_result(results)
        results = sorted(results, key=lambda x:x.P)

        ori_event_idxs = self.ori_subgraph_df.index.to_list() # all edge_idxs that gnn model can see
        candidates_idxs = self.mcts_state_map.candidate_events # candidates to remove

        print('\nsearched tree nodes (preserved edge idxs):')
        for node in results:
            # preserved_events = preserved_candidates(node.coalition, ori_event_idxs, candidates_idxs)
            # removed_idxs = obtain_removed_idxs(node.coalition, self.ori_subgraph_df.index.to_list())
            # preserved_events_gnn_score = self.tgnn_reward_wraper(preserved_events, event_idx)
            print(node.coalition, ': ', node.P)

        # best_preserved_events = preserved_candidates(tree_node_x.coalition, ori_event_idxs, candidates_idxs)
        important_events = tree_node_x.coalition
        # removed_idxs = obtain_removed_idxs(tree_node_x.coalition, self.ori_subgraph_df.index.to_list())

        tree_results = self.write_from_MCTSNode_list(results)
        return tree_results, ori_event_idxs, candidates_idxs, important_events
        # return tree_node_x

        # keep the important structure
        masked_node_list = [node for node in range(tree_node_x.data.x.shape[0])
                            if node in tree_node_x.coalition]

        # remove the important structure, for node_classification,
        # remain the node_idx when remove the important structure
        maskout_node_list = [node for node in range(tree_node_x.data.x.shape[0])
                             if node not in tree_node_x.coalition]
        # if not self.explain_graph:
        #     maskout_node_list += [self.new_node_idx]


        # masked_score = gnn_score(masked_node_list,
        #                          tree_node_x.data,
        #                          value_func=value_func,
        #                          subgraph_building_method=self.subgraph_building_method)

        # maskout_score = gnn_score(maskout_node_list,
        #                           tree_node_x.data,
        #                           value_func=value_func,
        #                           subgraph_building_method=self.subgraph_building_method)

        # sparsity_score = sparsity(masked_node_list, tree_node_x.data,
        #                           subgraph_building_method=self.subgraph_building_method)

        # results = self.write_from_MCTSNode_list(results)
        # related_pred = {'masked': masked_score,
        #                 'maskout': maskout_score,
        #                 'origin': probs[node_idx, label].item(),
        #                 'sparsity': sparsity_score}

        # return results, related_pred

    def __call__(self, node_idx: Union[int, None] = None, event_idx: Union[int, None] = None, max_events: Union[int, None] = None):
        r""" explain the GNN behavior for the graph using SubgraphX method
        Args:
            node_idx: the target node index to explain for node prediction tasks
            event_idx: the target event index to explain for edge prediction tasks
            max_events: default max events to preserve in the candidate events set. Not the number of events in the final sub temporal graph
        """
        

        # collect all the class index
        # labels = tuple(label for label in range(self.num_classes))
        # ex_labels = tuple(torch.tensor([label]).to(self.device) for label in labels)

        if self.load_results:
            assert os.path.isfile(os.path.join(self.save_dir, f"{self.filename}.pt"))
            saved_results = torch.load(os.path.join(self.save_dir, f"{self.filename}.pt"))
        else:
            saved_results = None
        
        # explanation_results, related_pred = self.explain()
        tree_results, ori_event_idxs, candidates_idxs, important_events = self.explain(node_idx=node_idx,
                                                    event_idx=event_idx, 
                                                    saved_MCTSInfo_list=saved_results)
        
        if self.save:
            torch.save(tree_results, os.path.join(self.save_dir, f"{self.filename}.pt"))

        # for label_idx, label in enumerate(ex_labels):
        #     results, related_pred = self.explain(x, edge_index,
        #                                          label=label,
        #                                          max_nodes=max_nodes,
        #                                          node_idx=node_idx,
        #                                          saved_MCTSInfo_list=saved_results)
        #     related_preds.append(related_pred)
        #     explanation_results.append(results)


        return tree_results, ori_event_idxs, candidates_idxs, important_events
