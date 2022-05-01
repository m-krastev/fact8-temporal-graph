
import os
import copy
import math
import time
import itertools
from sklearn.utils import shuffle
from tqdm import tqdm
import torch
import numpy as np
import networkx as nx
import pandas as pd
from pandas import DataFrame
from typing import Union
from functools import partial
from typing import List, Tuple, Dict
from torch_geometric.data import Batch, Data
from typing import Callable, Union, Optional


from dig import ROOT_DIR
from dig.xgraph.models.ext.tgat.module import TGAN
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


def print_nodes(tree_nodes):
    print('\nSearched tree nodes (preserved edge idxs in candidates):')
    for node in tree_nodes:
        # preserved_events = preserved_candidates(node.coalition, ori_event_idxs, candidates_idxs)
        # removed_idxs = obtain_removed_idxs(node.coalition, self.ori_subgraph_df.index.to_list())
        # preserved_events_gnn_score = self.tgnn_reward_wraper(preserved_events, event_idx)
        print(sorted(node.coalition), ': ', node.P)

def find_best_node_result(results):
    """ return the highest reward tree_node with its subgraph is smaller than max_nodes """
    results = sorted(results, key=lambda x: len(x.coalition))

    result_node = results[0]
    for result_idx in range(len(results)):
        x = results[result_idx]
        # if len(x.coalition) <= max_nodes and x.P > result_node.P:
        if x.P > result_node.P:
            result_node = x
    return result_node


class MCTSNode(object):
    def __init__(self, coalition: list = None, data: Data = None, created_by_remove: int = None, 
                #  ori_graph: nx.Graph = None,
                 c_puct: float = 10.0, W: float = 0, N: int = 0, P: float = 0,
                 load_dict: Optional[Dict] = None):
        self.data = data # node data or edge data? or subgraph data? ignore now.
        self.coalition = coalition  # in our case, the coalition should be edge indices?
        # self.ori_graph = ori_graph
        # self.device = device
        self.c_puct = c_puct
        self.children = []
        self.created_by_remove = created_by_remove # created by remove which edge from its parents
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
        # self.ori_graph = info_dict['ori_graph']
        # self.data = info_dict['data'].to(self.device)
        self.children = []
        return self


def compute_scores(score_func, base_events, children, target_event_idx):
    results = []
    for child in children:
        if child.P == 0:
            # score = score_func(child.coalition, child.data)
            score = score_func( base_events + child.coalition, target_event_idx)
        else:
            score = child.P
        results.append(score)
    return results

def base_and_important_events(base_events, candidate_events, coalition):
    return base_events + coalition

def base_and_unimportant_events(base_events, candidate_events, coalition):
    important_ = set(coalition)
    unimportant_events = list(filter(lambda x: x not in important_, candidate_events))
    return base_events + unimportant_events


class MCTS(object):
    r"""
    Monte Carlo Tree Search Method.
    Args:
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
                 base_events = None,
                 num_users: int = None,
                 n_rollout: int = 10, min_atoms: int = 5, c_puct: float = 10.0,
                #  expand_atoms: int = 14,
                 node_idx: int = None, event_idx: int = None, score_func: Callable = None, device='cpu'):
        
        self.events = events # subgraph events or total events? subgraph events
        self.num_users = num_users
        self.subgraph_num_nodes = self.events.iloc[:, 0].nunique() + self.events.iloc[:, 1].nunique()
        # self.graph = to_networkx_tg(events)
        # self.node_X = node_X # node features
        # self.event_X = event_X # event features
        self.node_idx = node_idx # node index to explain
        self.event_idx = event_idx # event index to explain

        # improve the strategy later
        # self.candidate_events = sorted(self.events.index.values.tolist())[-6:-1]
        # self.candidate_events = sorted(self.events.index.values.tolist())[-10:]
        # self.candidate_events = [10, 11, 12, 13, 14, 15, 19]
        self.candidate_events = candidate_events
        self.base_events = base_events

        # we only care these events, other events are preserved as is.
        # currently only take 10 temporal edges into consideration.
        
        self.device = device
        self.num_nodes = self.events.iloc[:, 0].nunique() + self.events.iloc[:, 1].nunique()


        self.score_func = score_func

        self.n_rollout = n_rollout
        self.min_atoms = min_atoms
        self.c_puct = c_puct
        # self.expand_atoms = expand_atoms
        # self.high2low = high2low
        self.new_node_idx = None
        self.data = None

        self.MCTSNodeClass = partial(MCTSNode, data=self.data, 
                                     c_puct=self.c_puct
                                     )


        self._initialize_tree()
        self._initialize_recorder()
        
    
    def _initialize_recorder(self):
        self.recorder = {
            'rollout': [],
            'runtime': [],
            'best_reward': [],
            'num_states': []
        }
    
    
    

    def mcts_rollout(self, tree_node):
        """
        The tree_node now is a set of events
        """
        # import ipdb; ipdb.set_trace()
        if len( tree_node.coalition ) < self.min_atoms:
        # if len(current_graph_coalition) <= len(self.events) - len(self.candidate_events):
            return tree_node.P # its score
        
        # Expand if this node has never been visited
        # Expand if this node has un-expanded children
        if len(tree_node.children) != len(tree_node.coalition):
            # expand_events = tree_node.coalition
            
            exist_children = set(map( lambda x: x.created_by_remove, tree_node.children ))
            not_exist_children = list(filter(lambda e_idx:e_idx not in exist_children, tree_node.coalition ) )
            not_exist_children_score = {}
            for event in not_exist_children:
                children_coalition = [e_idx for e_idx in tree_node.coalition if e_idx != event ]
                not_exist_children_score[event] = self.compute_time_score(children_coalition)
            
            # expand only one event
            expand_event = max( not_exist_children_score, key=not_exist_children_score.get )
            expand_events = [expand_event, ]

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
                    new_tree_node = self.MCTSNodeClass(important_events, created_by_remove=event)
                    self.state_map[subnode_coalition_key] = new_tree_node
                
                # find same child ?
                find_same_child = False
                for child in tree_node.children:
                    if self._node_key(child.coalition) == self._node_key(new_tree_node.coalition):
                        find_same_child = True
                        break
                
                # expand new childrens
                if not find_same_child:
                    tree_node.children.append(new_tree_node)
            
            # compute scores of all children
            scores = compute_scores(self.score_func, self.base_events, tree_node.children, self.event_idx)
            for child, score in zip(tree_node.children, scores):
                child.P = score

        # import ipdb; ipdb.set_trace()

        # If this node has children (it has been visited), then directly select one child
        sum_count = sum([c.N for c in tree_node.children])
        # import ipdb; ipdb.set_trace()
        # selected_node = max(tree_node.children, key=lambda x: x.Q() + x.U(sum_count))
        selected_node = max(tree_node.children, key=lambda x: self.compute_node_score(x, sum_count))

        
        v = self.mcts_rollout(selected_node)
        selected_node.W += v
        selected_node.N += 1
        return v
    
    def compute_time_score(self, coalition):
        beta = 3
        ts = self.events['ts'][coalition].values # np array
        delta_ts = self.curr_t - ts
        t_score_exp = np.exp( beta * -1 * delta_ts)
        t_score_exp = np.sum( t_score_exp )
        return t_score_exp

    
    def compute_node_score(self, node, sum_count):
        """
        node score computation strategy
        """
        # import ipdb; ipdb.set_trace()
        # time score
        # tscore_eff = -10 # 0.1
        tscore_coef = 0.1 # -100, -50, -10, -5, -1, 0, 0.5

        beta = 3
        max_event_idx = max(self.root.coalition)
        curr_t = self.events['ts'][max_event_idx]
        ts = self.events['ts'][node.coalition].values # np array
        delta_ts = curr_t - ts
        t_score_exp = np.exp( beta * -1 * delta_ts)
        t_score_exp = np.sum( t_score_exp )

        # uct score
        uct_score = node.Q() + node.U(sum_count)

        # final score
        final_score = uct_score + tscore_coef * t_score_exp

        return final_score


    def mcts(self, verbose=True):
        if verbose:
            print(f"The nodes in graph is {self.subgraph_num_nodes}")

        start_time = time.time()
        pbar = tqdm(range(self.n_rollout), total=self.n_rollout, desc='mcts simulating')
        for rollout_idx in pbar:
            self.mcts_rollout(self.root)
            if verbose:
                elapsed_time = time.time() - start_time
            pbar.set_postfix({'states': len(self.state_map)})
            # print(f"At the {rollout_idx} rollout, {len(self.state_map)} states have been explored. Time: {elapsed_time:.2f} s")
            
            # record
            self.recorder['rollout'].append(rollout_idx)
            self.recorder['runtime'].append(elapsed_time)
            self.recorder['best_reward'].append( np.max(list(map(lambda x: x.P, self.state_map.values()))) )
            self.recorder['num_states'].append( len(self.state_map) )

        end_time = time.time()
        self.run_time = end_time - start_time

        tree_nodes = list(self.state_map.values())

        return tree_nodes
    
    def _initialize_tree(self):
        # reset the search tree
        # self.root_coalition = self.events.index.values.tolist()
        self.root_coalition = copy.copy( self.candidate_events )
        self.root = self.MCTSNodeClass(self.root_coalition)
        self.root_key = self._node_key(self.root_coalition)
        self.state_map = {self.root_key: self.root}

        max_event_idx = max(self.root.coalition)
        self.curr_t = self.events['ts'][max_event_idx]

    def _node_key(self, coalition):
        return "_".join(map(lambda x: str(x), sorted(coalition) ) ) # NOTE: have sorted
    

class BaseExplainerTG(object):
    def __init__(self, model: Union[TGAN, None], model_name: str, explainer_name: str, dataset_name: str, all_events: str, explanation_level: str, device, 
                verbose: bool = True, results_dir: Optional[str] = None, debug_mode: bool=True) -> None:
        """
        results_dir: dir for saving value results, e.g., fidelity_sparsity. Not mcts_node_list
        """
        self.model = model
        self.model_name = model_name
        self.explainer_name = explainer_name # self's name
        self.dataset_name = dataset_name
        self.all_events = all_events
        self.num_users = all_events.iloc[:, 0].max() + 1
        self.explanation_level = explanation_level
        
        self.device = device
        self.verbose = verbose
        self.results_dir = results_dir
        self.debug_mode = debug_mode
        
        self.model.eval()
        self.model.to(self.device)

        # construct TGNN reward function
        self.tgnn_reward_wraper = TGNNRewardWraper(self.model, self.model_name, self.all_events, self.explanation_level)

    def find_candidates(self, target_event_idx):
        # TODO: implementation for other models
        from dig.xgraph.dataset.utils_dataset import tgat_node_reindex
        from dig.xgraph.method.tg_score import _set_tgat_events_idxs # NOTE: important
        _set_tgat_events_idxs
        if self.model_name == 'tgat':
            ngh_finder = self.model.ngh_finder
            num_layers = self.model.num_layers
            num_neighbors = self.model.num_neighbors # NOTE: important
            edge_idx_preserve_list = _set_tgat_events_idxs( self.ori_subgraph_df.index.values.tolist() ) # NOTE: important, make sure in the tgat space.
            # import ipdb; ipdb.set_trace()

            u = self.all_events.iloc[target_event_idx, 0]
            i = self.all_events.iloc[target_event_idx, 1]
            ts = self.all_events.iloc[target_event_idx, 2]

            new_u, new_i = tgat_node_reindex(u, i, self.num_users)
            # accu_e_idx = [ [target_event_idx+1, target_event_idx+1]] # NOTE: for subsequent '-1' operation
            accu_e_idx = [ ] # NOTE: important?
            accu_node = [ [new_u, new_i,] ]
            accu_ts = [ [ts, ts,] ]
            
            for i in range(num_layers):
                last_nodes = accu_node[-1]
                last_ts = accu_ts[-1]
                # import ipdb; ipdb.set_trace()

                out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch = ngh_finder.get_temporal_neighbor(
                                                                                    last_nodes, 
                                                                                    last_ts, 
                                                                                    num_neighbors=num_neighbors,
                                                                                    edge_idx_preserve_list=edge_idx_preserve_list,
                                                                                    )
                
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
            
            
        else:
            raise NotImplementedError
        
        candidate_events = unique_e_idx
        threshold_num = 20
        if len(candidate_events) > threshold_num:
            candidate_events = candidate_events[-threshold_num:]
            candidate_events = sorted(candidate_events)
        
        if self.debug_mode:
            print(f'{len(unique_e_idx)} seen events, used {len(candidate_events)} as candidates:')
            print(candidate_events)
        
        return candidate_events
    
    def _set_ori_subgraph(self, num_hops, event_idx):
        subgraph_df = k_hop_temporal_subgraph(self.all_events, num_hops=num_hops, event_idx=event_idx)
        self.ori_subgraph_df = subgraph_df


    def _set_candidate_events(self, event_idx):
        self.candidate_events = self.find_candidates(event_idx)
        # self.candidate_events = shuffle( candidate_events ) # strategy 1
        # self.candidate_events = candidate_events # strategy 2
        # self.candidate_events.reverse()
        # self.candidate_events = candidate_events # strategy 3
        candidate_events_set_ = set(self.candidate_events)
        assert hasattr(self, 'ori_subgraph_df')
        self.base_events = list(filter(lambda x: x not in candidate_events_set_, self.ori_subgraph_df.index.values.tolist())) # NOTE: ori_subgraph_df

    def _set_tgnn_wraper(self, event_idx):
        assert hasattr(self, 'ori_subgraph_df')
        self.tgnn_reward_wraper.compute_original_score(self.ori_subgraph_df.index.values.tolist(), event_idx)
    
    def _initialize(self, event_idx):
        self._set_ori_subgraph(num_hops=3, event_idx=event_idx)
        self._set_candidate_events(event_idx)
        self._set_tgnn_wraper(event_idx)


class SubgraphXTG(BaseExplainerTG):
    """
    MCTS based temporal graph GNN explainer
    """
    def __init__(self, model, model_name: str, explainer_name: str, dataset_name: str, all_events: DataFrame,  explanation_level: str, device, verbose: bool = True, results_dir = None,
                 rollout: int = 20, min_atoms: int = 1, c_puct: float = 150.0,
                 expand_atoms=14,
                 load_results=False, mcts_saved_dir: Optional[str] = None, save_results: bool= True,
                 debug_mode: bool = True
                 ):

        super(SubgraphXTG, self).__init__(model=model, 
                                          model_name=model_name,
                                          explainer_name=explainer_name,
                                          dataset_name=dataset_name,
                                          all_events=all_events,
                                          explanation_level=explanation_level,
                                          device=device,
                                          verbose=verbose,
                                          results_dir=results_dir,
                                          debug_mode=debug_mode
                                          )
        

        # mcts hyper-parameters
        self.rollout = rollout
        self.min_atoms = min_atoms
        self.c_puct = c_puct

        # saving and visualization
        self.load_results = load_results
        self.mcts_saved_dir = mcts_saved_dir # dir for saving mcts nodes, not evaluation results ( e.g., fidelity )
        # self.mcts_saved_filename = mcts_saved_filename
        self.save = save_results


    def get_mcts_class(self, events, event_idx: int = None, node_idx: int = None, score_func: Callable = None, candidate_events=None,
                        base_events=None,
    ):
        # if self.explanation_level == 'event':
        #     pass
        
        return MCTS(events=events,
                    candidate_events=candidate_events,
                    base_events=base_events,
                    num_users=self.num_users,
                    n_rollout=self.rollout,
                    min_atoms=self.min_atoms,
                    c_puct=self.c_puct,
                    score_func=score_func,
                    device=self.device,

                    event_idx=event_idx,
                    node_idx=node_idx,
                    )


    # @staticmethod
    def read_from_MCTSInfo_list(self, MCTSInfo_list):
        if isinstance(MCTSInfo_list[0], dict):
            ret_list = [MCTSNode().load_info(node_info) for node_info in MCTSInfo_list]
        else: raise NotImplementedError
        # elif isinstance(MCTSInfo_list[0][0], dict):
        #     ret_list = []
        #     for single_label_MCTSInfo_list in MCTSInfo_list:
        #         single_label_ret_list = [MCTSNode(device=self.device).load_info(node_info) for node_info in single_label_MCTSInfo_list]
        #         ret_list.append(single_label_ret_list)
        return ret_list

    def write_from_MCTSNode_list(self, MCTSNode_list):
        if isinstance(MCTSNode_list[0], MCTSNode):
            ret_list = [node.info for node in MCTSNode_list]
        else: raise NotImplementedError
        # elif isinstance(MCTSNode_list[0][0], MCTSNode):
        #     ret_list = []
        #     for single_label_MCTSNode_list in MCTSNode_list:
        #         single_label_ret_list = [node.info for node in single_label_MCTSNode_list]
        #         ret_list.append(single_label_ret_list)
        return ret_list
    
    def _saved_file_path(self, event_idx):
        filename = f"{self.model_name}_{self.dataset_name}_{event_idx}_mcts_node_info.pt"
        path = os.path.join(self.mcts_saved_dir, filename)
        return path

    def explain(self,
                node_idx: Optional[int] = None,
                time: Optional[float] = None,
                event_idx: Optional[int] = None,
                ):
        
        
        # support event-level first
        if self.explanation_level == 'node':
            # node_idx + event_idx?
            pass

        elif self.explanation_level == 'event': # we now only care node/edge(event) level explanations, graph-level explanation is temporarily suspended
            assert event_idx is not None
            

            # search
            self.mcts_state_map = self.get_mcts_class(events=self.ori_subgraph_df, event_idx=event_idx, 
                                                      candidate_events=self.candidate_events, 
                                                      base_events=self.base_events,
                                                      score_func=self.tgnn_reward_wraper)
            
            if self.debug_mode:
                print('search graph:')
                print(self.ori_subgraph_df.to_string(max_rows=50))
                # print(f'{len(self.candidate_events)} candicate events:', self.mcts_state_map.candidate_events)
            tree_nodes = self.mcts_state_map.mcts(verbose=self.verbose) # search

        else: raise NotImplementedError('Wrong explanaion level')

        tree_node_x = find_best_node_result(tree_nodes)
        tree_nodes = sorted(tree_nodes, key=lambda x:x.P)

        print_nodes(tree_nodes)

        return tree_nodes, tree_node_x
    
    def _load_saved_nodes_info(self, event_idx):
        path = self._saved_file_path(event_idx)
        assert os.path.isfile(path)
        saved_contents = torch.load(path)
        
        saved_MCTSInfo_list = saved_contents['saved_MCTSInfo_list']
        tree_nodes = self.read_from_MCTSInfo_list(saved_MCTSInfo_list)
        tree_node_x = find_best_node_result(tree_nodes)

        return tree_nodes, tree_node_x
        
    def _save_mcts_nodes_info(self, tree_nodes, event_idx):
        saved_contents = {
            # 'candidate_events': self.mcts_state_map.candidate_events,
            'saved_MCTSInfo_list': self.write_from_MCTSNode_list(tree_nodes),
            # 'original_scores': self.tgnn_reward_wraper.original_scores,
            # 'ori_subgraph_df': self.ori_subgraph_df
        }
        path = self._saved_file_path(event_idx)
        torch.save(saved_contents, path)
        print(f'results saved at {path}')

    def _save_mcts_recorder(self,):
        # save records
        recorder_df = pd.DataFrame(self.mcts_state_map.recorder)
        record_filename = ROOT_DIR.parent/'benchmarks'/'results'/f'{self.model_name}_{self.dataset_name}_{self.mcts_state_map.event_idx}_mcts_recorder.csv'
        recorder_df.to_csv(record_filename, index=False)
        print(f'mcts recorder saved at {str(record_filename)}')


    def __call__(self, node_idxs: Union[int, None] = None, event_idxs: Union[int, None] = None):
        """
        Args:
            node_idxs: the target node index to explain for node prediction tasks
            event_idxs: the target event index to explain for edge prediction tasks
        """

        if isinstance(event_idxs, int):
            event_idxs = [event_idxs, ]

        results_list = []
        for i, event_idx in enumerate(event_idxs):
            print(f'\nexplain {i}-th: {event_idx}')
            self._initialize(event_idx)

            if self.load_results:
                tree_nodes, tree_node_x = self._load_saved_nodes_info(event_idx)
            else:
                tree_nodes, tree_node_x = self.explain(event_idx=event_idx,
                                                    )
                self._save_mcts_recorder() # always store
                if self.save and not self.load_results: # sometimes store
                    self._save_mcts_nodes_info(tree_nodes, event_idx)
            
            results_list.append([tree_nodes, tree_node_x])

        return results_list
        # return tree_nodes, tree_node_x
