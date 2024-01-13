
import os
import copy
import math
import time
import random
from pathlib import Path
from sklearn.utils import shuffle
import torch
import numpy as np
import networkx as nx
import pandas as pd
from pandas import DataFrame
from typing import Union, Optional
from typing import Callable, Union, Optional
from tqdm import tqdm

from tgnnexplainer import ROOT_DIR
from tgnnexplainer.xgraph.method.base_explainer_tg import BaseExplainerTG
from tgnnexplainer.xgraph.method.other_baselines_tg import _create_explainer_input


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
    for i, node in enumerate(tree_nodes):
        # preserved_events = preserved_candidates(node.coalition, ori_event_idxs, candidates_idxs)
        # removed_idxs = obtain_removed_idxs(node.coalition, self.ori_subgraph_df.index.to_list())
        # preserved_events_gnn_score = self.tgnn_reward_wraper(preserved_events, event_idx)
        print(i, sorted(node.coalition), ': ', node.P)

def find_best_node_result(all_nodes, min_atoms=6):
    """ return the highest reward tree_node with its subgraph is smaller than max_nodes """
    all_nodes = filter( lambda x: len(x.coalition) <= min_atoms, all_nodes ) # filter using the min_atoms
    best_node = max(all_nodes, key=lambda x: x.P)
    return best_node

    # all_nodes = sorted(all_nodes, key=lambda x: len(x.coalition))
    # result_node = all_nodes[0]
    # for result_idx in range(len(all_nodes)):
    #     x = all_nodes[result_idx]
    #     # if len(x.coalition) <= max_nodes and x.P > result_node.P:
    #     if x.P > result_node.P:
    #         result_node = x
    # return result_node


class MCTSNode(object):
    def __init__(self, coalition: list = None, created_by_remove: int = None, 
                 c_puct: float = 10.0, W: float = 0, N: int = 0, P: float = 0, Sparsity: float = 1,
                 ):
        self.coalition = coalition  # in our case, the coalition should be edge indices?
        self.c_puct = c_puct
        self.children = []
        self.created_by_remove = created_by_remove # created by remove which edge from its parents
        self.W = W  # sum of node value
        self.N = N  # times of arrival
        self.P = P  # property score (reward)
        self.Sparsity = Sparsity # len(self.coalition)/len(candidates)

    

    def Q(self):
        return self.W / self.N if self.N > 0 else 0

    def U(self, n):
        # return self.c_puct * self.P * math.sqrt(n) / (1 + self.N)
        return self.c_puct * math.sqrt(n) / (1 + self.N)

    @property
    def info(self):
        info_dict = {
            'coalition': self.coalition,
            'created_by_remove': self.created_by_remove,
            'c_puct': self.c_puct,
            'W': self.W,
            'N': self.N,
            'P': self.P,
            'Sparsity': self.Sparsity,
        }
        return info_dict

    def load_info(self, info_dict):
        self.coalition = info_dict['coalition']
        self.created_by_remove = info_dict['created_by_remove']
        self.c_puct = info_dict['c_puct']
        self.W = info_dict['W']
        self.N = info_dict['N']
        self.P = info_dict['P']
        self.Sparsity = info_dict['Sparsity']

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
    def __init__(self, events: DataFrame, candidate_events = None, base_events = None, candidate_initial_weights = None,
                 node_idx: int = None, event_idx: int = None,
                 n_rollout: int = 10, min_atoms: int = 5, c_puct: float = 10.0,
                 score_func: Callable = None, 
                #  device='cpu'
                 ):
        
        self.events = events # subgraph events or total events? subgraph events
        # self.num_users = num_users
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
        self.candidate_initial_weights = candidate_initial_weights

        # we only care these events, other events are preserved as is.
        # currently only take 10 temporal edges into consideration.
        
        # self.device = device
        self.num_nodes = self.events.iloc[:, 0].nunique() + self.events.iloc[:, 1].nunique()


        self.score_func = score_func

        self.n_rollout = n_rollout
        self.min_atoms = min_atoms
        self.c_puct = c_puct
        # self.expand_atoms = expand_atoms
        # self.high2low = high2low
        self.new_node_idx = None
        # self.data = None

        # self.MCTSNodeClass = partial(MCTSNode,
        #                              c_puct=self.c_puct,
        #                              )


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
        # if len( tree_node.coalition ) < self.min_atoms:
        if len( tree_node.coalition ) < 1:
            return tree_node.P # its score
        
        # Expand if this node has never been visited
        # Expand if this node has un-expanded children
        if len(tree_node.children) != len(tree_node.coalition):
            # expand_events = tree_node.coalition
            
            exist_children = set(map( lambda x: x.created_by_remove, tree_node.children ))
            not_exist_children = list(filter(lambda e_idx:e_idx not in exist_children, tree_node.coalition ) )
            
            expand_events = self._select_expand_candidates(not_exist_children)

            # not_exist_children_score = {}
            # for event in not_exist_children:
            #     children_coalition = [e_idx for e_idx in tree_node.coalition if e_idx != event ]
            #     not_exist_children_score[event] = self.compute_action_score(children_coalition, expand_event=event)
            # # expand only one event
            # # expand_event = max( not_exist_children_score, key=not_exist_children_score.get )
            # expand_event = min( not_exist_children_score, key=not_exist_children_score.get ) # NOTE: min
            
            # expand_events = [expand_events[0], ]

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
                    # new_tree_node = self.MCTSNodeClass(
                    #     coalition=important_events, created_by_remove=event)
                    new_tree_node = MCTSNode(
                        coalition=important_events,
                        created_by_remove=event,
                        c_puct=self.c_puct,
                        Sparsity=len(important_events)/len(self.candidate_events)
                        )

                    self.state_map[subnode_coalition_key] = new_tree_node
                
                # find same child ?
                find_same_child = False
                for child in tree_node.children:
                    if self._node_key(child.coalition) == self._node_key(new_tree_node.coalition):
                        find_same_child = True
                        break
                if not find_same_child:
                    tree_node.children.append(new_tree_node)

                # coutinue until one valid child is expanded, otherewize this rollout will be wasted
                if not find_same:
                    break
                else: continue
            
            # compute scores of all children
            scores = compute_scores(self.score_func, self.base_events, tree_node.children, self.event_idx)
            # import ipdb; ipdb.set_trace()
            for child, score in zip(tree_node.children, scores):
                child.P = score

        # import ipdb; ipdb.set_trace()

        # If this node has children (it has been visited), then directly select one child
        sum_count = sum([c.N for c in tree_node.children])
        # import ipdb; ipdb.set_trace()
        # selected_node = max(tree_node.children, key=lambda x: x.Q() + x.U(sum_count))
        selected_node = max(tree_node.children, key=lambda x: self._compute_node_score(x, sum_count))
        
        v = self.mcts_rollout(selected_node)
        selected_node.W += v
        selected_node.N += 1
        return v
    
    def _select_expand_candidates(self, not_exist_children):
        assert self.candidate_initial_weights is not None
        return sorted(not_exist_children, key=self.candidate_initial_weights.get)

        
        if self.candidate_initial_weights is not None:
            # return min(not_exist_children, key=self.candidate_initial_weights.get)
            
            # v1
            if np.random.random() > 0.5:
                return min(not_exist_children, key=self.candidate_initial_weights.get)
            else:
                return np.random.choice(not_exist_children)
            
            # v2
            # return sorted(not_exist_children, key=self.candidate_initial_weights.get) # ascending
            

        else:
            # return np.random.choice(not_exist_children)
            # return sorted(not_exist_children)[0]
            return shuffle(not_exist_children)

    
    def _compute_node_score(self, node, sum_count):
        """
        score for selecting a path
        """
        # import ipdb; ipdb.set_trace()
        # time score
        # tscore_eff = -10 # 0.1
        # tscore_coef = 0.1 # -100, -50, -10, -5, -1, 0, 0.5
        tscore_coef = 0
        beta = -3

        max_event_idx = max(self.root.coalition)
        curr_t = self.events['ts'][max_event_idx-1]
        ts = self.events['ts'][self.events.e_idx.isin(node.coalition)].values
        # np.array(node.coalition)-1].values # np array
        delta_ts = curr_t - ts
        t_score_exp = np.exp( beta * delta_ts)
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
            # self.recorder['best_reward'].append( np.max(list(map(lambda x: x.P, self.state_map.values()))) )
            curr_best_node = find_best_node_result(self.state_map.values(), self.min_atoms)
            self.recorder['best_reward'].append( curr_best_node.P )
            self.recorder['num_states'].append( len(self.state_map) )

        end_time = time.time()
        self.run_time = end_time - start_time

        tree_nodes = list(self.state_map.values())

        return tree_nodes
    
    def _initialize_tree(self):
        # reset the search tree
        # self.root_coalition = self.events.index.values.tolist()
        self.root_coalition = copy.copy( self.candidate_events )
        self.root = MCTSNode(self.root_coalition, created_by_remove=-1, c_puct=self.c_puct, Sparsity=1.0)
        self.root_key = self._node_key(self.root_coalition)
        self.state_map = {self.root_key: self.root}

        max_event_idx = max(self.root.coalition)
        self.curr_t = self.events['ts'][self.events.e_idx==max_event_idx].values[0]

    def _node_key(self, coalition):
        return "_".join(map(lambda x: str(x), sorted(coalition) ) ) # NOTE: have sorted
    



class SubgraphXTG(BaseExplainerTG):
    """
    MCTS based temporal graph GNN explainer
    """

    def __init__(self, model, model_name: str, explainer_name: str, dataset_name: str, all_events: DataFrame,  explanation_level: str, device, 
                verbose: bool = True, results_dir = None, debug_mode: bool = True,
                # specific params
                rollout: int = 20, min_atoms: int = 1, c_puct: float = 10.0,
                # expand_atoms=14,
                load_results=False, mcts_saved_dir: Optional[str] = None, save_results: bool= True,
                pg_explainer_model=None, pg_positive=True,
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
        self.pg_explainer_model = pg_explainer_model # to assign initial weights using a trained pg_explainer_tg
        self.pg_positive = pg_positive
        self.suffix = self._path_suffix(pg_explainer_model, pg_positive)

    @staticmethod
    def read_from_MCTSInfo_list(MCTSInfo_list):
        if isinstance(MCTSInfo_list[0], dict):
            ret_list = [MCTSNode().load_info(node_info) for node_info in MCTSInfo_list]
        else: 
            raise NotImplementedError
        return ret_list

    def write_from_MCTSNode_list(self, MCTSNode_list):
        if isinstance(MCTSNode_list[0], MCTSNode):
            ret_list = [node.info for node in MCTSNode_list]
        else: 
            raise NotImplementedError
        return ret_list

    def explain(self,
                node_idx: Optional[int] = None,
                time: Optional[float] = None,
                event_idx: Optional[int] = None,
                ):
        # support event-level first
        if self.explanation_level == 'node':
            raise NotImplementedError
            # node_idx + event_idx?

        elif self.explanation_level == 'event': # we now only care node/edge(event) level explanations, graph-level explanation is temporarily suspended
            assert event_idx is not None
            # search
            self.mcts_state_map = MCTS(events=self.ori_subgraph_df,
                                       candidate_events=self.candidate_events,
                                       base_events=self.base_events,
                                       node_idx=node_idx, 
                                       event_idx=event_idx,
                                       n_rollout=self.rollout,
                                       min_atoms=self.min_atoms,
                                       c_puct=self.c_puct,
                                       score_func=self.tgnn_reward_wraper,
                                    #    device=self.device,
                                       candidate_initial_weights=self.candidate_initial_weights, # BUG: never pass through this parameter?????
                                    )
            
            if self.debug_mode:
                print('search graph:')
                print(self.ori_subgraph_df.to_string(max_rows=50))
                # print(f'{len(self.candidate_events)} candicate events:', self.mcts_state_map.candidate_events)
            tree_nodes = self.mcts_state_map.mcts(verbose=self.verbose) # search

        else: raise NotImplementedError('Wrong explanaion level')

        tree_node_x = find_best_node_result(tree_nodes, self.min_atoms)
        tree_nodes = sorted(tree_nodes, key=lambda x:x.P)

        if self.debug_mode:
            print_nodes(tree_nodes)

        return tree_nodes, tree_node_x
    
    @staticmethod
    def _path_suffix(pg_explainer_model, pg_positive):
        if pg_explainer_model is not None:
            suffix = 'pg_true'
        else:
            suffix = 'pg_false'
        
        if pg_explainer_model is not None:
            if pg_positive is True:
                suffix += '_pg_positive'
            else:
                suffix += '_pg_negative'
        
        return suffix
        

    @staticmethod
    def _mcts_recorder_path(result_dir, model_name, dataset_name, event_idx, suffix):
        if suffix is not None:
            record_filename = result_dir/f'{model_name}_{dataset_name}_{event_idx}_mcts_recorder_{suffix}.csv'
        else:
            record_filename = result_dir/f'{model_name}_{dataset_name}_{event_idx}_mcts_recorder.csv'
        
        return record_filename
    
    @staticmethod
    def _mcts_node_info_path(node_info_dir, model_name, dataset_name, event_idx, suffix):
        if suffix is not None:
            nodeinfo_filename = Path(node_info_dir)/f"{model_name}_{dataset_name}_{event_idx}_mcts_node_info_{suffix}.pt"
        else:
            nodeinfo_filename = Path(node_info_dir)/f"{model_name}_{dataset_name}_{event_idx}_mcts_node_info.pt"

        return nodeinfo_filename

    def _save_mcts_recorder(self, event_idx):
        # save records
        recorder_df = pd.DataFrame(self.mcts_state_map.recorder)
        # ROOT_DIR.parent/'benchmarks'/'results'
        record_filename = self._mcts_recorder_path(self.results_dir, self.model_name, self.dataset_name, event_idx, suffix=self.suffix)
        recorder_df.to_csv(record_filename, index=False)

        print(f'mcts recorder saved at {str(record_filename)}')
    
    def _save_mcts_nodes_info(self, tree_nodes, event_idx):
        saved_contents = {
            'saved_MCTSInfo_list': self.write_from_MCTSNode_list(tree_nodes),
        }
        path = self._mcts_node_info_path(self.mcts_saved_dir, self.model_name, self.dataset_name, event_idx, suffix=self.suffix)
        torch.save(saved_contents, path)
        print(f'results saved at {path}')
    
    def _load_saved_nodes_info(self, event_idx):
        path = self._mcts_node_info_path(self.mcts_saved_dir, self.model_name, self.dataset_name, event_idx, suffix=self.suffix)
        assert os.path.isfile(path)
        saved_contents = torch.load(path)
        
        saved_MCTSInfo_list = saved_contents['saved_MCTSInfo_list']
        tree_nodes = self.read_from_MCTSInfo_list(saved_MCTSInfo_list)
        tree_node_x = find_best_node_result(tree_nodes, self.min_atoms)

        return tree_nodes, tree_node_x

    def _set_candidate_weights(self, event_idx):
        # save candidates' initial weights computed by the pg_explainer_tg
        from tgnnexplainer.xgraph.method.tg_score import _set_tgat_data
        from tgnnexplainer.xgraph.method.attn_explainer_tg import AttnExplainerTG

        candidate_events = self.candidate_events

        self.pg_explainer_model.eval() # mlp
        input_expl = _create_explainer_input(self.model, self.model_name, self.all_events, \
                    candidate_events=self.candidate_events, event_idx=event_idx, device=self.device)
        
        edge_weights = self.pg_explainer_model(input_expl) # compute importance scores
        # event_idx_scores = event_idx_scores.cpu().detach().numpy().flatten()


        ################### added to original model attention scores
        candidate_weights_dict = {'candidate_events': torch.tensor(self.candidate_events, dtype=torch.int64, device=self.device),
                                    'edge_weights': edge_weights,
                }
        src_idx_l, target_idx_l, cut_time_l = _set_tgat_data(self.all_events, event_idx)
        output = self.model.get_prob( src_idx_l, target_idx_l, cut_time_l, logit=True, candidate_weights_dict=candidate_weights_dict)
        e_idx_weight_dict = AttnExplainerTG._agg_attention(self.model, self.model_name)
        edge_weights = np.array([ e_idx_weight_dict[e_idx] for e_idx in candidate_events ])
        ################### added to original model attention scores

        if not self.pg_positive:
            edge_weights = -1 * edge_weights
        
        # import ipdb; ipdb.set_trace()

        # event_idx_scores = np.random.random(size=(len(event_idx_scores,))) # ??
        candidate_initial_weights = { candidate_events[i]: edge_weights[i] for i in range(len(candidate_events)) }
        self.candidate_initial_weights = candidate_initial_weights

    def _initialize(self, event_idx):
        super(SubgraphXTG, self)._initialize(event_idx)
        if self.pg_explainer_model is not None: # use pg model 
            self._set_candidate_weights(event_idx)


    def __call__(self, node_idxs: Union[int, None] = None, event_idxs: Union[int, None] = None, return_dict=None, device=None):
        """
        Args:
            node_idxs: the target node index to explain for node prediction tasks
            event_idxs: the target event index to explain for edge prediction tasks
        """
        self.model.eval()
        if device is not None:
            self._to_device(device)

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
                self._save_mcts_recorder(event_idx) # always store
                if self.save and not self.load_results: # sometimes store
                    self._save_mcts_nodes_info(tree_nodes, event_idx)
            
            result = [tree_nodes, tree_node_x]
            results_list.append(result)
            
            if return_dict is not None:
                return_dict[event_idx] = result

        return results_list
        # return tree_nodes, tree_node_x

    def _to_device(self, device):
        pass
        # if torch.cuda.is_available():
        #     device = torch.device('cuda', index=device)
        # else:
        #     device = torch.device('cpu')
        
        # self.device = device
        # self.model.device = device
        # self.model.to(device)

        # if self.model_name == 'tgat':
        #     self.model.node_raw_embed = self.model.node_raw_embed.to(device)
        #     self.model.edge_raw_embed = self.model.edge_raw_embed.to(device)
        #     pass
        # elif self.model_name == 'tgn':
        #     self.model.node_raw_features = self.model.node_raw_features.to(device)
        #     self.model.edge_raw_features = self.model.edge_raw_features.to(device)

        # import ipdb; ipdb.set_trace()


