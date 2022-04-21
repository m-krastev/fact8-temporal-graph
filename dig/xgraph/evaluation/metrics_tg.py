from typing import List
import numpy as np
from sklearn import tree
from tqdm import tqdm

from dig.xgraph.method.subgraphx_tg import MCTS, MCTSNode
from dig.xgraph.method.tg_score import TGNNRewardWraper

def fidility_tg(ori_probs, unimportant_probs):
    """
    unimportant_probs: prediction with only searched unimportant events
    Generally the larger the better.
    """
    res = ori_probs - unimportant_probs
    return res

def fidility_tg_inv(ori_probs, important_probs):
    """
    important_probs: prediction with only searched important events
    Generally the smaller the better.
    """
    res = ori_probs - important_probs
    return res

def sparsity_tg(tree_node: MCTSNode, mcts_object: MCTS):
    candidate_events_set_ = set(mcts_object.candidate_events)
    # important_events = list(filter(lambda x: x in candidate_events_set_, tree_node.coalition))
    important_events = tree_node.coalition
    
    return 1.0 - len(important_events) / len(candidate_events_set_)



class ExplanationProcessorTG():
    def __init__(self, tgnn_reward_wraper: TGNNRewardWraper, mcts_state_map: MCTS, sparsity: float, target_event_idx: int) -> None:
        self.tgnn_reward_wraper = tgnn_reward_wraper
        self.mcts_state_map = mcts_state_map
        self.sparsity = sparsity
        self.ori_pred = self.tgnn_reward_wraper.original_scores
        self.target_event_idx = target_event_idx
    
    def evaluate(self):
        sparsity_list = []
        fid_inv_list = []
        fid_list = []

        print('evaluating...')
        for tree_node in tqdm(self.mcts_state_map.state_map.values(), total=len(self.mcts_state_map.state_map)) :
            spar = sparsity_tg(tree_node, self.mcts_state_map)
            if spar >= self.sparsity:
                base_and_important_events = self.mcts_state_map.obtain_base_and_important_events(tree_node)
                important_pred = self.tgnn_reward_wraper._compute_gnn_score(base_and_important_events, self.target_event_idx)
                base_and_unimportant_events = self.mcts_state_map.obtain_base_and_unimportant_events(tree_node)
                unimportant_pred = self.tgnn_reward_wraper._compute_gnn_score(base_and_unimportant_events, self.target_event_idx)

                fid_inv = fidility_tg_inv(self.ori_pred, important_pred)
                fid = fidility_tg(self.ori_pred, unimportant_pred)

                fid_inv_list.append(fid_inv)
                fid_list.append(fid)

                sparsity_list.append(spar)

        fid_inv_best = min(fid_inv_list)
        fid_best = max(fid_list)

        result_dict = {
            'sparsity threshold': self.sparsity,
            'sparsity avg': np.mean(sparsity_list),
            'fidility- avg': np.mean(fid_inv_list),
            'fidility+ avg': np.mean(fid_list),
            'fidility- best': fid_inv_best,
            'fidility+ best': fid_best,
            
        }

        return result_dict

