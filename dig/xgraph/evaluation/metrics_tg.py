from typing import List
import numpy as np
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

def sparsity_tg(tree_node: MCTSNode, candidate_number: int):
    # candidate_events = mcts_object.candidate_events
    # important_events = list(filter(lambda x: x in candidate_events_set_, tree_node.coalition))
    important_events = tree_node.coalition
    
    return 1.0 - len(important_events) / candidate_number



class ExplanationProcessorTG():
    def __init__(self, tgnn_reward_wraper: TGNNRewardWraper, tree_results: List, sparsity: float, target_event_idx: int,
        candidate_number: int = None, mcts_state_map: MCTS = None ) -> None:
        self.tgnn_reward_wraper = tgnn_reward_wraper
        self.mcts_state_map = mcts_state_map
        self.tree_results = tree_results
        self.candidate_number = candidate_number
        self.sparsity = sparsity
        self.ori_pred = self.tgnn_reward_wraper.original_scores
        self.target_event_idx = target_event_idx
    
    def evaluate(self):
        sparsity_list = []
        fid_inv_list = []
        fid_list = []
        score_list = []

        print('evaluating...')

        # for tree_node in tqdm(self.mcts_state_map.state_map.values(), total=len(self.mcts_state_map.state_map)):
        for tree_node in tqdm(self.tree_results, total=len(self.tree_results)) :
            # import ipdb; ipdb.set_trace()
            spar = sparsity_tg(tree_node, self.candidate_number)
            if spar >= self.sparsity:
                if self.mcts_state_map is not None:
                    base_and_important_events = self.mcts_state_map.obtain_base_and_important_events(tree_node)
                    important_pred = self.tgnn_reward_wraper._compute_gnn_score(base_and_important_events, self.target_event_idx)
                    base_and_unimportant_events = self.mcts_state_map.obtain_base_and_unimportant_events(tree_node)
                    unimportant_pred = self.tgnn_reward_wraper._compute_gnn_score(base_and_unimportant_events, self.target_event_idx)

                    fid_inv = fidility_tg_inv(self.ori_pred, important_pred) # if positive sample, the smaller the better; if negative sample, the larger the better
                    fid = fidility_tg(self.ori_pred, unimportant_pred) # if positive sampler, the larger the better; if negative sample, the smaller the better

                    fid_inv_list.append(fid_inv)
                    fid_list.append(fid)

                # score_list.append( tree_node.Q() ) # TODO: difference?
                score_list.append( tree_node.P )

                sparsity_list.append(spar)

        score_list = np.array(score_list)
        sparsity_list = np.array(sparsity_list)

        fid_inv_list = np.array(fid_inv_list)
        fid_list = np.array(fid_list)

        if self.mcts_state_map is not None:
            fid_inv_best = fid_inv_list.min() * -1 if self.ori_pred > 0 else fid_inv_list.max() # the larger the better
            fid_best = fid_list.max() * -1 if self.ori_pred > 0 else fid_list.min()  # the smaller the better
        else:
            fid_inv_best = None
            fid_best = None

        result_dict = {
            'sparsity threshold': self.sparsity,
            'sparsity avg': np.mean(sparsity_list),
            # 'fidility- avg': np.mean(fid_inv_list),
            # 'fidility+ avg': np.mean(fid_list),
            'fidility- best': fid_inv_best,
            'fidility+ best': fid_best,
            'avg score': np.mean(score_list),
            'best score': np.max(score_list),

            
        }

        return result_dict

