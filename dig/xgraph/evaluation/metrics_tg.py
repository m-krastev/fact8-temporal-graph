from typing import List
import numpy as np
from pandas import DataFrame
from tqdm import tqdm
from pathlib import Path

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

class BaseEvaluator():
    def __init__(self, model_name: str, explainer_name: str, dataset_name: str, 
                ori_subgraph_df: DataFrame, candidate_events: List, tgnn_reward_wraper: TGNNRewardWraper,
                target_event_idx: int, results_dir=None
    ) -> None:
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.explainer_name = explainer_name

        self.ori_subgraph_df = ori_subgraph_df
        self.candidate_events = candidate_events
        self.tgnn_reward_wraper = tgnn_reward_wraper
        self.ori_pred = self.tgnn_reward_wraper.original_scores

        self.target_event_idx = target_event_idx
        self.results_dir = results_dir



    def _save_value_results(self, event_idx, value_results, suffix=None):
        """save to a csv for plotting"""
        if suffix is not None:
            filename = Path(self.results_dir)/f'{self.model_name}_{self.dataset_name}_{event_idx}_{self.explainer_name}_{suffix}.csv'
        else:
            filename = Path(self.results_dir)/f'{self.model_name}_{self.dataset_name}_{event_idx}_{self.explainer_name}.csv'
        
        df = DataFrame(value_results)
        df.to_csv(filename, index=False)
        print(f'value results saved at {str(filename)}')



class EvaluatorAttenTG(BaseEvaluator):
    def __init__(self, model_name: str, explainer_name: str, dataset_name: str, 
                ori_subgraph_df: DataFrame, candidate_events: List, tgnn_reward_wraper: TGNNRewardWraper, 
                target_event_idx: int, results_dir=None,
        ) -> None:
        super(EvaluatorAttenTG, self).__init__(model_name, explainer_name, dataset_name, 
                                               ori_subgraph_df, candidate_events, tgnn_reward_wraper, target_event_idx, results_dir)

        self.base_events_ = None

    @property
    def base_events(self,):
        if self.base_events_ is None:
            candidate_events_ = set(self.candidate_events)
            self.base_events_ = [e_idx for e_idx in self.ori_subgraph_df.index.values if e_idx not in candidate_events_]
            return self.base_events_
        else:
            return self.base_events_

    def evaluate(self, e_idx_weight_list):
        """
        return the fidelity at each sparsity
        assume the `e_idx_weight_list` only has candidate e_idxs
        """
        assert len(self.candidate_events) == len(e_idx_weight_list)

        sparsity_list = np.arange(1, 11, step=1) * 0.1
        fid_inv_list = []
        # base_events = 
        for sparsity in sparsity_list:
            num = int(sparsity * len(self.candidate_events))
            important_events = [x[0] for x in e_idx_weight_list[:num+1] ] # (e_idx, weight)

            base_and_important_events = self.base_events + important_events
            important_pred = self.tgnn_reward_wraper._compute_gnn_score(base_and_important_events, self.target_event_idx)
            fid_inv = important_pred - self.ori_pred
            fid_inv_list.append(fid_inv)

        fid_best = [ max(fid_inv_list[:i+1]) for i in range(0, len(fid_inv_list)) ]
        results = {
            'sparsity': sparsity_list,
            'fid_inv_best': fid_best,
            'fid_inv_list': fid_inv_list,
        }

        self._save_value_results(self.target_event_idx, results)
        return results



class EvaluatorMCTSTG(BaseEvaluator):
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

