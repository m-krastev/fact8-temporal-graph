from ctypes import Union
from typing import List
import numpy as np
from pandas import DataFrame
from tqdm import tqdm
from pathlib import Path
from dig.xgraph.method.attn_explainer_tg import AttnExplainerTG

from dig.xgraph.method.subgraphx_tg import MCTS, BaseExplainerTG, MCTSNode, SubgraphXTG, base_and_important_events, base_and_unimportant_events
from dig.xgraph.method.tg_score import TGNNRewardWraper


# def fidelity_tg(ori_probs, unimportant_probs):
#     """
#     unimportant_probs: prediction with only searched unimportant events
#     Generally the larger the better.
#     """
#     res = ori_probs - unimportant_probs
#     return res

def fidelity_inv_tg(ori_probs, important_probs):
    """
    important_probs: prediction with only searched important events
    Generally the smaller the better.
    """
    if ori_probs >= 0.5: # logit
        # res = ori_probs - important_probs
        res = important_probs - ori_probs
    else: res = ori_probs - important_probs

    return res

def sparsity_tg(tree_node: MCTSNode, candidate_number: int):
    # return 1.0 - len(tree_node.coalition) / candidate_number
    return len(tree_node.coalition) / candidate_number

class BaseEvaluator():
    def __init__(self, model_name: str, explainer_name: str, dataset_name: str, 
                # ori_subgraph_df: DataFrame = None, candidate_events: List = None, tgnn_reward_wraper: TGNNRewardWraper = None,
                # target_event_idx: int = None, 
                explainer: BaseExplainerTG = None,
                results_dir=None
    ) -> None:
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.explainer_name = explainer_name

        self.explainer = explainer

        self.results_dir = results_dir


    def _save_value_results(self, event_idxs, value_results, suffix=None):
        """save to a csv for plotting"""
        if isinstance(event_idxs, int):
            if suffix is not None:
                filename = Path(self.results_dir)/f'{self.model_name}_{self.dataset_name}_{event_idxs}_{self.explainer_name}_{suffix}.csv'
            else:
                filename = Path(self.results_dir)/f'{self.model_name}_{self.dataset_name}_{event_idxs}_{self.explainer_name}.csv'
        elif isinstance(event_idxs, list):
            if suffix is not None:
                filename = Path(self.results_dir)/f'{self.model_name}_{self.dataset_name}_{event_idxs[0]}_to_{event_idxs[-1]}_{self.explainer_name}_{suffix}.csv'
            else:
                filename = Path(self.results_dir)/f'{self.model_name}_{self.dataset_name}_{event_idxs[0]}_to_{event_idxs[-1]}_{self.explainer_name}.csv'
        else: raise ValueError

        df = DataFrame(value_results)
        df.to_csv(filename, index=False)
        
        print(f'value results saved at {str(filename)}')

    def _evaluate_one(self, single_results, event_idx):
        raise NotImplementedError
        
    
    def evaluate(self, explainer_results, event_idxs):
        event_idxs_results = []
        sparsity_results = []
        fid_inv_results = []
        fid_inv_best_results = []

        print('\nevaluating...')
        for i, (single_results, event_idx) in enumerate(zip(explainer_results, event_idxs)):
            print(f'\nevaluate {i}th: {event_idx}')
            self.explainer._initialize(event_idx)

            sparsity_list, fid_inv_list, fid_inv_best_list =  self._evaluate_one(single_results, event_idx)

            # import ipdb; ipdb.set_trace()
            event_idxs_results.extend([event_idx]*len(sparsity_list))
            sparsity_results.extend(sparsity_list)
            fid_inv_results.extend(fid_inv_list)
            fid_inv_best_results.extend(fid_inv_best_list)
        
        results = {
            'event_idx': event_idxs_results,
            'sparsity': sparsity_results,
            'fid_inv': fid_inv_results,
            'fid_inv_best': fid_inv_best_results,
            
        }

        self._save_value_results(event_idxs, results)
        return results



class EvaluatorAttenTG(BaseEvaluator):
    def __init__(self, model_name: str, explainer_name: str, dataset_name: str,
                explainer: AttnExplainerTG,
                results_dir=None,
        ) -> None:
        super(EvaluatorAttenTG, self).__init__(model_name=model_name,
                                              explainer_name=explainer_name,
                                              dataset_name=dataset_name,
                                              results_dir=results_dir
                                              )
        self.explainer = explainer



    
    # SOLVED: why 0 in the first row of results csv? sparsity calculation is wrong
    def _evaluate_one(self, single_results, event_idx):
        candidates, candidate_weights = single_results

        sparsity = np.arange(0, 10.5, step=0.5) * 0.1
        candidate_events = self.explainer.candidate_events
        candidate_num = len(candidate_events)
        assert len(candidates) == candidate_num

        fid_inv_list = []
        for spar in sparsity:
            num = int(spar * candidate_num)
            important_events = candidates[:num+1]

            b_i_events = self.explainer.base_events + important_events
            important_pred = self.explainer.tgnn_reward_wraper._compute_gnn_score(b_i_events, event_idx)
            ori_pred = self.explainer.tgnn_reward_wraper.original_scores
            fid_inv = fidelity_inv_tg(ori_pred, important_pred)
            fid_inv_list.append(fid_inv)
            
        # import ipdb; ipdb.set_trace()
        fid_inv_best = array_best(fid_inv_list)

        return sparsity, fid_inv_list, fid_inv_best


def array_best(values):
    if len(values) == 0:
        return values
    best_values = [values[0], ]
    best = values[0]
    for i in range(1, len(values)):
        if best < values[i]:
            best = values[i]
        best_values.append(best)
    return np.array(best_values)

class EvaluatorMCTSTG(BaseEvaluator):
    def __init__(self, 
        model_name: str, explainer_name: str, dataset_name: str, 
        explainer: SubgraphXTG,
        # tgnn_reward_wraper: TGNNRewardWraper,
        # base_events = None, candidate_events = None,
        # sparsity: float, target_event_idx: int,
        # mcts_state_map: MCTS = None 
        results_dir = None
        ) -> None:
        super(EvaluatorMCTSTG, self).__init__(model_name=model_name,
                                              explainer_name=explainer_name,
                                              dataset_name=dataset_name,
                                              results_dir=results_dir
                                              )
        self.explainer = explainer
       
    
    def _evaluate_one(self, single_results, event_idx):
        
        tree_nodes, _ = single_results
        sparsity_list = []
        fid_inv_list = []
        
        candidate_events = self.explainer.candidate_events
        candidate_num = len(candidate_events)
        for node in tqdm(tree_nodes, total=len(tree_nodes)):
            # import ipdb; ipdb.set_trace()
            spar = sparsity_tg(node, candidate_num)

            b_i_events = self.explainer.base_events + node.coalition
            important_pred = self.explainer.tgnn_reward_wraper._compute_gnn_score(b_i_events, event_idx)
            # important_pred = node.P
            
            ori_pred = self.explainer.tgnn_reward_wraper.original_scores
            fid_inv = fidelity_inv_tg(ori_pred, important_pred) # the larger the better
            
            fid_inv_list.append(fid_inv)
            sparsity_list.append(spar)
        
        sparsity_list = np.array(sparsity_list)
        fid_inv_list = np.array(fid_inv_list)
        
        # sort according to sparsity
        sort_idx = np.argsort(sparsity_list) # ascending of sparsity
        sparsity_list = sparsity_list[sort_idx]
        fid_inv_list = fid_inv_list[sort_idx]
        fid_inv_best = array_best(fid_inv_list)

        # import ipdb; ipdb.set_trace()
        # only preserve a subset of results
        indices = np.arange(0, len(sparsity_list)+1, 5)
        indices = np.append(indices, len(sparsity_list)-1)
        sparsity_list = sparsity_list[indices]
        fid_inv_list = fid_inv_list[indices]
        fid_inv_best = fid_inv_best[indices]

        return sparsity_list, fid_inv_list, fid_inv_best

    # def evaluate(self, explainer_results: List, event_idxs: List):
    #     """
    #     """
    #     sparsity_results = []
    #     fid_inv_results = []
    #     fid_inv_best_results = []
    #     # fid_list = []
    #     # score_list = []

    #     print('evaluating...')
    #     # import ipdb; ipdb.set_trace()
    #     for i, ((tree_nodes, _), event_idx) in enumerate(zip(explainer_results, event_idxs)):
    #         print(f'evaluate {i}th: {event_idx}')
    #         self.explainer._initialize(event_idx)

    #         base_events = self.explainer.base_events
    #         candidate_events = self.explainer.candidate_events
    #         candidate_num = len(candidate_events)

    #         sparsity_list = []
    #         fid_inv_list = []
    #         for node in tqdm(tree_nodes, total=len(tree_nodes)):
    #             # import ipdb; ipdb.set_trace()
    #             spar = sparsity_tg(node, candidate_num)
    #             # b_i_events = base_and_important_events(base_events, candidate_events, node.coalition)
    #             # b_ui_events = base_and_unimportant_events(base_events, candidate_events, node.coalition)
    #             # important_pred = self.explainer.tgnn_reward_wraper._compute_gnn_score(b_i_events, event_idx)
    #             # unimportant_pred = self.explainer.tgnn_reward_wraper._compute_gnn_score(b_ui_events, event_idx)
    #             important_pred = node.P
    #             ori_pred = self.explainer.tgnn_reward_wraper.original_scores
    #             fid_inv = fidelity_inv_tg(ori_pred, important_pred) # if positive sample, the smaller the better; if negative sample, the larger the better
    #             # fid = fidility_tg(self.ori_pred, unimportant_pred) # if positive sampler, the larger the better; if negative sample, the smaller the better
                
    #             fid_inv_list.append(fid_inv)
    #             # fid_list.append(fid)

    #             # score_list.append( tree_node.Q() ) # TODO: difference?
    #             # score_list.append( tree_node.P )
    #             sparsity_list.append(spar)
            
    #         sparsity_list = np.array(sparsity_list)
    #         fid_inv_list = np.array(fid_inv_list)
            
    #         sort_idx = np.argsort(sparsity_list)
    #         sparsity_list = sparsity_list[sort_idx]
    #         fid_inv_list = fid_inv_list[sort_idx]
    #         fid_inv_best = array_best(fid_inv_list)

    #         # quantile indices, 0, 0.5, ..., 0.95, 1.0
    #         # quantiles = np.quantile(sparsity_list, np.arange(0, 1.05, 0.05))
    #         # indices = [(np.abs(sparsity_list - i)).argmax() for i in quantiles]
    #         indices = np.arange(0, len(sparsity_list)+1, 5)
    #         indices = np.append(indices, len(sparsity_list)-1)
    #         sparsity_list = sparsity_list[indices]
    #         fid_inv_list = fid_inv_list[indices]
    #         fid_inv_best = fid_inv_best[indices]


    #         sparsity_results.append(sparsity_list)
    #         fid_inv_results.append(fid_inv_list)
    #         fid_inv_best_results.append(fid_inv_best)



    #     result_dict = {
    #         'event_idx': [],
    #         'sparsity': [],
    #         'fid_inv': [],
    #         'fid_inv_best': []
    #     }

    #     for i, e_idx in enumerate(event_idxs):
    #         num = len(sparsity_results[i])
    #         result_dict['event_idx'].extend([e_idx]*num )
    #         result_dict['sparsity'].extend(sparsity_results[i])
    #         result_dict['fid_inv'].extend(fid_inv_results[i])
    #         result_dict['fid_inv_best'].extend(fid_inv_best_results[i])
        
    #     self._save_value_results(event_idxs, result_dict)

    #     return result_dict

