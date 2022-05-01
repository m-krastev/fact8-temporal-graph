from typing import Optional, Union
from unittest import result
from pandas import DataFrame
import numpy as np

from dig.xgraph.method.subgraphx_tg import BaseExplainerTG
from dig.xgraph.method.tg_score import TGNNRewardWraper
from dig.xgraph.dataset.utils_dataset import k_hop_temporal_subgraph


class AttnExplainerTG(BaseExplainerTG):
    def __init__(self, model, model_name: str, explainer_name: str, dataset_name: str, 
                 all_events: DataFrame,  explanation_level: str, device, verbose: bool = True, result_dir = None,
                ):
        super(AttnExplainerTG, self).__init__(model=model,
                                              model_name=model_name,
                                              explainer_name=explainer_name,
                                              dataset_name=dataset_name,
                                              all_events=all_events,
                                              explanation_level=explanation_level,
                                              device=device,
                                              verbose=verbose,
                                              results_dir=result_dir
                                              )
        # assert model_name in ['tgat', 'tgn']
        

    def _agg_attention(self, atten_weights_list):
        e_idx_weight_dict = {}
        for item in atten_weights_list:
            src_ngh_eidx = item['src_ngh_eidx']
            weights = item['attn_weight'].mean(dim=0)

            src_ngh_eidx = src_ngh_eidx.detach().cpu().numpy().flatten()
            weights = weights.detach().cpu().numpy().flatten()

            for e_idx, w in zip(src_ngh_eidx, weights):
                if e_idx_weight_dict.get(e_idx, None) is None:
                    e_idx_weight_dict[e_idx] = [w,]
                else:
                    e_idx_weight_dict[e_idx].append(w)

        for e_idx in e_idx_weight_dict.keys():
            e_idx_weight_dict[e_idx] = np.mean(e_idx_weight_dict[e_idx])

        return e_idx_weight_dict


    def explain(self, node_idx=None, event_idx=None):

        # compute attention weights
        events_idxs = self.ori_subgraph_df.index.values.tolist()
        score = self.tgnn_reward_wraper._compute_gnn_score(events_idxs, event_idx)

        # aggregate attention weights
        atten_weights_list = self.model.atten_weights_list
        e_idx_weight_dict = self._agg_attention(atten_weights_list)

        # TODO: note here is only for tgat!!!!
        # TODO: the whole explain function may need to be altered to support other models, e.g., tgn
        
        new_e_idx_weight_dict = { key-1: e_idx_weight_dict[key] for key in e_idx_weight_dict.keys() } # NOTE: important, the keys in e_idx_weight_dict has been added 1 for tgat model.

        # import ipdb; ipdb.set_trace()
        candidate_weights = { e_idx: new_e_idx_weight_dict[e_idx] for e_idx in self.candidate_events }
        candidate_weights = dict( sorted(candidate_weights.items(), key=lambda x: x[1], reverse=True) ) # NOTE: descending, important


        return candidate_weights

    def __call__(self, node_idxs: Union[int, None] = None, event_idxs: Union[int, None] = None):
        results_list = []
        for i, event_idx in enumerate(event_idxs):
            print(f'\nexplain {i}-th: {event_idx}')
            self._initialize(event_idx)

            candidate_weights = self.explain(event_idx=event_idx)
            
            results_list.append( [ list(candidate_weights.keys()), list(candidate_weights.values()) ] )
        
        return results_list
        