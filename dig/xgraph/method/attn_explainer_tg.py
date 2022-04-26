from typing import Optional, Union
from unittest import result
from pandas import DataFrame
import numpy as np

from dig.xgraph.method.subgraphx_tg import BaseExplainerTG
from dig.xgraph.method.tg_score import TGNNRewardWraper
from dig.xgraph.dataset.utils_dataset import k_hop_temporal_subgraph


class AttnExplainerTG(BaseExplainerTG):
    def __init__(self, model, model_name: str, explainer_name: str, dataset_name: str, all_events: DataFrame,  explanation_level: str, device, verbose: bool = True, result_dir = None,
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
        
        self._set_ori_subgraph(num_hops=3, event_idx=event_idx)
        self._set_candidate_events(event_idx)
        self._set_tgnn_wraper(event_idx)
        

        # compute attention weights
        events_idxs = self.ori_subgraph_df.index.values.tolist()
        score = self.tgnn_reward_wraper._compute_gnn_score(events_idxs, event_idx)

        # aggregate attention weights
        # import ipdb; ipdb.set_trace()
        atten_weights_list = self.model.atten_weights_list
        e_idx_weight_dict = self._agg_attention(atten_weights_list)
        # e_idx_weight_list = [ (e_idx, w) for e_idx, w in sorted(e_idx_weight_dict.items(), key=lambda item:item[1], reverse=True) ] # descending order

        return e_idx_weight_dict

    def __call__(self, node_idx: Union[int, None] = None, event_idx: Union[int, None] = None):

        results = self.explain(node_idx=node_idx, event_idx=event_idx)
        # self._save_value_results(event_idx=event_idx, value_results=results)
        e_idx_weight_list = []
        for e_idx in self.candidate_events:
            e_idx_weight_list.append((e_idx, results[e_idx]))
        
        e_idx_weight_list = sorted(e_idx_weight_list, key=lambda x: x[1], reverse=True) # descending

        return e_idx_weight_list
        