from time import time
import numpy as np
from typing import Union
from pandas import DataFrame

from tgnnexplainer.xgraph.method.base_explainer_tg import BaseExplainerTG


class AttnExplainerTG(BaseExplainerTG):
    def __init__(self,
                 model,
                 model_name: str,
                 explainer_name: str,
                 dataset_name: str,
                 all_events: DataFrame,
                 explanation_level: str,
                 device,
                 verbose: bool = True,
                 results_dir = None,
                 debug_mode=True,
                 threshold_num=25):
        super(AttnExplainerTG, self).__init__(model=model,
                                              model_name=model_name,
                                              explainer_name=explainer_name,
                                              dataset_name=dataset_name,
                                              all_events=all_events,
                                              explanation_level=explanation_level,
                                              device=device,
                                              verbose=verbose,
                                              results_dir=results_dir,
                                              debug_mode=debug_mode,
                                              threshold_num=threshold_num)
        # assert model_name in ['tgat', 'tgn']

    @staticmethod
    def _agg_attention(model, model_name):
        # after a forward computation in the model

        # aggregate attention weights
        if model_name == 'tgat':
            atten_weights_list = model.atten_weights_list
        elif model_name == 'tgn':
            atten_weights_list = model.embedding_module.atten_weights_list


        e_idx_weight_dict = {}
        for item in atten_weights_list:
            if model_name == 'tgat':
                edge_idxs = item['src_ngh_eidx']
                weights = item['attn_weight'].mean(dim=0) # a special process
            elif model_name == 'tgn':
                edge_idxs = item['src_ngh_eidx']
                weights = item['attn_weight']

            edge_idxs = edge_idxs.detach().cpu().numpy().flatten()
            weights = weights.detach().cpu().numpy().flatten()

            for e_idx, w in zip(edge_idxs, weights):
                if e_idx_weight_dict.get(e_idx, None) is None:
                    e_idx_weight_dict[e_idx] = [w,]
                else:
                    e_idx_weight_dict[e_idx].append(w)

        for e_idx in e_idx_weight_dict.keys():
            e_idx_weight_dict[e_idx] = np.mean(e_idx_weight_dict[e_idx])

        return e_idx_weight_dict


    def explain(self, node_idx=None, event_idx=None):
        # compute attention weights
        # events_idxs = self.ori_subgraph_df.index.values.tolist()
        events_idxs = self.ori_subgraph_df.e_idx.values
        score = self.tgnn_reward_wraper._compute_gnn_score(events_idxs, event_idx) # NOTE: required.

        e_idx_weight_dict = self._agg_attention(self.model, self.model_name)

        # import ipdb; ipdb.set_trace()
        candidate_weights = { e_idx: e_idx_weight_dict[e_idx] for e_idx in self.candidate_events }
        candidate_weights = dict( sorted(candidate_weights.items(), key=lambda x: x[1], reverse=True) ) # NOTE: descending, important

        # import ipdb; ipdb.set_trace()

        return candidate_weights

    def __call__(self, node_idxs: Union[int, None] = None, event_idxs: Union[int, None] = None):
        results_list = []
        for i, event_idx in enumerate(event_idxs):
            print(f'\nexplain {i}-th: {event_idx}')
            self._initialize(event_idx)
            tick = time()
            
            candidate_weights = self.explain(event_idx=event_idx)
            
            tock = time() - tick
            results_list.append( [ list(candidate_weights.keys()), list(candidate_weights.values())] )
            self._save_candidate_scores(candidate_weights, event_idx, len(candidate_weights)* [tock])
        
        return results_list
