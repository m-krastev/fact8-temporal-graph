import numpy as np
import itertools
from pathlib import Path

from typing import Optional, Union

from pandas import DataFrame
from tgnnexplainer.xgraph.models.ext.tgat.module import TGAN
from tgnnexplainer.xgraph.method.tg_score import TGNNRewardWraper
from tgnnexplainer.xgraph.dataset.utils_dataset import k_hop_temporal_subgraph

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
        self.results_dir = Path(results_dir)
        self.debug_mode = debug_mode
        
        self.model.eval()
        self.model.to(self.device)

        # construct TGNN reward function
        self.tgnn_reward_wraper = TGNNRewardWraper(self.model, self.model_name, self.all_events, self.explanation_level)

    def find_candidates(self, target_event_idx):
        # TODO: implementation for other models
        # from tgnnexplainer.xgraph.dataset.utils_dataset import tgat_node_reindex
        # from tgnnexplainer.xgraph.method.tg_score import _set_tgat_events_idxs # NOTE: important

        if self.model_name in ['tgat', 'tgn']:
            ngh_finder = self.model.ngh_finder
            num_layers = self.model.num_layers
            num_neighbors = self.model.num_neighbors # NOTE: important
            edge_idx_preserve_list = self.ori_subgraph_df.e_idx.to_list() # NOTE: e_idx column

            u = self.all_events.iloc[target_event_idx-1, 0] # because target_event_idx should represent e_idx. e_idx = index + 1
            i = self.all_events.iloc[target_event_idx-1, 1]
            ts = self.all_events.iloc[target_event_idx-1, 2]

            # new_u, new_i = tgat_node_reindex(u, i, self.num_users)
            # accu_e_idx = [ [target_event_idx+1, target_event_idx+1]] # NOTE: for subsequent '-1' operation
            accu_e_idx = [ ] # NOTE: important?
            accu_node = [ [u, i,] ]
            accu_ts = [ [ts, ts,] ]
            
            for i in range(num_layers):
                last_nodes = accu_node[-1]
                last_ts = accu_ts[-1]
                # import ipdb; ipdb.set_trace()

                out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch = ngh_finder.get_temporal_neighbor(
                                                                                    last_nodes, 
                                                                                    last_ts, 
                                                                                    num_neighbors=num_neighbors,
                                                                                    edge_idx_preserve_list=edge_idx_preserve_list, # NOTE: not needed?
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
                # import ipdb; ipdb.set_trace()

            unique_e_idx = np.array(list(itertools.chain.from_iterable(accu_e_idx)))
            unique_e_idx = unique_e_idx[ unique_e_idx != 0 ] # NOTE: 0 are padded e_idxs
            # unique_e_idx = unique_e_idx - 1 # NOTE: -1, because ngh_finder stored +1 e_idxs
            unique_e_idx = np.unique(unique_e_idx).tolist()

            # TODO: to test self.base_events = unique_e_idx, will this influence the speed?

            
        else:
            raise NotImplementedError
        
        candidate_events = unique_e_idx
        threshold_num = 20
        if len(candidate_events) > threshold_num:
            candidate_events = candidate_events[-threshold_num:]
            candidate_events = sorted(candidate_events)
        # import ipdb; ipdb.set_trace()
        
        if self.debug_mode:
            print(f'{len(unique_e_idx)} seen events, used {len(candidate_events)} as candidates:')
            print(candidate_events)
        
        return candidate_events, unique_e_idx
    
    def _set_ori_subgraph(self, num_hops, event_idx):
        subgraph_df = k_hop_temporal_subgraph(self.all_events, num_hops=num_hops, event_idx=event_idx)
        self.ori_subgraph_df = subgraph_df


    def _set_candidate_events(self, event_idx):
        self.candidate_events, unique_e_idx = self.find_candidates(event_idx)
        # self.candidate_events = shuffle( candidate_events ) # strategy 1
        # self.candidate_events = candidate_events # strategy 2
        # self.candidate_events.reverse()
        # self.candidate_events = candidate_events # strategy 3
        candidate_events_set_ = set(self.candidate_events)
        assert hasattr(self, 'ori_subgraph_df')
        # self.base_events = list(filter(lambda x: x not in candidate_events_set_, self.ori_subgraph_df.e_idx.values) ) # NOTE: ori_subgraph_df.e_idx.values
        self.base_events = list(filter(lambda x: x not in candidate_events_set_, unique_e_idx) ) # NOTE: an importanct change, need test. largely influence the speed!



    def _set_tgnn_wraper(self, event_idx):
        assert hasattr(self, 'ori_subgraph_df')
        # self.tgnn_reward_wraper.compute_original_score(self.ori_subgraph_df.e_idx.values, event_idx)
        self.tgnn_reward_wraper.compute_original_score(self.base_events+self.candidate_events, event_idx)
    
    def _initialize(self, event_idx):
        self._set_ori_subgraph(num_hops=3, event_idx=event_idx)
        self._set_candidate_events(event_idx)
        self._set_tgnn_wraper(event_idx)
        # self.candidate_initial_weights = None
        np.random.seed(1)
        self.candidate_initial_weights = { e_idx: np.random.random() for e_idx in self.candidate_events }

        

    @staticmethod
    def _score_path(results_dir, model_name, dataset_name, explainer_name, event_idx,):
        """
        only for baseline explainer, save their computed candidate scores.
        """
        score_filename = results_dir/f'{model_name}_{dataset_name}_{explainer_name}_{event_idx}_candidate_scores.csv'
        return score_filename

    def _save_candidate_scores(self, candidate_weights, event_idx):
        """
        only for baseline explainer, save their computed candidate scores.
        """
        assert isinstance(candidate_weights, dict)
        filename = self._score_path(self.results_dir, self.model_name, self.dataset_name, self.explainer_name, event_idx)
        data_dict = {
            'candidates': [],
            'scores': []
        }
        for k, v in candidate_weights.items():
            data_dict['candidates'].append(k)
            data_dict['scores'].append(v)
        
        df = DataFrame(data_dict)
        df.to_csv(filename, index=False)
        print(f'candidate scores saved at {filename}')


