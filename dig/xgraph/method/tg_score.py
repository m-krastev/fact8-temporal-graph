from typing import Union
from typing import List
import numpy as np
from pandas import DataFrame
import torch.nn as nn

from dig.xgraph.dataset.utils_dataset import tgat_node_reindex


def _set_tgat_events_idxs(events_idxs):
    """ supporter for tgat """
    if isinstance(events_idxs, (int, np.int64)):
        return events_idxs + 1
    events_idxs_new = [ idx+1 for idx in events_idxs]
    return events_idxs_new 

def _set_tgat_data(all_events: DataFrame, n_users: int, target_event_idx: Union[int, List]):
    """ supporter for tgat """
    if isinstance(target_event_idx, (int, np.int64)):
        target_u = all_events.iloc[target_event_idx, 0]
        target_i = all_events.iloc[target_event_idx, 1]
        target_t = all_events.iloc[target_event_idx, 2]
        new_u, new_i = tgat_node_reindex(target_u, target_i, n_users)

        src_idx_l = np.array([new_u, ])
        target_idx_l = np.array([new_i, ])
        cut_time_l = np.array([target_t, ])
    elif isinstance(target_event_idx, list):
        target_u = all_events.iloc[target_event_idx, 0].to_numpy()
        target_i = all_events.iloc[target_event_idx, 1].to_numpy()
        target_t = all_events.iloc[target_event_idx, 2].to_numpy()
        new_u, new_i = tgat_node_reindex(target_u, target_i, n_users)

        src_idx_l = new_u
        target_idx_l = new_i
        cut_time_l = target_t
    else: 
        raise ValueError

    input_data = [src_idx_l, target_idx_l, cut_time_l]
    return input_data


class TGNNRewardWraper(object):
    def __init__(self, model, model_name, all_events, explanation_level):
        """
        """
        self.model = model
        self.model_name = model_name
        self.all_events = all_events
        self.n_users = all_events.iloc[:, 0].max() + 1
        self.explanation_level = explanation_level

        self.gamma = 0.05
    
    # def error(self, ori_pred, ptb_pred):

    #     pass


    
    def _get_model_prob(self, target_event_idx, seen_events_idxs):
        if self.model_name == 'tgat':
            input_data = _set_tgat_data(self.all_events, self.n_users, target_event_idx)
            seen_events_idxs = _set_tgat_events_idxs(seen_events_idxs) # NOTE: important
            score = self.model.get_prob(*input_data, edge_idx_preserve_list=seen_events_idxs, logit=True)
        else:
            raise NotImplementedError
        
        return score.item()


    def compute_original_score(self, events_idxs, target_event_idx):
        """
        events_idxs: could be seen by model
        """
        # self.sub_events = sub_events
        if self.model_name == 'tgat':
            self.original_scores = self._get_model_prob(target_event_idx, events_idxs)
            self.orininal_size = len(events_idxs)
            # print('original score: ', self.original_scores)
        else:
            raise NotImplementedError
        

    def __call__(self, events_idxs, target_event_idx):
        """
        events_idxs the all the events' indices could be seen by the gnn model.
        target_event_idx is the target edge that we want to compute a reward by the temporal GNN model.
        """

        if self.model_name == 'tgat':
            scores = self._get_model_prob(target_event_idx, events_idxs)
            # import ipdb; ipdb.set_trace()
            reward = self._compute_reward(scores, self.orininal_size-len(events_idxs))
            return reward

        elif self.model_name == 'abc':
            pass

        pass

    def _compute_gnn_score(self, events_idxs, target_event_idx):
        """
        events_idxs the all the events' indices could be seen by the gnn model. idxs in the all_events space, not in the tgat space.
        target_event_idx is the target edge that we want to compute a gnn score by the temporal GNN model.
        """
        return self._get_model_prob(target_event_idx, events_idxs)

        
    def _compute_reward(self, scores_petb, remove_size):
        """
        Reward should be the larger the better.
        """

        # import ipdb; ipdb.set_trace()
        # t1 = self.error(self.original_scores, scores_petb.item())
        if self.original_scores >= 0:
            t1 = scores_petb - self.original_scores
        else:
            t1 = self.original_scores - scores_petb
        
        t2 = remove_size
        # r = -1*t1 + -self.gamma * t2
        # r = -t1
        r = t1
        return r


