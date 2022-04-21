import numpy as np
import torch.nn as nn

from dig.xgraph.dataset.utils_dataset import tgat_node_reindex

class TGNNRewardWraper(object):
    def __init__(self, model, model_name, all_events, explanation_level):
        """
        """
        self.model = model
        self.model_name = model_name
        # self.events = events
        self.all_events = all_events
        self.n_users = all_events.iloc[:, 0].nunique()
        self.explanation_level = explanation_level

        self.gamma = 0.05
    
    def error(self, ori_pred, ptb_pred):

        pass

    
    def _set_tgat_data(self, target_event_idx):
        """ support for tgat """
        # set the self.model itself, e.g., update its neighbor finder, node/edge feature set, etc.
        # then construct model input data
        # sub_events = self.all_events.iloc[events_idxs, :]

        # construct graph, rename nodes
        # node features and edge features don't need to alter?


        # subgraph_df = self.all_events.iloc[events_idxs, :]
        # neig_finder, old_new_mapping = construct_tgat_neighbor_finder(subgraph_df, self.n_users)
        # self.model.ngh_finder = neig_finder
        # self.old_new_mapping = old_new_mapping
        # cut_time_l = [self.all_events.iloc[target_event_idx, 2], ]
        # src_idx_l = [old_new_mapping['user'][ self.all_events.iloc[target_event_idx, 0] ], ]
        # target_idx_l = [old_new_mapping['item'][ self.all_events.iloc[target_event_idx, 1] ], ]

        # src_idx_l = [self.all_events.iloc[target_event_idx, 0], ]
        # target_idx_l = [self.all_events.iloc[target_event_idx, 1], ]
        target_t = self.all_events.iloc[target_event_idx, 2]
        target_u, target_i = self.all_events.iloc[target_event_idx, 0], self.all_events.iloc[target_event_idx, 1]
        new_u, new_i = tgat_node_reindex(target_u, target_i, self.n_users)

        src_idx_l = np.array([new_u, ])
        target_idx_l = np.array([new_i, ])
        cut_time_l = np.array([target_t, ])

        input_data = [src_idx_l, target_idx_l, cut_time_l]
        return input_data

    def compute_original_score(self, events_idxs, target_event_idx):
        """
        `events_idx` acts as the original graph data (the root node of the MCTS).
        # Generally it is a 3-hop temporal subgraph of the whole temporal graph.
        Compute and store the original prediction of TGNNs.
        """
        # self.sub_events = sub_events
        if self.model_name == 'tgat':
            input_data = self._set_tgat_data(target_event_idx)
            scores = self.model.get_prob(*input_data, edge_idx_preserve_list=events_idxs, logit=True)
            self.original_scores = scores.item()
            self.orininal_size = len(events_idxs)
            print('original score: ', self.original_scores)
            # if self.explanation_level == 'event':
            #     self.original_prediction = None
            #     pass
        else:
            raise NotImplementedError
        

    def __call__(self, events_idxs, target_event_idx):
        # events_idxs is the events' indices searched by the MCTS, i.e., the coalition list of MC tree nodes.
        # only events in events_idx can be utilized by the tgat model, or other models

        # target_event_idx is the target edge that we want to compute a score by the temporal GNN model.

        if self.model_name == 'tgat':
            input_data = self._set_tgat_data(target_event_idx)
            scores = self.model.get_prob(*input_data, edge_idx_preserve_list=events_idxs, logit=True)
            # import ipdb; ipdb.set_trace()
            reward = self._compute_reward(scores, self.orininal_size-len(events_idxs))
            return reward

        elif self.model_name == 'abc':
            pass

        pass

    def _compute_gnn_score(self, events_idxs, target_event_idx):
        input_data = self._set_tgat_data(target_event_idx)
        scores = self.model.get_prob(*input_data, edge_idx_preserve_list=events_idxs, logit=True)
        scores = scores.item()
        return scores

    def _compute_reward(self, scores_petb, remove_size):
        """
        Reward should be the larger the better.
        """

        # import ipdb; ipdb.set_trace()
        # t1 = self.error(self.original_scores, scores_petb.item())
        if self.original_scores >= 0:
            t1 = (scores_petb - self.original_scores).item()
        else:
            t1 = (self.original_scores - scores_petb).item()
        
        t2 = remove_size
        # r = -1*t1 + -self.gamma * t2
        # r = -t1
        r = t1
        return r


