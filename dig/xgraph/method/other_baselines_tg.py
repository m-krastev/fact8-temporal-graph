

# TODO: another baseline?
# independent event score baseline


from typing import Optional, Union
from unittest import result
from matplotlib.pyplot import cla
from pandas import DataFrame
from path import Path
from sklearn.utils import shuffle
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from dig.xgraph.method.subgraphx_tg import BaseExplainerTG
from dig.xgraph.evaluation.metrics_tg import fidelity_inv_tg
from dig.xgraph.method.tg_score import _set_tgat_data, _set_tgat_events_idxs
from dig import ROOT_DIR

class PGExplainerExt(BaseExplainerTG):
    def __init__(self, model, model_name: str, explainer_name: str, dataset_name: str, 
                 all_events: DataFrame,  explanation_level: str, device, verbose: bool = True, result_dir = None,
                 # specific params for PGExplainerExt
                 train_epochs: int = 50, explainer_ckpt_dir = None, reg_coefs = None, batch_size = 64, lr=1e-4
                ):
        super(PGExplainerExt, self).__init__(model=model,
                                              model_name=model_name,
                                              explainer_name=explainer_name,
                                              dataset_name=dataset_name,
                                              all_events=all_events,
                                              explanation_level=explanation_level,
                                              device=device,
                                              verbose=verbose,
                                              results_dir=result_dir
                                              )
        self.train_epochs = train_epochs
        self.explainer_ckpt_dir = explainer_ckpt_dir
        self.reg_coefs = reg_coefs
        self.batch_size = batch_size
        self.lr = lr
        self.expl_input_dim = None
        self._init_explainer()
        

    def _init_explainer(self):
        if self.model_name == 'tgat':
            self.expl_input_dim = self.model.model_dim * 8 # 2 * (dim_u + dim_i + dim_t + dim_e)
        elif self.model_name == 'tgn':
            raise NotImplementedError

        self.explainer_model = nn.Sequential(
            nn.Linear(self.expl_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.explainer_model.to(self.device)
    
    def __call__(self, node_idxs: Union[int, None] = None, event_idxs: Union[int, None] = None):
        self.explainer_ckpt_path = Path(self.explainer_ckpt_dir)/f'{self.model_name}_{self.dataset_name}_{self.explainer_name}_expl_ckpt.pt'
        if not self.explainer_ckpt_path.exists():
            self._train() # we need to train the explainer first
        else:
            state_dict = torch.load(self.explainer_ckpt_path)
            self.explainer_model.load_state_dict(state_dict)

        results_list = []
        for i, event_idx in enumerate(event_idxs):
            print(f'\nexplain {i}-th: {event_idx}')
            self._initialize(event_idx)
            candidate_weights = self.explain(event_idx=event_idx)
            results_list.append( [ list(candidate_weights.keys()), list(candidate_weights.values()) ] )
        import ipdb; ipdb.set_trace()
        return results_list

    
    def _tg_predict(self, event_idx, use_explainer=False):
        if self.model_name == 'tgat':
            src_idx_l, target_idx_l, cut_time_l = _set_tgat_data(self.all_events, self.num_users, event_idx)
            edge_weights = None
            if use_explainer:
                candidate_events_new = _set_tgat_events_idxs(self.candidate_events) # these temporal edges to alter attn weights in tgat
                input_expl = self._create_explainer_input( self.candidate_events, event_idx ) # u, i, ts_emb, e_emb (target); u, i, ts_emb, e_emb (a candidate)
                # import ipdb; ipdb.set_trace()
                edge_weights = self.explainer_model(input_expl)
                candidate_weights_dict = {'candidate_events': torch.tensor(candidate_events_new, dtype=torch.int64, device=self.device),
                                    'edge_weights': edge_weights,
                }
            else:
                candidate_weights_dict = None
            # NOTE: use the 'src_ngh_eidx_batch' in module to locate mask fill positions
            output = self.model.get_prob( src_idx_l, target_idx_l, cut_time_l, logit=True, candidate_weights_dict=candidate_weights_dict)
        elif self.model_name == 'abc':
            raise NotImplementedError
        else: 
            raise NotImplementedError

        return output, edge_weights



    def _loss(self, masked_pred, original_pred, mask, reg_coefs):
        # TODO:
        size_reg = reg_coefs[0]
        entropy_reg = reg_coefs[1]

        # Regularization losses
        size_loss = torch.sum(mask) * size_reg
        # mask_ent_reg = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        # mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)

        # Explanation loss
        cce_loss = torch.nn.functional.mse_loss(masked_pred, original_pred)

        #return cce_loss
        # return cce_loss + size_loss + mask_ent_loss
        return cce_loss + size_loss

    def _train(self,):
        self.explainer_model.train()
        optimizer = torch.optim.Adam(self.explainer_model.parameters(), lr=self.lr)
        
        np.random.seed(1)
        train_e_idxs = np.random.randint(1, len(self.all_events)-1, (1000, )) # NOTE: set train event idxs
        for e in range(self.train_epochs):
            train_e_idxs = shuffle(train_e_idxs)
            optimizer.zero_grad()
            loss = torch.tensor([0], dtype=torch.float32, device=self.device)
            counter = 0

            for i, event_idx in tqdm(enumerate(train_e_idxs), total=len(train_e_idxs), desc=f'epoch {e}' ): # training
                self._initialize(event_idx) # NOTE: needed
                if len(self.candidate_events) == 0: # skip bad samples
                    continue

                original_pred, mask_values_ = self._tg_predict(event_idx, use_explainer=False)
                masked_pred, mask_values = self._tg_predict(event_idx, use_explainer=True)

                id_loss = self._loss(masked_pred, original_pred, mask_values, self.reg_coefs)
                loss += id_loss
                counter += 1

                if counter % self.batch_size == 0:
                    loss = loss/self.batch_size
                    loss.backward()
                    optimizer.step()
                    loss = torch.tensor([0], dtype=torch.float32, device=self.device)
                    optimizer.zero_grad()
        
            tqdm.write(f"epoch {e} loss epoch {loss.cpu().detach().item()}")
        # print("epoch %d loss %.6f acc: %.4f masked_acc: %.4f "%(e, loss/total_pred, right_pred/total_pred, right_pred_masked/total_pred ))

        # ckpt_path = ROOT_DIR/'xgraph'/'explainer_ckpts'/f'{self.model_name}_{self.dataset_name}_{self.explainer_name}_expl_ckpt.pt'
        state_dict = self.explainer_model.state_dict()
        torch.save(state_dict, self.explainer_ckpt_path)
        print('train finished')
        print(f'explainer ckpt saved at {str(self.explainer_ckpt_path)}')

    def _create_explainer_input(self, candidate_events=None, event_idx=None):
        # TODO: explainer input should have both the target event and the event that we want to assign a weight to.
        device = self.device

        if self.model_name == 'tgat':
            event_idx_u, event_idx_i, event_idx_t = _set_tgat_data(self.all_events, self.num_users, event_idx)
            event_idx_new = _set_tgat_events_idxs(event_idx)
            t_idx_u_emb = self.model.node_raw_embed( torch.tensor(event_idx_u, dtype=torch.int64, device=device) )
            t_idx_i_emb = self.model.node_raw_embed( torch.tensor(event_idx_i, dtype=torch.int64, device=device) )
            # import ipdb; ipdb.set_trace()
            t_idx_t_emb = self.model.time_encoder( torch.tensor(event_idx_t, dtype=torch.float32, device=device).reshape((1, -1)) ).reshape((1, -1))
            t_idx_e_emb = self.model.edge_raw_embed( torch.tensor([event_idx_new, ], dtype=torch.int64, device=device) )
            
            target_event_emb = torch.cat([t_idx_u_emb,t_idx_i_emb, t_idx_t_emb, t_idx_e_emb ], dim=1)
            
            candidate_events_u, candidate_events_i, candidate_events_t = _set_tgat_data(self.all_events, self.num_users, candidate_events)
            candidate_events_new = _set_tgat_events_idxs(self.candidate_events)

            candidate_u_emb = self.model.node_raw_embed( torch.tensor(candidate_events_u, dtype=torch.int64, device=device) )
            candidate_i_emb = self.model.node_raw_embed( torch.tensor(candidate_events_i, dtype=torch.int64, device=device) )
            candidate_t_emb = self.model.time_encoder( torch.tensor(candidate_events_t, dtype=torch.float32, device=device).reshape((1, -1)) ).reshape((len(candidate_events_t), -1))
            candidate_e_emb = self.model.edge_raw_embed( torch.tensor(candidate_events_new, dtype=torch.int64, device=device) )

            candiadte_events_emb = torch.cat([candidate_u_emb, candidate_i_emb, candidate_t_emb, candidate_e_emb], dim=1)

            input_expl = torch.cat([ target_event_emb.repeat(candiadte_events_emb.shape[0], 1),  candiadte_events_emb], dim=1)
            # import ipdb; ipdb.set_trace()

        elif self.model_name == 'abc':
            raise NotImplementedError
            
        return input_expl



    def explain(self, node_idx=None, event_idx=None):
        input_expl = self._create_explainer_input(candidate_events=self.candidate_events, event_idx=event_idx)
        event_idx_scores = self.explainer_model(input_expl) # compute importance scores
        event_idx_scores = event_idx_scores.cpu().detach().numpy().flatten()

        # the same as Attn explainer
        candidate_weights = { self.candidate_events[i]: event_idx_scores[i] for i in range(len(self.candidate_events)) }
        candidate_weights = dict( sorted(candidate_weights.items(), key=lambda x: x[1], reverse=True) ) # NOTE: descending, important

        return candidate_weights



class PBOneExplainerTG(BaseExplainerTG):
    """
    perturb only one event to evaluate its influence, then leverage the rank info.
    """
    def __init__(self, model, model_name: str, explainer_name: str, dataset_name: str, 
                 all_events: DataFrame,  explanation_level: str, device, verbose: bool = True, result_dir = None,
                ):
        super(PBOneExplainerTG, self).__init__(model=model,
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
        # perturb each one
        e_idx_logit_dict = {}
        base_events = self.base_events
        for e_idx in self.candidate_events:
            preserved = [idx for idx in self.candidate_events if idx != e_idx]
            total = base_events + preserved
            score = self.tgnn_reward_wraper._compute_gnn_score(total, event_idx)
            e_idx_logit_dict[e_idx] = score
        
        # import ipdb; ipdb.set_trace()
        # compute an importance score for ranking them
        e_idx_score_dict = {}
        ori_score = self.tgnn_reward_wraper.original_scores
        for e_idx in e_idx_logit_dict.keys():
            e_idx_score_dict[e_idx] = fidelity_inv_tg(ori_score, e_idx_logit_dict[e_idx])
        
        # the same as Attn explainer
        candidate_weights = { e_idx: e_idx_score_dict[e_idx] for e_idx in self.candidate_events }
        candidate_weights = dict( sorted(candidate_weights.items(), key=lambda x: x[1], reverse=True) ) # NOTE: descending, important
        
        return candidate_weights

    def __call__(self, node_idxs: Union[int, None] = None, event_idxs: Union[int, None] = None):
        results_list = []
        for i, event_idx in enumerate(event_idxs):
            print(f'\nexplain {i}-th: {event_idx}')
            self._initialize(event_idx)

            candidate_weights = self.explain(event_idx=event_idx)
            results_list.append( [ list(candidate_weights.keys()), list(candidate_weights.values()) ] )
            # import ipdb; ipdb.set_trace()
        
        return results_list
        