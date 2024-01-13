import pandas as pd
import itertools
import numpy as np
from sklearn import metrics
from sklearn.metrics import auc

from dig import ROOT_DIR
from dig.xgraph.evaluation.metrics_tg import BaseEvaluator
from dig.xgraph.dataset.tg_dataset import load_explain_idx
from dig.xgraph.models.ext import tgn

class Results():
    result_dir = ROOT_DIR.parent/'benchmarks'/'results'
    def __init__(self, model_name_list, explainer_name_list) -> None:
        self.model_name_list = model_name_list
        # self.dataset_name_list = dataset_name_list
        self.explainer_name_list = explainer_name_list
        # self.event_idxs_list = event_idxs_list
        # assert len(event_idxs_list) == len(dataset_name_list)
        
        # self.result_dir = ROOT_DIR.parent/'benchmarks'/'results'
        # self.event_idx_dir = ROOT_DIR/'xgraph'/'dataset'/'explain_index'

    # def _filename_collector(self, type: str, event_idx_file):
    #     files = []
    #     if type == 'fidelity_sparsity':
    #         pass

    #     for e_idx in event_idxs:
    #         pass

    #     pass

    def _init_result_dict(self,):
        result_dict = {}
        for model_name in self.model_name_list:
            result_dict[model_name] = {}
            for explainer_name in self.explainer_name_list:
                result_dict[model_name][explainer_name] = None
        return result_dict
    
    @staticmethod
    def _compute_metric(df, metric):
        groupby = df.groupby(by='event_idx')
        value_list = []
        for event_idx in groupby.groups: # different event_idxs
            sub_df = groupby.get_group(event_idx)
            fidelity_best = sub_df['fid_inv_best'].values
            sparsity = sub_df['sparsity'].values
            # import ipdb; ipdb.set_trace()

            if metric == 'auc':
                value = auc(sparsity, fidelity_best)
            elif metric == 'fid_best':
                value = fidelity_best.max()
            else: raise NotImplementedError

            # value = fidelity_best.iloc[10]
            # value = sparsity.iloc[10]
            # value = sparsity[ np.where(fidelity_best == fidelity_best.max())[0].min() ]
            
            if sparsity.max() > 1:
                import ipdb; ipdb.set_trace()

            value_list.append(value)

        return np.mean(value_list), np.var(value_list)

    def compute_metric(self, dataset_name, event_idxs, suffix_dict, metric='auc'):
        result_dict = self._init_result_dict()

        for model_name, explainer_name in itertools.product(self.model_name_list, self.explainer_name_list):
            suffix = suffix_dict.get(explainer_name, None)
            filename = BaseEvaluator._save_path(self.result_dir, model_name, dataset_name, explainer_name, event_idxs, suffix)
            df = pd.read_csv(filename)

            mean, var = self._compute_metric(df, metric) # over all event idxs
            result_dict[model_name][explainer_name] = f'{mean:.3f}({var:.3f})'
        
        result_df = pd.DataFrame(result_dict)
        print(f'metric: {metric}')
        print(result_df.to_string())
    

if __name__ == '__main__':
    model_name_list = ['tgat']
    explainer_name_list = ['subgraphx_tg', 'attn_explainer_tg', 'pg_explainer_tg', 'pbone_explainer_tg']
    
    dataset_name = 'simulate_v1' # wikipedia, reddit, simulate_v1, simulate_v2   
    explain_idx_file = 'simulate_v1'
    
    suffix_dict = {
        'subgraphx_tg': 'pg_true_pg_positive',
    }

    explain_index_dir = ROOT_DIR/'xgraph'/'dataset'/'explain_index'
    
    file = explain_index_dir/f'{explain_idx_file}.csv'
    event_idxs = load_explain_idx(file)


    resulter = Results(model_name_list, explainer_name_list)
    metric = 'fid_best' # auc, fid_best
    resulter.compute_metric(dataset_name, event_idxs, suffix_dict=suffix_dict, metric=metric)


