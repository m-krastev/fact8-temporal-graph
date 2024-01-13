import numpy as np
import pandas as pd
import argparse

from torch import positive

from tgnnexplainer import ROOT_DIR

def check_wiki_reddit_dataformat(df):
    assert df.iloc[:, 0].min() == 0
    assert df.iloc[:, 0].max() + 1 == df.iloc[:, 0].nunique() # 0, 1, 2, ...
    assert df.iloc[:, 1].min() == 0
    assert df.iloc[:, 1].max() + 1 == df.iloc[:, 1].nunique() # 0, 1, 2, ...
    
    for col in ['u', 'i', 'ts', 'label']:
        assert col in df.columns.to_list()


def verify_dataframe_unify(df):
    for col in ['u', 'i', 'ts', 'label', 'e_idx', 'idx']:
        assert col in df.columns.to_list()
    
    assert df.iloc[:, 0].min() == 1
    assert df.iloc[:, 0].max() == df.iloc[:, 0].nunique()
    assert df.iloc[:, 1].min() == df.iloc[:, 0].max() + 1
    assert df.iloc[:, 1].max() == df.iloc[:, 0].max() + df.iloc[:, 1].nunique()
    assert df['e_idx'].min() == 1
    assert df['e_idx'].max() == len(df)
    assert df['idx'].min() == 1
    assert df['idx'].max() == len(df)

    
def load_events_data(path):
    df = pd.read_csv(path)
    verify_dataframe_unify(df)
    return df

def load_tg_dataset(dataset_name):
    data_dir = ROOT_DIR/'xgraph'/'models'/'ext'/'tgat'/'processed'
    df = pd.read_csv(data_dir/f'ml_{dataset_name}.csv')
    edge_feats = np.load(data_dir/f'ml_{dataset_name}.npy')
    node_feats = np.load(data_dir/f'ml_{dataset_name}_node.npy')

    # df['e_idx'] = df.idx.values
    
    verify_dataframe_unify(df)

    assert df.i.max() + 1 == len(node_feats)
    assert df.e_idx.max() + 1 == len(edge_feats)

    # print
    n_users = df.iloc[:, 0].max()
    n_items = df.iloc[:, 1].max() - df.iloc[:, 0].max()
    print(f"#Dataset: {dataset_name}, #Users: {n_users}, #Items: {n_items}, #Interactions: {len(df)}, #Timestamps: {df.ts.nunique()}")
    print(f'#node feats shape: {node_feats.shape}, #edge feats shape: {edge_feats.shape}')
    
    return df, edge_feats, node_feats


def load_explain_idx(explain_idx_filepath, start=0, end=None):
    df = pd.read_csv(explain_idx_filepath)
    event_idxs = df['event_idx'].to_list()
    if end is not None:
        event_idxs = event_idxs[start:end]
    else: event_idxs = event_idxs[start:]
    
    print(f'{len(event_idxs)} events to explain')

    return event_idxs



def generate_explain_index(file, explainer_idx_dir, dataset_name, explain_idx_name=None):
    df = pd.read_csv(file)
    verify_dataframe_unify(df)
    
    size = 500 # 100, 200, 300, 400, 500

    if dataset_name in ['simulate_v1', 'simulate_v2']:
        indices = df.label == 1
        # indices = (df.label == 1) | (df.label == 0)
        explain_idxs = np.random.choice(df[indices].e_idx.values, size=size, replace=False)
        # import ipdb; ipdb.set_trace()
    elif dataset_name in ['wikipedia', 'reddit']:
        np.random.seed(1024)
        e_num = len(df)
        start_ratio = 0.7
        end_ratio = 0.99
        low = int(e_num*start_ratio)
        high = int(e_num*end_ratio)
        explain_idxs = np.random.randint(low=low, high=high, size=size)

    ############## save
    explain_idxs = sorted(explain_idxs)
    explain_idxs_dict = {
        'event_idx': explain_idxs, 
    }
    explain_idxs_df = pd.DataFrame(explain_idxs_dict)
    if explain_idx_name is None:
        out_file = explainer_idx_dir/f'{dataset_name}.csv'
    else:
        out_file = explainer_idx_dir/f'{explain_idx_name}.csv'
    
    explain_idxs_df.to_csv(out_file, index=False)
    print(f'explain index file {str(out_file)} saved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='wikipedia')
    parser.add_argument('-c', type=str, choices=['format', 'index'])
    args = parser.parse_args()

    data_dir = ROOT_DIR/'xgraph'/'models'/'ext'/'tgat'/'processed'
    explainer_idx_dir = ROOT_DIR/'xgraph'/'dataset'/'explain_index'
    file = data_dir/f'ml_{args.d}.csv'

    if args.c == 'format':
        pass
    elif args.c == 'index':
        generate_explain_index(file, explainer_idx_dir, args.d)
    else:
        raise NotImplementedError



