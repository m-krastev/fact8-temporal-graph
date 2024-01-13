from platform import node
from matplotlib import use
import numpy as np
import pandas as pd
import argparse

from pathlib import Path

def simulate_dataset_train_flag(df):
    labels = df['label'].to_numpy()
    mask = (labels == 1) | (labels == 0)
    return mask

def rename_columns_wiki_reddit(file):
    
    df = pd.read_csv(file, skiprows=1, header=None)
    feat_nums = df.shape[1] - 4
    new_columns = ['u', 'i', 'ts', 'label']
    
    for i in range(feat_nums):
        new_columns.append( f'f{i}' )
    
    rename_dict = {i: new_columns[i] for i in range(len(new_columns))}
    df.rename(columns=rename_dict, inplace=True)
    df.to_csv(file, index=False)
    print(f'rename the columns of {file}.')

def reindex(df):
    df['i'] += df['u'].max() + 1
    df['u'] += 1
    df['i'] += 1
    df['e_idx'] = df.index.values + 1
    df['idx'] = df.e_idx
    return df

def run(data_name, out_dir=None):
    from tgnnexplainer import ROOT_DIR
    from tgnnexplainer.xgraph.dataset.tg_dataset import verify_dataframe_unify, check_wiki_reddit_dataformat

    data_dir = ROOT_DIR/'xgraph'/'dataset'/'data'
    data_path = data_dir/f'{data_name}.csv'

    # PATH = './processed/{}.csv'.format(data_name)
    if out_dir is None:
        out_dir = Path('./processed/')
    else:
        out_dir = Path(out_dir)

    out_dir.mkdir(exist_ok=True)
    OUT_DF = out_dir/'ml_{}.csv'.format(data_name)
    OUT_EDGE_FEAT = out_dir/'ml_{}.npy'.format(data_name)
    OUT_NODE_FEAT = out_dir/'ml_{}_node.npy'.format(data_name)
    
    df = pd.read_csv(data_path)

    # the very raw format
    if 'comma_separated_list_of_features' in df.columns.tolist():
        rename_columns_wiki_reddit(data_path)
        df = pd.read_csv(data_path)

    check_wiki_reddit_dataformat(df)

    # import ipdb; ipdb.set_trace()
    df = reindex(df)
    verify_dataframe_unify(df)
    
    new_df = df

    # set edge feature and node feature
    if data_name == 'simulate_v2':
        raise NotImplementedError
    elif data_name == 'simulate_v1':
        raise NotImplementedError

    elif data_name == 'wikipedia' or data_name == 'reddit':
        select_columns = [c for c in new_df.columns if 'f' in c] # features
        edge_feat = np.zeros((len(df) + 1, len(select_columns))) # 0-th pad with 0
        edge_feat[1:, :] = new_df[select_columns].to_numpy()

        edge_feat_dim = edge_feat.shape[1]
        num_nodes = new_df.i.max()
        node_feat = np.zeros((num_nodes + 1, edge_feat_dim))

    else: 
        raise NotImplementedError

    assert len(node_feat) == new_df.i.max() + 1
    assert len(edge_feat) == len(new_df) + 1

    print('dataset: ', data_name)
    print('edge feature shape: ', edge_feat.shape)
    print('node feature shape: ', node_feat.shape)
    new_df[['u', 'i', 'ts', 'label', 'idx', 'e_idx']].to_csv(OUT_DF, index=False)
    np.save(OUT_EDGE_FEAT, edge_feat) # edge feature matrix
    np.save(OUT_NODE_FEAT, node_feat) # node feature matrix
    print(f'{OUT_DF} saved')
    print(f'{OUT_EDGE_FEAT} saved')
    print(f'{OUT_NODE_FEAT} saved')


def process_garden_5():
    from tgnnexplainer import ROOT_DIR
    data_dir = ROOT_DIR/'xgraph'/'dataset'/'data'
    data_path = data_dir/'garden_5.csv'
    df = pd.read_csv(data_path)
    if 'label' not in df.columns.to_list():
        df['label'] = np.ones((len(df),))
        df.to_csv(data_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, default='simulate')
    parser.add_argument('-rename_w_r', action='store_true', help='rename columns of wikipedia and reddit')
    args = parser.parse_args()
    dataset = args.data

    run(dataset)