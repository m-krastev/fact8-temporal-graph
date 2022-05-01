from platform import node
from matplotlib import use
import numpy as np
import pandas as pd
import argparse

from path import Path



# def preprocess(data_name):
#     u_list, i_list, ts_list, label_list = [], [], [], []
#     feat_l = []
#     idx_list = []
    
#     with open(data_name) as f:
#         s = next(f)
#         print(s)
#         for idx, line in enumerate(f):
#             e = line.strip().split(',')
#             u = int(e[0])
#             i = int(e[1])

#             ts = float(e[2])
#             label = int(e[3])
            
#             feat = np.array([float(x) for x in e[4:]])
            
#             u_list.append(u)
#             i_list.append(i)
#             ts_list.append(ts)
#             label_list.append(label)
#             idx_list.append(idx)
            
#             feat_l.append(feat)
#     return pd.DataFrame({'u': u_list, 
#                          'i':i_list, 
#                          'ts':ts_list, 
#                          'label':label_list, 
#                          'idx':idx_list}), np.array(feat_l)

# def simulate_dataset_train_flag(df, ui_list=None):
#     """
#     select some event types as training indices
#     (u, i): (0, 0), (0, 1) for training
    
#     """
#     # import ipdb; ipdb.set_trace()
#     ui_list = [(0, 0), (0, 1)]
#     num_users = df['u'].nunique()

#     used_flag = np.zeros((len(df),)).astype(bool)
#     for (u, i) in ui_list:
#         u, i = reindex_nodes(u, i, num_users)
#         mask = (df['u'] == u) & (df['i'] == i)
#         used_flag = (used_flag | mask)

#     used_flag = used_flag.to_numpy()
#     return used_flag

def simulate_dataset_train_flag(df):
    labels = df['label'].to_numpy()
    mask = (labels == 1) | (labels == 0)
    return mask

# def reindex_nodes(u, i, num_users):
#     u = u + 1
#     i = i + 1 + num_users
#     return u, i

def reindex(df: pd.DataFrame):
    # assert(df.u.max() - df.u.min() + 1 == len(df.u.unique())) # u names are continue
    # assert(df.i.max() - df.i.min() + 1 == len(df.i.unique())) # i names are continue
    new_df = df.copy()

    num_u = df.u.max() + 1 #! number of users
    new_df.i = df.i + num_u

    new_df.u += 1 # user index starts as 1, 2, ..., k
    new_df.i += 1 # item index starts as k+1, k+2, ...
    
    new_df.idx += 1 # NOTE: should change
    
    print('number of users: ', new_df.u.max())
    print('number of items: ', new_df.i.max() - new_df.u.max())
    print('number of users+items: ', new_df.i.max())
    return new_df

def check_df(df):
    for label in ['u', 'i', 'ts', 'label']:
        assert label in df.columns.to_list()
    assert df['u'].min() == 0
    assert df['i'].min() == 0
    assert df['u'].max() + 1 == df['u'].nunique() # users: 0, 1, ...
    assert df['i'].max() + 1 == df['i'].nunique() # items: 0, 1, ...
    assert df.index.values.min() == 0
    assert df.index.values.max() + 1 == len(df)
    print('input data format ok')

def verify_df(df):
    """
    after reindexing
    """
    # assert df.columns.to_list() == ['u', 'i', 'ts', 'label', 'idx']
    for label in ['u', 'i', 'ts', 'label', 'idx']:
        assert label in df.columns.to_list()
    assert df['idx'].min() == 1
    assert df['idx'].max() == len(df)
    assert df['u'].min() == 1
    assert df['u'].max() == df['u'].nunique()
    assert df['i'].min() == df['u'].max() + 1
    assert df['i'].max() - df['i'].min() + 1 == df['i'].nunique()
    print('verified data format ok')

def rename_columns_wiki_reddit():
    from dig import ROOT_DIR
    data_dir = ROOT_DIR/'xgraph'/'dataset'/'data'
    data_names = ['wikipedia', 'reddit']

    for data_name in data_names:
        data_path = data_dir/f'{data_name}.csv'
        df = pd.read_csv(data_path, skiprows=1, header=None)
        feat_nums = df.shape[1] - 4
        new_columns = ['u', 'i', 'ts', 'label']
        for i in range(feat_nums):
            new_columns.append( f'f{i}' )
        
        rename_dict = {i: new_columns[i] for i in range(len(new_columns))}
        df.rename(columns=rename_dict, inplace=True)
        df.to_csv(data_path, index=False)

def run(data_name, out_dir=None):
    from dig import ROOT_DIR
    data_dir = ROOT_DIR/'xgraph'/'dataset'/'data'
    data_path = data_dir/f'{data_name}.csv'

    # PATH = './processed/{}.csv'.format(data_name)
    if out_dir is None:
        out_dir = Path('./processed/')
    else:
        out_dir = Path(out_dir)

    OUT_DF = out_dir/'ml_{}.csv'.format(data_name)
    OUT_EDGE_FEAT = out_dir/'ml_{}.npy'.format(data_name)
    OUT_NODE_FEAT = out_dir/'ml_{}_node.npy'.format(data_name)
    
    df = pd.read_csv(data_path)

    check_df(df)
    # import ipdb; ipdb.set_trace()

    df['idx'] = df.index.values
    new_df = reindex(df)

    verify_df(new_df)

    # set edge feature and node feature
    if data_name == 'simulate_v2':
        feature_dim = 64
        num_nodes = new_df.i.max()
        node_feat = []
        for i in range(num_nodes):
            if i < num_nodes//2:
                if i % 2 == 0:
                    node_feat.append([1, 0, 0, 0]) # user 0
                else: 
                    node_feat.append([0, 1, 0, 0]) # user 1
            else:
                if i % 2 == 0:
                    node_feat.append([0, 0, 1, 0]) # item 0
                else: 
                    node_feat.append([0, 0, 0, 1]) # item 1
        node_feat = np.array(node_feat)
        node_feat = np.hstack([node_feat, np.zeros((node_feat.shape[0], feature_dim-node_feat.shape[1]))])
        assert node_feat.shape == (num_nodes, feature_dim)
        node_feat = np.vstack([np.zeros((1, feature_dim)), node_feat])
        edge_feat = np.zeros((len(df) + 1, feature_dim))

    elif data_name == 'wikipedia' or data_name == 'reddit':
        select_columns = [c for c in new_df.columns if 'f' in c] # features
        edge_feat = new_df[select_columns].to_numpy()
        edge_feat_dim = edge_feat.shape[1]
        num_nodes = new_df.i.max()
        node_feat = np.zeros((num_nodes + 1, edge_feat_dim))
    
    # elif data_name == 'simulate': # v1
    else: # 'simulate', 'garden_5
        feature_dim = 64
        num_nodes = new_df.i.max()
        node_feat = np.zeros((num_nodes + 1, feature_dim)) # node feature, all zeros, the 0-th is not used
        edge_feat = np.zeros((len(df) + 1, feature_dim)) #! here set as 64, the 0-th is not used
    # else: raise NotImplementedError


    print('dataset: ', data_name)
    print('edge feature shape: ', edge_feat.shape)
    print('node feature shape: ', node_feat.shape)
    new_df[['u', 'i', 'ts', 'label', 'idx']].to_csv(OUT_DF, index=False)
    np.save(OUT_EDGE_FEAT, edge_feat) # edge feature matrix
    np.save(OUT_NODE_FEAT, node_feat) # node feature matrix
    print(f'{OUT_DF} saved')
    print(f'{OUT_EDGE_FEAT} saved')
    print(f'{OUT_NODE_FEAT} saved')


def process_garden_5():
    from dig import ROOT_DIR
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

    # process_garden_5()
    if args.rename_w_r:
        rename_columns_wiki_reddit()
        exit()

    run(dataset)