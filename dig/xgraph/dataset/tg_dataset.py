from tabnanny import check
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

def check_dataframe(df):
    assert df.iloc[:, 0].min() == 0
    assert df.iloc[:, 0].max() + 1 == df.iloc[:, 0].nunique() # 0, 1, 2, ...
    assert df.iloc[:, 1].min() == 0
    assert df.iloc[:, 1].max() + 1 == df.iloc[:, 1].nunique() # 0, 1, 2, ...

    if len( df.columns.to_list() ) == 3:
        assert df.columns.to_list() == ['u', 'i', 'ts']
        df['label'] = np.ones((len(df), ))
    elif len( df.columns.to_list() ) == 4:
        assert df.columns.to_list() == ['u', 'i', 'ts', 'label']

def verify_dataframe(df):
    assert df.iloc[:, 0].min() == 0
    assert df.iloc[:, 0].max() + 1 == df.iloc[:, 0].nunique()
    assert df.iloc[:, 1].min() == 0
    assert df.iloc[:, 1].max() + 1 == df.iloc[:, 1].nunique()
    for col in ['u', 'i', 'ts', 'label']:
        assert col in df.columns.to_list()


    
def load_events_data(path):
    df = pd.read_csv(path)
    check_dataframe(df)
    return df

# def construct_adj(df, adj, cols):
#     for grouo_id, group in df.groupby(by=[cols]):
#         neighbors = group["index"].values
#         count = len(neighbors)
#         for i in range(count):
#             for j in range(0, i):
#                 #adj[neighbors[i],neighbors[j]] = 1
#                 adj[neighbors[j],neighbors[i]] = 1

# def construct_line_graph(df):
#     df.reset_index(inplace=True)
#     n_event = len(df)
#     line_graph = np.zeros((n_event, n_event))
#     construct_adj(df, line_graph, "user") #连接相同user id的event
#     construct_adj(df, line_graph, "item") #连接相同item id的event
#     return line_graph

def _preprocess_tgat(df, node_feats, edge_feats):
    df_new = df.copy()
    df_new['u'] += 1
    df_new['i'] += 1
    df_new['i'] += df_new['u'].max()
    node_feats = np.vstack([np.zeros((1, node_feats.shape[1])), node_feats])
    edge_feats = np.vstack([np.zeros((1, edge_feats.shape[1])), edge_feats])

    return df_new, node_feats, edge_feats


def load_tg_dataset(dataset_path, dataset_params=None, target_model='tgat'):
    df = load_events_data(dataset_path)

    n_users = df.iloc[:, 0].max() + 1
    n_items = df.iloc[:, 1].max() + 1

    print(f"#Dataset: {dataset_params.dataset_name}, #Users: {n_users}, #Items: {n_items}, #Interactions: {len(df)}, #Timestamps: {df.ts.nunique()}")
    
    # normalize time, need this? # TODO: need to consider this
    # t_max = df.iloc[:, 2].max()
    # df.iloc[:,2] = df.iloc[:,2]/t_max
    
    # time index, need this?
    le = LabelEncoder()
    df["time_index"] = le.fit_transform(df["ts"])

    # if dataset_params.node_feat_zero is True:
    #     node_feats = np.zeros((n_users+n_items, dataset_params.node_feat_dim))
    # if dataset_params.edge_feat_zero is True:
    #     edge_feats = np.zeros((len(df), dataset_params.edge_feat_dim))
    

    # if target_model == 'tgat': # TODO: need to implement this in other places
    #     node_feats, edge_feats = feature_process_tgat(node_feats, edge_feats)

    verify_dataframe(df)
    return df, n_users, n_items


