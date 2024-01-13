import argparse
import numpy as np
from tick.hawkes import SimuHawkesExpKernels
from itertools import product
import pandas as pd

from tgnnexplainer import ROOT_DIR
from tgnnexplainer.xgraph.dataset.tg_dataset import verify_dataframe_unify


RAW_DATA_DIR = ROOT_DIR/'xgraph'/'dataset'/'data'

def happen_rate(source_ts, target_ts, interval, reverse=False):
    if reverse:
        source_ts, target_ts = target_ts, source_ts
        source_ts = -1 * source_ts
        target_ts = -1 * target_ts


    positive = []
    for t in source_ts:
        t_start = t + 1e-4
        t_end = t + interval
        happened = np.where(np.logical_and(target_ts>=t_start, target_ts<=t_end))
        
        happened = happened[0]
        if len(happened) >= 1:
            positive.append(1)
        else:
            positive.append(0)
    return np.mean(positive)


def print_statistics(hawkes, source_type=2, target_type=3):
    intervals = np.arange(0.1, 3.1, 0.1)
    # [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    # source_types = [3]
    # target_types = [3,]
    source_types = [source_type, ]
    target_types = [target_type, ]

    print('source type --> ')
    
    statistics = {
        'interval': [],
        'real_happen_rate': [],
        'random_happen_rate': [],
    }
    for idx, (interval, source_type, target_type) in enumerate(product(intervals, source_types, target_types)):
        
        real_happen_rate = happen_rate(hawkes.timestamps[source_type], hawkes.timestamps[target_type], interval, reverse=True)
        
        random_ts = np.random.uniform( hawkes.timestamps[source_type].min(),  
                                        hawkes.timestamps[source_type].max(), 
                                        size=hawkes.timestamps[source_type].shape,
                                        )
        # random_happen_rate = happen_rate(random_ts, hawkes.timestamps[target_type], interval)
        random_happen_rate = happen_rate(hawkes.timestamps[source_type], random_ts, interval)
        

        print(f'source type {source_type} <-- target_type {target_type}, {real_happen_rate*100:.4f}% happened in interval {interval:.4f}')
        # print(f'source type R --> target_type {target_type}, {random_happen_rate*100:.4f}% happened in interval {interval:.4f}')
        print(f'source type {source_type} <-- target_type R, {random_happen_rate*100:.4f}% happened in interval {interval:.4f}')

        statistics['interval'].append(f'{interval:.1f}')
        statistics['real_happen_rate'].append(real_happen_rate)
        statistics['random_happen_rate'].append(random_happen_rate)

    return statistics

def save_data(df, node_feats, edge_feats, dataset_name):
    ####### output to target directories
    verify_dataframe_unify(df)
    assert len(node_feats) == df.i.max() + 1
    assert len(edge_feats) == len(df) + 1

    print('node feat shape:', node_feats.shape)
    print('edge feat shape:', edge_feats.shape)

    raw_data_dir = ROOT_DIR/'xgraph'/'dataset'/'data'
    processed_data_dir = ROOT_DIR/'xgraph'/'models'/'ext'/'tgat'/'processed'
    
    RAW_DF = raw_data_dir/f'{dataset_name}.csv'
    OUT_DF = processed_data_dir/'ml_{}.csv'.format(dataset_name)
    OUT_EDGE_FEAT = processed_data_dir/'ml_{}.npy'.format(dataset_name)
    OUT_NODE_FEAT = processed_data_dir/'ml_{}_node.npy'.format(dataset_name)

    df.to_csv(RAW_DF, index=False)
    df.to_csv(OUT_DF, index=False)
    np.save(OUT_EDGE_FEAT, edge_feats)
    np.save(OUT_NODE_FEAT, node_feats)
    print(f'{RAW_DF} saved')
    print(f'{OUT_DF} saved')
    print(f'{OUT_EDGE_FEAT} saved')
    print(f'{OUT_NODE_FEAT} saved')
    pass

def make_dataset(hawkes, type_edge_mapping, num_nodes, dataset_name):
    # target_type = 2
    target_type = 3

    data = {
        'u': [],
        'i': [],
        'ts': [],
        'label': [],
    }
    for type_i, timestamps in enumerate(hawkes.timestamps):
        u, i = type_edge_mapping[type_i]
        if type_i == target_type:
            label = 1 # positive
        else: label = -1
        
        for ts in timestamps:
            data['u'].append(u)
            data['i'].append(i)
            data['ts'].append(ts)
            data['label'].append(label)
        
    
    negative_times = np.random.uniform(low=hawkes.timestamps[target_type].min(), 
                                        high=hawkes.timestamps[target_type].max(), 
                                        size=(int(hawkes.timestamps[target_type].size), ))
    u, i = type_edge_mapping[target_type]
    # u, i = type_edge_mapping[1]
    for ts in negative_times:
        data['u'].append(u)
        data['i'].append(i)
        data['ts'].append(ts)
        data['label'].append(0) # nagetive
    
    df = pd.DataFrame(data)
    df = df.sort_values(by=['ts'], ignore_index=True)

    ####### idx and e_idx
    df['idx'] = df.index.values + 1
    df['e_idx'] = df.index.values + 1
    verify_dataframe_unify(df)

    # print(df.to_string(max_rows=100))
    ####### node and edge features
    dim = num_nodes
    node_feats = np.concatenate([np.zeros((1, dim)),  np.random.random(size=(num_nodes, dim))], axis=0)
    edge_feats = node_feats[df.u.values] + node_feats[df.i.values]
    edge_feats = np.concatenate([np.zeros((1, dim)),  edge_feats], axis=0)

    
    save_data(df, node_feats, edge_feats, dataset_name)
    # import ipdb; ipdb.set_trace()



def simulate_hawkes_v1():
    n_nodes = 4  # dimension of the Hawkes process
    type_edge_mapping = {
        0: (1, 3), # user, item
        1: (1, 4),
        2: (2, 3),
        3: (2, 4)
    }
    base_miu = [0.5, 0.5, 0, 0]

    A_alpha = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, -1, 0, 0],
        [0, 0, 2, 0]
    ], dtype=float)
    decay = 10
    seed = 2022


    hawkes = SimuHawkesExpKernels(adjacency=A_alpha, decays=decay,
                              baseline=base_miu,
                              force_simulation=False,
                              verbose=True, seed=seed)

    run_time = 5000
    hawkes.end_time = run_time
    dt = 0.5
    hawkes.threshold_negative_intensity(allow=True)
    # hawkes.track_intensity(dt)
    hawkes.simulate()

    print('length:')
    for i, timestamps in enumerate(hawkes.timestamps):
        print(i, len(timestamps), 'min:', min(timestamps), 'max:', max(timestamps),)
    
    total_timestamps = np.sum([ len(x) for x in hawkes.timestamps ])
    print('total timestamps:', total_timestamps)

    # intervals = np.arange(0.1, 3.1, 0.1)
    # for interv in intervals:
    #     print(f'interval {interv} average ts {(total_timestamps/run_time)*interv }')

    return hawkes, type_edge_mapping, n_nodes



def simulate_hawkes_v2():
    n_nodes = 4  # dimension of the Hawkes process
    type_edge_mapping = {
        0: (1, 3), # user, item
        1: (1, 4),
        2: (2, 3),
        3: (2, 4)
    }
    base_miu = [0.5, 0.5, 0, 0]

    A_alpha = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, -1, 0, 0],
        [0, 0, 2, -2]
    ], dtype=float)
    decay = 10
    seed = 2022


    hawkes = SimuHawkesExpKernels(adjacency=A_alpha, decays=decay,
                              baseline=base_miu,
                              force_simulation=False,
                              verbose=True, seed=seed)

    run_time = 10000
    hawkes.end_time = run_time
    dt = 0.5
    hawkes.threshold_negative_intensity(allow=True)
    # hawkes.track_intensity(dt)
    hawkes.simulate()

    print('length:')
    for i, timestamps in enumerate(hawkes.timestamps):
        print(i, len(timestamps), 'min:', min(timestamps), 'max:', max(timestamps),)
    
    total_timestamps = np.sum([ len(x) for x in hawkes.timestamps ])
    print('total timestamps:', total_timestamps)

    # intervals = np.arange(0.1, 3.1, 0.1)
    # for interv in intervals:
    #     print(f'interval {interv} average ts {(total_timestamps/run_time)*interv }')

    return hawkes, type_edge_mapping, n_nodes



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='simulate_v1', choices=['simulate_v1', 'simulate_v2'])
    args = parser.parse_args()
    np.random.seed(2022)

    dataset_name = 'simulate_v2' # simulate_v1, simulate_v2

    if dataset_name == 'simulate_v1':
        hawkes, type_edge_mapping, n_nodes = simulate_hawkes_v1()
        print_statistics(hawkes)
        make_dataset(hawkes, type_edge_mapping, n_nodes, dataset_name=dataset_name)
    elif dataset_name == 'simulate_v2':
        hawkes, type_edge_mapping, n_nodes = simulate_hawkes_v2()
        print_statistics(hawkes)
        make_dataset(hawkes, type_edge_mapping, n_nodes, dataset_name=dataset_name)








