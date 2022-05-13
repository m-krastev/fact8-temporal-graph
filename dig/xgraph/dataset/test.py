
from cgi import test

from dig.xgraph.dataset.utils_dataset import k_hop_temporal_subgraph
import pandas as pd

def test_k_hop():
    df = {
        'u': [1, 1, 1, 2],
        'i': [3, 4, 5, 5],
        'ts': [1, 2, 3, 4],
        'label': [0, 0, 0, 0],
        'e_idx': [1, 2, 3, 4]
        
    }
    df = pd.DataFrame(df)
    import ipdb; ipdb.set_trace()
    subgraph_df = k_hop_temporal_subgraph(df, num_hops=2, event_idx=4)
    print(subgraph_df)

if __name__ == '__main__':
    test_k_hop()