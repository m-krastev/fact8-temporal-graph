import os
import pandas as pd


def check_dir(save_dirs):
    if save_dirs:
        if os.path.isdir(save_dirs):
            pass
        else:
            os.makedirs(save_dirs)

def load_explain_idx(explain_idx_filepath):
    df = pd.read_csv(explain_idx_filepath)
    event_idxs = df['event_idx'].to_list()
    return event_idxs