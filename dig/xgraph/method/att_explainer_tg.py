from typing import Optional
from pandas import DataFrame

class AttnExplainerTG(object):
    def __init__(self, model, model_name: str, dataset_name: str, all_events: DataFrame,  explanation_level: str, device, num_hops: Optional[int] = None, verbose: bool = True,
                 explain_graph: bool = True, rollout: int = 20, min_atoms: int = 1, c_puct: float = 150.0,
                 expand_atoms=14, local_radius=4, sample_num=100,
                 load_results=False, save_dir: Optional[str] = None, save_results: bool= True, save_filename: str = None,
                 filename: str = 'example', vis: bool = True):
        pass

    def __call__(self, *args, **kwds) -> None:
        
        pass
        