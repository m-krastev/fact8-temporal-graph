import numpy as np
from tgnnexplainer.xgraph.method.base_explainer_tg import BaseExplainerTG
from tgnnexplainer.xgraph.method.other_baselines_tg import PGExplainerExt
from tgnnexplainer.xgraph.method.other_baselines_tg import _create_explainer_input
from tgnnexplainer.xgraph.method.tg_score import _set_tgat_data
from pandas import DataFrame
from copy import deepcopy

class MLPNavigator():
    def __init__(self,
                 model,
                 model_name: str,
                 explainer_name: str,
                 dataset_name: str,
                 all_events: DataFrame,
                 explanation_level: str,
                 device,
                 verbose: bool = True,
                 results_dir = None,
                 debug_mode=True,
                 train_epochs: int = 50,
                 explainer_ckpt_dir = None,
                 reg_coefs = None,
                 batch_size = 64,
                 lr=1e-4):
        """
            Create a PgExplainerExt instance and call expose_explainer_model on it.
            Return the exposed explainer model, which is the MLP.
            
            Note: the exposed explainer model is a torch.nn.Sequential object.
            Note: the decision whether to load the parameters or to pre-train the explainer
                is made in the constructor of the PGExplainerExt class.
        """
        self.model = model
        self.model_name = model_name
        self.device = device
        self.all_events = all_events
        # handles loading/training the MLP
        pg_explainer = PGExplainerExt(model=model,
                                    model_name=model_name,
                                    explainer_name=explainer_name,
                                    dataset_name=dataset_name,
                                    all_events=all_events,
                                    explanation_level=explanation_level,
                                    device=device,
                                    verbose=verbose,
                                    results_dir=results_dir,
                                    debug_mode=debug_mode,
                                    train_epochs=train_epochs,
                                    explainer_ckpt_dir=explainer_ckpt_dir,
                                    reg_coefs=reg_coefs,
                                    batch_size=batch_size,
                                    lr=lr
                                    )
        # deepcopy the exposed explainer model, we will discard the PGExplainerExt instance
        self.mlp = deepcopy(pg_explainer.expose_explainer_model())

    def __call__(self, candidate_event_idx, target_idx):
        """
            Construct input for the pre-trained navigator (MLP)
            Call the navigator (MLP) on the input
            Return the edge weights

            Note: the input consists of pair-wise concatenation 
                of the target event and the candidate events.
        """
        # ensure evaluation mode
        self.mlp.eval()
        input_expl = _create_explainer_input(self.model, self.model_name, self.all_events,
                    candidate_events=candidate_event_idx, event_idx=target_idx, device=self.device)

        # compute importance scores
        edge_weights = self.mlp(input_expl)

        return edge_weights


class DotProductNavigator():
    """
        Compute importance scores by taking the dot product
        between the target event and the candidate events.
    """
    def __init__(self,
                 model,
                 model_name,
                 device,
                 all_events):
        self.model = model
        self.model_name = model_name
        self.device = device
        self.all_events = all_events
        
    def __call__(self, candidate_event_idx, target_idx):
        """
            Evaluate the target model on the candidate events.
            Obtain output embeddings at the end of the encoder module of the target model.
            Compute dot product between the output embeddings of the target event and the candidate events
            
            Note: the input consists of pair-wise concatenation 
                of the target event and the candidate events.
        """
        # split the input into source, destionations and cut time (time stamp of target)
        src, dst, cut_time = _set_tgat_data(self.all_events, target_idx)
        # the length of all of the above should be equal to the id of the target event
        assert len(src) == target_idx and len(dst) == target_idx and len(cut_time) == target_idx
        # obtain output embeddings of the target model
        if self.model_name == 'tgn':
            src_embed, dst_embed, _ = self.model.compute_temporal_embeddings(
                source_nodes=src,
                destination_nodes=dst,
                negative_nodes=np.array([0]), # like in TGN.get_prob
                edge_times=cut_time,
                edge_idxs=None, # this is fine (accoring to comments made in TGN.get_prob)
                n_neighbors=self.model.num_neighbors,
                edge_idx_preserve_list=None, # this would mask out edges when looking for the neighbourhood
                candidate_weights_dict=None  # this is what we are trying to compute here
            )
        elif self.model_name == 'tgat':
            src_embed = self.model.tem_conv(src, cut_time, self.model.num_layers)
            dst_embed = self.model.tem_conv(dst, cut_time, self.model.num_layers)
        # concatenate source and destination embeddings for each event
        embed = np.concatenate((src_embed, dst_embed), axis=1)
        # compute dot product between the target event and the candidate events
        dot_product = np.dot(embed, embed[target_idx])
        # we can also normalize the dot product, but it may not matter much since these scores are just used for sorting the candidates

        # return the scores for all inputs. The selection of candidates is done elsewhere.
        return dot_product
