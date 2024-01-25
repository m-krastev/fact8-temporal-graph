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
