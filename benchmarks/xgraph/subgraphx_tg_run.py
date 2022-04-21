import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from dig.xgraph.dataset.tg_dataset import load_tg_dataset
from dig.xgraph.dataset.utils_dataset import construct_tgat_model_data
from dig.xgraph.method.subgraphx_tg import SubgraphXTG
from dig.xgraph.models.ext.tgat.module import TGAN

from dig import ROOT_DIR


@hydra.main(config_path='config', config_name='config')
def pipeline(config: DictConfig):
    # set configurations
    # model config
    # explainer config
    config.models.param = config.models.param[config.datasets.dataset_name]
    config.explainers.param = config.explainers.param[config.datasets.dataset_name]
    config.datasets.dataset_path = str(ROOT_DIR/'xgraph'/'dataset'/'data'/f'{config.datasets.dataset_name}.csv')
    config.models.ckpt_path = str(ROOT_DIR/'xgraph'/'models'/'checkpoints'/f'{config.models.model_name}_{config.datasets.dataset_name}_best.pth')
    print(OmegaConf.to_yaml(config))

    # import ipdb; ipdb.set_trace()

    if torch.cuda.is_available():
        device = torch.device('cuda', index=config.device_id)
    else:
        device = torch.device('cpu')

    events, n_users, n_items = load_tg_dataset(config.datasets.dataset_path, 
                                                        target_model=config.models.model_name, dataset_params=config.datasets)


    if config.models.model_name == 'tgat':
        ngh_finder, node_feats, edge_feats = construct_tgat_model_data(events, config.datasets)
        model = TGAN(ngh_finder, node_feats, edge_feats, num_layers=config.models.param.num_layers, 
                    use_time=config.models.param.use_time, agg_method=config.models.param.agg_method, 
                    attn_mode=config.models.param.attn_mode, seq_len=config.models.param.seq_len, 
                    n_head=config.models.param.num_heads, drop_out=config.models.param.dropout)
    else: raise NotImplementedError('To do.')

    state_dict = torch.load(config.models.ckpt_path)
    model.load_state_dict(state_dict)
    model.to(device)

    explainer = SubgraphXTG(model, config.models.model_name, events, config.explainers.param.explanation_level, device=device,
                            rollout=60
    )


    # run the explainer
    # tree_results, receptive_idxs, candidates_idxs, removed_idxs = explainer.explain(event_idx=9909)
    # target_event_idx = 9909
    # target_event_idx = 19
    # target_event_idx = 29
    target_event_idx = config.explainers.param.target_event_idx
    tree_results, ori_event_idxs, candidates_idxs, important_events = explainer(event_idx=target_event_idx)

    # evaluation
    from dig.xgraph.evaluation.metrics_tg import ExplanationProcessorTG
    import json
    sparsity = 0.0
    evaluator = ExplanationProcessorTG(explainer.tgnn_reward_wraper, explainer.mcts_state_map, sparsity, target_event_idx)
    evaluation_dict = evaluator.evaluate()
    print(json.dumps(evaluation_dict, indent=4))
    
    print('candidate event idxs: ', candidates_idxs)
    print('searched important event idxs: ', important_events)



if __name__ == '__main__':
    pipeline()


