import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from dig.xgraph.dataset.tg_dataset import load_tg_dataset
from dig.xgraph.dataset.utils_dataset import construct_tgat_model_data
from dig.xgraph.method.subgraphx_tg import SubgraphXTG
from dig.xgraph.method.attn_explainer_tg import AttnExplainerTG
from dig.xgraph.models.ext.tgat.module import TGAN

from dig import ROOT_DIR


@hydra.main(config_path='config', config_name='config')
def pipeline(config: DictConfig):
    # set configurations
    # model config
    # explainer config
    
    config.models.param = config.models.param[config.datasets.dataset_name]
    config.datasets.dataset_path = str(ROOT_DIR/'xgraph'/'dataset'/'data'/f'{config.datasets.dataset_name}.csv')
    config.models.ckpt_path = str(ROOT_DIR/'xgraph'/'models'/'checkpoints'/f'{config.models.model_name}_{config.datasets.dataset_name}_best.pth')
    config.explainers.param = config.explainers.param[config.datasets.dataset_name]
    
    config.explainers.results_dir = str(ROOT_DIR.parent/'benchmarks'/'results')
    config.explainers.mcts_saved_dir = str(ROOT_DIR/'xgraph'/'saved_mcts_results')
    config.explainers.mcts_saved_filename = f'{config.datasets.dataset_name}_{config.models.model_name}_{config.explainers.param.target_event_idx}.pt'
    print(OmegaConf.to_yaml(config))

    # import ipdb; ipdb.set_trace()

    if torch.cuda.is_available() and  config.explainers.use_gpu:
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
    else: 
        raise NotImplementedError('To do.')

    state_dict = torch.load(config.models.ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(device)

    # import ipdb; ipdb.set_trace()
    target_event_idx = config.explainers.param.target_event_idx

    if config.explainers.explainer_name == 'subgraphx_tg':
        explainer = SubgraphXTG(model, 
                                config.models.model_name, 
                                config.explainers.explainer_name,
                                config.datasets.dataset_name,
                                events,
                                config.explainers.param.explanation_level, 
                                device=device,
                                save_results=config.explainers.results_save,
                                save_dir=config.explainers.results_saved_dir,
                                save_filename=config.explainers.results_saved_filename,
                                load_results=config.explainers.load_results,
                                rollout=config.explainers.param.rollout,
        )
        
    elif config.explainers.explainer_name == 'attn_explainer_tg':
        explainer = AttnExplainerTG(
                                model,
                                config.models.model_name,
                                config.explainers.explainer_name,
                                config.datasets.dataset_name,
                                events,
                                config.explainers.param.explanation_level, 
                                device=device,
        )



    # run the explainer
    # target_event_idx = 9909
    # target_event_idx = 19
    # target_event_idx = 29
    
    # tree_results, tree_node_x = explainer(event_idx=target_event_idx)
    e_idx_weight_list = explainer(event_idx=target_event_idx)
    
    if config.explainers.explainer_name == 'subgraphx_tg':
        from dig.xgraph.evaluation.metrics_tg import EvaluatorMCTSTG
        evaluator = EvaluatorMCTSTG() # TODO: fill params
    elif config.explainers.explainer_name == 'attn_explainer_tg':
        from dig.xgraph.evaluation.metrics_tg import EvaluatorAttenTG
        evaluator = EvaluatorAttenTG(config.models.model_name,
                                     config.explainers.explainer_name,
                                     config.datasets.dataset_name,
                                     explainer.ori_subgraph_df,
                                     explainer.candidate_events,
                                     explainer.tgnn_reward_wraper,
                                     target_event_idx,
                                     config.explainers.results_dir)

    evaluator.evaluate(e_idx_weight_list)
    exit(0)
    # evaluate
    
    import json
    sparsity = 0.0
    mcts_state_map = explainer.mcts_state_map if not config.explainers.load_results else None
    evaluator = EvaluatorMCTSTG(explainer.tgnn_reward_wraper, tree_results, sparsity, target_event_idx,
                                    candidate_number=len(explainer.candidate_events),
                                    mcts_state_map=mcts_state_map,
                                    )
    evaluation_dict = evaluator.evaluate()
    print(json.dumps(evaluation_dict, indent=4))

    if not config.explainers.load_results: # TODO: to optimize this
        mcts_info_dict = {
            'rollout number': explainer.mcts_state_map.n_rollout,
            'state number': len(explainer.mcts_state_map.state_map),
            'rollout avg runtime': explainer.mcts_state_map.run_time / explainer.mcts_state_map.n_rollout,
            'rollout total time': explainer.mcts_state_map.run_time
        }
        print(json.dumps(mcts_info_dict, indent=4))
    
    print('candidate event idxs: ', explainer.candidate_events)
    print('searched important event idxs: ', tree_node_x.coalition)



if __name__ == '__main__':
    pipeline()


