#!/usr/bin/env zsh
# run all explainers

datasets=(simulate_v1 simulate_v2 reddit wikipedia)
models=(tgat tgn)

cd "$ROOT/benchmarks/xgraph"

# Number of parallel processes. Set to an appropriate value based on the available CPU cores.
processes=1

for dataset in ${datasets[@]}
do
    echo "dataset: ${dataset}\n"
    for model in ${models[@]}
    do
        echo "model: ${model}\n"
        # _____________________________________________
        # _________ T-GNNExplainer variations _________
        # _____________________________________________

        # MLPNavigator
        python subgraphx_tg_run.py datasets=${dataset} device_id=0 explainers=subgraphx_tg models=${model} ++explainers.parallel_degree=$processes ++explainers.navigator_type=mlp
        # PGNavigator
        python subgraphx_tg_run.py datasets=${dataset} device_id=0 explainers=subgraphx_tg models=${model} ++explainers.parallel_degree=$processes ++explainers.navigator_type=pg
        # DotProductNavigator
        python subgraphx_tg_run.py datasets=${dataset} device_id=0 explainers=subgraphx_tg models=${model} ++explainers.parallel_degree=$processes ++explainers.navigator_type=dot
        # No navigator
        python subgraphx_tg_run.py datasets=${dataset} device_id=0 explainers=subgraphx_tg models=${model} ++explainers.parallel_degree=$processes ++explainers.use_navigator=False
        # _____________________________________________
        # _________________ Baselines _________________
        # _____________________________________________

        # ATTN baseline
        python subgraphx_tg_run.py datasets=${dataset} device_id=0 explainers=attn_explainer_tg models=${model} ++explainers.parallel_degree=$processes
        # PGExplainer baseline
        python subgraphx_tg_run.py datasets=${dataset} device_id=0 explainers=pg_explainer_tg models=${model} ++explainers.parallel_degree=$processes
        # PBONE baseline
        python subgraphx_tg_run.py datasets=${dataset} device_id=0 explainers=pbone_explainer_tg models=${model} ++explainers.parallel_degree=$processes
    done
done