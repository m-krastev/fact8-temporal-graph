#!/usr/bin/env zsh
# run all explainers

datasets=(simulate_v1 simulate_v2 reddit wikipedia)
models=(tgat tgn)

cd "$ROOT/benchmarks/xgraph"
echo "cwd: $PWD"
processes=4
threshold_num=$1 # NOTE: MUST BE SET
echo "threshold num: ${threshold_num}"

for dataset in ${datasets[@]}
do
    echo "dataset: ${dataset}\n"
    for model in ${models[@]}
    do
        echo "model: ${model}\n"
        # ours
        python subgraphx_tg_run.py datasets=${dataset} device_id=0 explainers=subgraphx_tg models=${model} ++explainers.parallel_degree=$processes explainers.param.$dataset.threshold_num=$threshold_num

        # baselines
        python subgraphx_tg_run.py datasets=${dataset} device_id=0 explainers=attn_explainer_tg models=${model} ++explainers.parallel_degree=$processes ++explainers.param.threshold_num=$threshold_num
        python subgraphx_tg_run.py datasets=${dataset} device_id=0 explainers=attn_explainer_v2 models=${model} ++explainers.parallel_degree=$processes ++explainers.param.threshold_num=$threshold_num
        python subgraphx_tg_run.py datasets=${dataset} device_id=0 explainers=pg_explainer_tg models=${model} ++explainers.parallel_degree=$processes ++explainers.param.threshold_num=$threshold_num
        python subgraphx_tg_run.py datasets=${dataset} device_id=0 explainers=pbone_explainer_tg models=${model} ++explainers.parallel_degree=$processes ++explainers.param.threshold_num=$threshold_num
    done
done