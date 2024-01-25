# run all explainers

# datasets=(simulate_v1 simulate_v2 wikipedia reddit)
datasets=(simulate_v1)
# models=(tgat tgn)
models=(tgat)

for dataset in ${datasets}
do
    echo "dataset: ${dataset}\n"
    for model in ${models}
    do
        echo "model: ${model}\n"
        # ours
        python subgraphx_tg_run.py datasets=${dataset} device_id=0 explainers=subgraphx_tg models=${model}

        # baselines
        python subgraphx_tg_run.py datasets=${dataset} device_id=0 explainers=attn_explainer_tg models=${model}
#        python subgraphx_tg_run.py datasets=${dataset} device_id=0 #explainers=pg_explainer_tg models=${model}
#        python subgraphx_tg_run.py datasets=${dataset} device_id=0 #explainers=pbone_explainer_tg models=${model}
    done
done
