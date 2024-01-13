# run baselines
dataset=simulate_v1 # wikipedia, reddit, simulate_v1, 
model=tgn # tgat, tgn

# baselines
cd ~/workspace/DIG/benchmarks/xgraph
python subgraphx_tg_run.py datasets=${dataset} device_id=0 explainers=attn_explainer_tg models=${model}
python subgraphx_tg_run.py datasets=${dataset} device_id=0 explainers=pg_explainer_tg models=${model}
python subgraphx_tg_run.py datasets=${dataset} device_id=0 explainers=pbone_explainer_tg models=${model}


# ours
python subgraphx_tg_run.py datasets=${dataset} device_id=1 explainers=subgraphx_tg models=${model}


#########
# HYDRA_FULL_ERROR=1 python -m ipdb -c 'continue'
