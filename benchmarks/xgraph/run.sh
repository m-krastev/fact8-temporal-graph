# run all explainers
dataset=wikipedia # wikipedia, reddit, simulate_v1, simulate_v2
model=tgat # tgat, tgn

# ours
python subgraphx_tg_run.py datasets=${dataset} device_id=0 explainers=subgraphx_tg models=${model}

# baselines
python subgraphx_tg_run.py datasets=${dataset} device_id=0 explainers=attn_explainer_tg models=${model}
python subgraphx_tg_run.py datasets=${dataset} device_id=0 explainers=pg_explainer_tg models=${model}
python subgraphx_tg_run.py datasets=${dataset} device_id=0 explainers=pbone_explainer_tg models=${model}
