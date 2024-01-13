model=tgat
dataset=simulate_v1
epoch=20

source_path=~/workspace/DIG/dig/xgraph/explainer_ckpts/${model}_${dataset}_pg_explainer_tg_expl_ckpt_ep${epoch}.pt
target_path=~/workspace/DIG/dig/xgraph/explainer_ckpts/${model}_${dataset}_pg_explainer_tg_expl_ckpt.pt


cp ${source_path} ${target_path}

echo ${source_path} ${target_path} 'copied'


ls -l ~/workspace/DIG/dig/xgraph/explainer_ckpts/ |grep '.*expl_ckpt.pt'