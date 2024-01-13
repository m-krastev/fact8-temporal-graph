model=tgat
dataset=simulate_v2
epoch=99

source_path=./saved_checkpoints/${dataset}-attn-prod-${epoch}.pth
target_path=$current_dir/tgnnexplainer/xgraph/models/checkpoints/${model}_${dataset}_best.pth
cp ${source_path} ${target_path}

echo ${source_path} ${target_path} 'copied'


ls -l $current_dir/dig/xgraph/models/checkpoints