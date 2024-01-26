#!/usr/bin/env bash

model=tgat

cd  $ROOT/tgnnexplainer/xgraph/models/ext/tgat

# create the savepath directory for the checkpoints
mkdir -p $ROOT/tgnnexplainer/xgraph/models/checkpoints

sim_datasets=(simulate_v1 simulate_v2)
real_datasets=(wikipedia wikipedia)
runs=(0 1 2)

sim_epochs=100
real_epochs=10
for run in ${runs[@]}
do
    echo "Iteration no. ${run}"

    # ========== Train on simulated datasets ==========
    for dataset in ${sim_datasets[@]}
    do
        echo "dataset: ${dataset}"
        python learn_simulate.py -d ${dataset} --bs 256 --n_degree 10 --n_epoch ${sim_epochs} --agg_method attn --attn_mode prod --gpu 1 --n_head 2 --prefix ${dataset}

        source_path=./saved_checkpoints/${dataset}-attn-prod-${$(($sim_epochs-1))}.pth
        target_path=$ROOT/tgnnexplainer/xgraph/models/checkpoints/${model}_${dataset}_best.pth
        cp ${source_path} ${target_path}

        echo ${source_path} ${target_path} 'copied'
    done

    # ========== Train on real datasets ==========
    for dataset in ${real_datasets[@]}
    do
        echo "dataset: ${dataset}"
        python learn_edge.py -d ${dataset} --bs 512 --n_degree 10 --n_epoch ${real_epochs} --agg_method attn --attn_mode prod --gpu 1 --n_head 2 --prefix ${dataset}

        source_path=./saved_checkpoints/${dataset}-attn-prod-${(($real_epochs-1))}.pth
        target_path=$ROOT/tgnnexplainer/xgraph/models/checkpoints/${model}_${dataset}_best.pth
        cp ${source_path} ${target_path}
        echo ${source_path} ${target_path} 'copied'

    done
done