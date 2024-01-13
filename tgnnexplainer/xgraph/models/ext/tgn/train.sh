# simulate_v1, simulate_v2, wikipedia, reddit
sim_datasets=(simulate_v1 simulate_v2)
real_datasets=(wikipedia reddit)

sim_datasets=(simulate_v1)
real_datasets=(reddit)
runs=(0)
for run in ${runs[@]}
do
    echo "Iteration no. ${run}\n"

    for dataset in ${sim_datasets[@]}
    do
        echo "dataset: ${dataset}\n"
        python train_simulate.py -d ${dataset} --prefix tgn-attn --n_runs 1 --n_epoch 100 --n_layer 2 --n_degree 10 --use_memory --memory_update_at_end --gpu 0 \
        --memory_dim 4 # memory_dim should equal to node/edge feature dim
    done

    # for dataset in ${real_datasets[@]}
    # do
    #     echo "dataset: ${dataset}\n"
    #     python train_self_supervised.py -d ${dataset} --prefix tgn-attn --n_runs 1 --n_epoch 20 --n_layer 2 --n_degree 10 --use_memory --gpu 0
    # done

done
