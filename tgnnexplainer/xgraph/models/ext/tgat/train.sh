
# simulate_v1, simulate_v2, wikipedia, reddit
sim_datasets=(simulate_v1 simulate_v2)
# real_datasets=(wikipedia reddit)

sim_datasets=(simulate_v1)
real_datasets=(reddit)
runs=(0)
for run in ${runs[@]}
do
    echo "Iteration no. ${run}\n"

    for dataset in ${sim_datasets[@]}
    do
        echo "dataset: ${dataset}\n"
        python learn_simulate.py -d ${dataset} --bs 256 --n_degree 10 --n_epoch 90 --agg_method attn --attn_mode prod --gpu 1 --n_head 2 --prefix ${dataset}
    done

    # for dataset in ${real_datasets[@]}
    # do
    #     echo "dataset: ${dataset}\n"
    #     python learn_edge.py -d ${dataset} --bs 512 --n_degree 10 --n_epoch 10 --agg_method attn --attn_mode prod --gpu 1 --n_head 2 --prefix ${dataset}
    # done

done