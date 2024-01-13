
# simulate_v1, wikipedia, reddit
dataset=simulate_v1

# python learn_edge.py -d ${dataset} --bs 512 --n_degree 8 --n_epoch 10 --agg_method attn --attn_mode prod --gpu 1 --n_head 2 --prefix ${dataset}
python learn_simulate.py -d ${dataset} --bs 256 --n_degree 10 --n_epoch 100 --agg_method attn --attn_mode prod --gpu 1 --n_head 2 --prefix ${dataset}
# python learn_edge.py -d ${dataset} --bs 256 --n_degree 20 --n_epoch 10 --agg_method attn --attn_mode prod --gpu 1 --n_head 2 --prefix ${dataset}
# -m ipdb -c 'continue'


# cp saved_checkpoints/${dataset}-attn-prod-2.pth  ~/workspace/DIG/dig/xgraph/models/checkpoints/tgat_${dataset}_best.pth
