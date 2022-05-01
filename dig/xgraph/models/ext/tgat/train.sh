
# garden_5, wikipedia, reddit
dataset=wikipedia
python -u learn_edge.py -d ${dataset} --bs 64 --n_degree 8 --n_epoch 20 --agg_method attn --attn_mode prod --gpu 0 --n_head 2 --prefix ${dataset}



# cp saved_checkpoints/${dataset}-attn-prod-2.pth  ~/workspace/DIG/dig/xgraph/models/checkpoints/tgat_${dataset}_best.pth
