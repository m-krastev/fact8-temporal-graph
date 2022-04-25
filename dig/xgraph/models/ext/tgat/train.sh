
# garden_5
python -u learn_edge.py -d garden_5 --bs 64 --n_degree 20 --agg_method attn --attn_mode prod --gpu 1 --n_head 2 --prefix garden_5
cp saved_checkpoints/garden_5-attn-prod-2.pth  ~/workspace/DIG/dig/xgraph/models/checkpoints/tgat_garden_5_best.pth
