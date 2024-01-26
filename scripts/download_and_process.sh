#!/usr/bin/bash

# Data downloading and preprocessing
curl http://snap.stanford.edu/jodie/reddit.csv > $ROOT/tgnnexplainer/xgraph/dataset/data/reddit.csv
curl http://snap.stanford.edu/jodie/wikipedia.csv > $ROOT/tgnnexplainer/xgraph/dataset/data/wikipedia.csv

# generate simulated dataset
# NOTE: the simulated dataset is already pre-generated and pre-processed
curl https://m-krastev.github.io/hawkes-sim-datasets/simulate_v1 > $ROOT/tgnnexplainer/xgraph/dataset/data/simulate_v1.csv
curl https://m-krastev.github.io/hawkes-sim-datasets/simulate_v2 > $ROOT/tgnnexplainer/xgraph/dataset/data/simulate_v2.csv
# cd  $ROOT/tgnnexplainer/xgraph/dataset
# python generate_simulate_dataset.py -d simulate_v1
# python generate_simulate_dataset.py -d simulate_v2


# process the real dataset
cd  $ROOT/tgnnexplainer/xgraph/models/ext/tgat
python process.py -d wikipedia
python process.py -d reddit

# generate indices to-be-explained
cd $ROOT/tgnnexplainer/xgraph/dataset
python tg_dataset.py -d wikipedia -c index
python tg_dataset.py -d reddit -c index
python tg_dataset.py -d simulate_v1 -c index
python tg_dataset.py -d simulate_v2 -c index
