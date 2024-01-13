# Download wikipedia and reddit datasets
Download from http://snap.stanford.edu/jodie/wikipedia.csv and http://snap.stanford.edu/jodie/reddit.csv and put them into ~/workspace/TGNNEXPLAINER-PUBLIC/tgnnexplainer/xgraph/dataset/data


# Preprocess real-world datasets
```
cd  ~/workspace/TGNNEXPLAINER-PUBLIC/tgnnexplainer/xgraph/models/ext/tgat
python process.py -d wikipedia
python process.py -d reddit

```

# Generate simulate dataset
```
cd  ~/workspace/TGNNEXPLAINER-PUBLIC/tgnnexplainer/xgraph/dataset
python generate_simulate_dataset.py -d simulate_v1(simulate_v2)
```



# Generate explain indexs
```
cd  ~/workspace/GNNEXPLAINER-PUBLIC/tgnnexplainer/xgraph/dataset
python tg_dataset.py -d wikipedia(reddit, simulate_v1, simulate_v2) -c index
```

# Train tgat/tgn model
tgat:
```
cd  ~/workspace/TGNNEXPLAINER-PUBLIC/tgnnexplainer/xgraph/models/ext/tgat
./train.sh
./cpckpt.sh
```

tgn:
```
cd  ~/workspace/TGNNEXPLAINER-PUBLIC/tgnnexplainer/xgraph/models/ext/tgn
./train.sh
./cpckpt.sh
```

# Run our explainer and other  baselines
```
cd  ~/workspace/TGNNEXPLAINER-PUBLIC/benchmarks
./run.sh
``` 


