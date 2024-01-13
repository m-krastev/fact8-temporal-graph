# generate simulate dataset
```
cd ~/workspace/DIG/dig/xgraph/dataset
python generate_simulate_dataset.py -d simulate_v1
# simulate_v1, simulate_v2

```

# preprocess datasets
```
cd ~/workspace/DIG/dig/xgraph/models/ext/tgat
<!-- python process.py -rename_w_r # only the very first time run -->
python process.py -d wikipedia
python process.py -d reddit

```

# generate explain indexs
```
cd ~/workspace/DIG/dig/xgraph/dataset
python tg_dataset.py -d wikipedia -c index
```

# train tgat/tgn model
tgat:
```
cd ~/workspace/DIG/dig/xgraph/models/ext/tgat

```

tgn:
```
cd ~/workspace/DIG/dig/xgraph/models/ext/tgn
./train.sh
./cpckpt.sh
```


# run subgraphx_tg and baselines
in current dir:
```
./run.sh
``` 

before run, config each explainer's parameters in the config/ directory.



