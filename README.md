# Getting Started

## Installation

```Bash
python -m venv .venv
source .venv/bin/activate
pip install .
```

Note: sometimes, you may be getting import errors. Those can be patched by exporting the PYTHONPATH variable to the root of the project.

```Bash
export ROOT="/Users/matey/project/fact8" # change with your root folder
export PYTHONPATH=$ROOT:.:$PYTHONPATH
```

---

## Download wikipedia and reddit datasets

```Bash
curl http://snap.stanford.edu/jodie/reddit.csv > $ROOT/tgnnexplainer/xgraph/dataset/data/reddit.csv

curl http://snap.stanford.edu/jodie/wikipedia.csv > $ROOT/tgnnexplainer/xgraph/dataset/data/wikipedia.csv
```

## Preprocess real-world datasets

```Bash
cd  $ROOT/tgnnexplainer/xgraph/models/ext/tgat
python process.py -d wikipedia
python process.py -d reddit

```

## Generate simulated dataset

```Bash
cd  $ROOT/tgnnexplainer/xgraph/dataset
python generate_simulate_dataset.py -d simulate_v1(simulate_v2)
```

## Generate explain indexs

```Bash
cd $ROOT/tgnnexplainer/xgraph/dataset
python tg_dataset.py -d wikipedia(reddit, simulate_v1, simulate_v2) -c index
```

## Train tgat/tgn model

### tgat

```Bash
cd  $ROOT/tgnnexplainer/xgraph/models/ext/tgat
./train.sh
./cpckpt.sh
```

### tgn

```Bash
cd  $ROOT/tgnnexplainer/xgraph/models/ext/tgn
./train.sh
./cpckpt.sh
```

## Run our explainer and other  baselines

```Bash
cd  $ROOT/benchmarks/xgraph
./run.sh
```
