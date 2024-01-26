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

The simulated datasets can be downloaded as:

```Bash
curl https://m-krastev.github.io/hawkes-sim-datasets/simulate_v1 > $ROOT/tgnnexplainer/xgraph/dataset/data/simulate_v1.csv
curl https://m-krastev.github.io/hawkes-sim-datasets/simulate_v2 > $ROOT/tgnnexplainer/xgraph/dataset/data/simulate_v2.csv
```

... or generated (note: this requires the [tick](https://https://github.com/X-DataInitiative/tick/issues) library):

```Bash
cd  $ROOT/tgnnexplainer/xgraph/dataset
python generate_simulate_dataset.py -d simulate_v1(simulate_v2)
```

## Generate indices to-be-explained

This will generate the indices of the edges to be explained for each dataset.

```Bash
cd $ROOT/tgnnexplainer/xgraph/dataset
python tg_dataset.py -d wikipedia(reddit, simulate_v1, simulate_v2) -c index
```

## Train tgat/tgn model

### tgat

```Bash
./scripts/train_tgat.sh
```

### tgn

```Bash
./scripts/train_tgn.sh
```

## Run our explainer and other  baselines

```Bash
cd  $ROOT/benchmarks/xgraph
./run.sh
```
