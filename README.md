# Getting Started

## Installation

To install this package, please make sure the current working directory is set to the package root folder, e.g. `/home/user/fact8-temporal-graph` and run the following commands:

```Bash
export ROOT="$PWD"
export PYTHONPATH="$ROOT:$PYTHONPATH:."

# create empty directories
mkdir -p "$ROOT/tgnnexplainer/xgraph/dataset/data"
mkdir -p "$ROOT/tgnnexplainer/xgraph/models/ext/tgat/processed"
mkdir -p "$ROOT/tgnnexplainer/xgraph/dataset/explain_index"
mkdir -p "$ROOT/tgnnexplainer/xgraph/explainer_ckpts"
mkdir -p "$ROOT/tgnnexplainer/xgraph/saved_mcts_results"
mkdir -p "$ROOT/tgnnexplainer/xgraph/models/checkpoints"

# create new environment
python3 -m venv "$ROOT/.venv"
source "$ROOT/.venv/bin/activate"
# upgrade pip and install the package
python3 -m pip install --upgrade pip
python3 -m pip install .
```

in the end, you should have a virtual environment in the `.venv` folder and the package with all its dependencies installed in it.

Finally, to save some time we made all the datasets, model weights and our rported results available for download. Please obtain this file manually and save it in the project root folder, e.g. `$ROOT/data.zip`. 

Once the archive is available, you can run the following command to extract the data:

```Bash
./scripts/unpack.sh --source $ROOT/data.zip --data --weights --results
```

This will extract all the datasets, model weights and our reported results. To exclude any of these, you can omit the corresponding flags from the above command.


## Run the explainer and other baselines

```Bash
cd  $ROOT/benchmarks/xgraph
./run.sh
```


## Download and process all datasets

In case the aforementioned supplementary archive is not available, one can run the following command to download and process all datasets:

```Bash
./scripts/download_and_process.sh
```

This will download the wikipedia and reddit datasets as well as the simulated datasets. Although the simulated datasets can also be generated, (see next section), we recommend using the provided datasets as they are the same as the ones used in the paper. Also, in our experience, installing the `tick` library is not trivial and we could only do it by rolling back to python 3.8. 

## Generate simulated dataset

The simulated datasets can be generated (note: this requires the [tick](https://https://github.com/X-DataInitiative/tick/issues) library):

```Bash
cd  $ROOT/tgnnexplainer/xgraph/dataset
python generate_simulate_dataset.py -d simulate_v1(simulate_v2)
```

## Preprocess real-world datasets

Can be done manually:
```Bash
cd  $ROOT/tgnnexplainer/xgraph/models/ext/tgat
python process.py -d wikipedia
python process.py -d reddit

cd $ROOT/tgnnexplainer/xgraph/dataset
python tg_dataset.py -d wikipedia(reddit, simulate_v1, simulate_v2) -c index
```
or using the provided script:
```
./scripts/download_and_process.sh
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

