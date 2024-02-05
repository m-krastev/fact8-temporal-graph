# Getting Started

## Installation

```Bash
./scripts/install.sh
```

This will create a new virtual environment and install the required packages.
Next, we need to set two environment variables and activate the virtual environment:

```Bash
export ROOT=$PWD # assuming PWD is the root of the project
export PYTHONPATH=$ROOT:.:$PYTHONPATH
source $ROOT/venv/bin/activate # activate the virtual environment, this is where install.sh installs the packages
```

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

