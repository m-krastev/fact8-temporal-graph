# [RE] Explaining Temporal Graph Models through an Explorer-Navigator Framework

This repository is based on the code provided by the authors of the paper "Explaining Temporal Graph Models through an Explorer-Navigator Framework" (https://openreview.net/forum?id=BR_ZhvcYbGJ). We optimize and extend the original code with additional features and installation scripts. Namely, we disambiguate the nomenclature of the PGExplainer baseline and the PGNavigator model, which are identical in their inference but used in two distinct roles. We also define two additional navigator models, the MLPNavigator, which strictly follows the definition of the navigator described in the paper and the DotProductNavigator, which computes similarity scores between output embeddings of the target model. 

We also provide a setup script, which populates the repository with supplementary data, such as datasets, model weights and reported results. 

Finally, we provide two notebooks, which can be used to generate the figures and tables reported in our paper.

## Requirements

To install the package, please make sure the current working directory is set to the package root folder, e.g. `/home/user/fact8-temporal-graph` and run the following commands:

```Bash
./scripts/install.sh # this will create necessary directories and install the package
export ROOT="$PWD"
export PYTHONPATH="$ROOT:$PYTHONPATH:." # these are necessary to avoid pathing issues
source .venv/bin/activate # activate the virtual environment where the package is installed
```

Please make sure the above defined environment variables are set before running the package and that the virtual environment is activated. 

The package was tested with Python >= 3.11.0 with the packages defined in the `pyproject.toml` file.

## Populating the repository with supplementary data

To save some time we made all the datasets, (both raw and processed), model weights and our reported results available for download. Please obtain this file manually and save it in the project root folder, e.g. `$ROOT/data.zip`. Once this is done, you can run the following command to extract the data:

```Bash
./scripts/unpack.sh --source $ROOT/data.zip --data --weights --results
```

This will extract all the datasets, model weights and our reported results. To exclude any of these, you can omit the corresponding flags from the above command.

To manually download and process all the datasets, please refer to [Section 6](#download-and-process-all-datasets). Sections [7](#generate-simulated-dataset) and [8](#preprocess-real-world-datasets) provide instructions on how to generate the simulated datasets and preprocess the real-world datasets, respectively.

Instructions on how to train the target models can be found in [Section 4](#training).

## Run the explainer and other baselines


The main insertion porint for running the explainers is in `benchmarks/xgraph/subgraphx_tg_run.py`. To avoid having to deal with the command line arguments, the `benchmarks/xgraph/run.sh` script is provided. To include/exclude any of the models, please comment our the corresponding lines in this script.

Futher hyper-parameters can be found under the `benchmarks/xgraph/config` directory. Currently, all parameters are set to the values reported our paper.

```Bash
cd "$ROOT/bechmarks/xgraph"
./run.sh
```
This will run the explainer models and the baselines. The results will be saved in the `benchmarks/results` directory.

> **Note**: The explainer models can only be run if there are existing model weights for the target models. If this is not the case, please refer to the end of the previous section or the next section for instructions on how to train the target models.
>
> In the absence of explainer checkpoints (used in the pre-trained navigator), a training run will automatically trigger when the navigator is instantiated. This is the case for the PGNavigator and MLPNavigator models. Also note that the weights are shared across these models, as well as with the PGExplainer model. Running either of these will traing the weights for all of them.

## Training

Training the target models needs to be done prior to running any of the explainer models. Note that the training time on the real-world datasets can take several hours so we recommend using the provided model weights (please refer to [section 2](#populating-the-repository-with-supplementary-data)).

### tgat

```Bash
./scripts/train_tgat.sh
```

### tgn

```Bash
./scripts/train_tgn.sh
```

## Results

The results of the explainer models can be found under the `benchmarks/results` directory and numpy binary files containing the states explored by the MCTS algorithm can be found under the `tgnnexplainer/xgraph/saved_mcts_results` directory.

To generate the figures and tables reported in our paper, please refer to the `notebooks` directory. There, the `results_processing.ipynb` notebook contains most of the code used to generate the plots and tables for our results with threshold 20. Additionally, the `threshold_results.ipynb` notebook contains the code used to generate the results regarding our hyper-parameter runing experiments with the number of candidate events.

> **NOTE**: Make sure all results are available before the notebooks are run. Namely, all T-GNNExplainer and baseline results for threshold 20, 25 as well as threshold 5 and 10 results on the PGNavigator variation over the two synthetic datasets. 
>
> We made these available in the supplementary material, but of course the reader is welcome to replicate them individually.

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

