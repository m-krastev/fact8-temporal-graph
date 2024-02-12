# [RE] Explaining Temporal Graph Models through an Explorer-Navigator Framework

This repository is based on the code provided by the authors of the paper ["Explaining Temporal Graph Models through an Explorer-Navigator Framework"](https://openreview.net/forum?id=BR_ZhvcYbGJ) by Xia et al (2023). We optimize and extend the original code with additional features and installation scripts. Namely, we disambiguate the nomenclature of the PGExplainer baseline and the PGNavigator model, which are identical in their inference but used in two distinct roles. We also define two additional navigator models, the MLPNavigator, which strictly follows the definition of the navigator described in the paper and the DotProductNavigator, which computes similarity scores between output embeddings of the target model. 

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

```Bash
# TGAT model
./scripts/train_tgat.sh

# TGN model
./scripts/train_tgn.sh
```

## Results

### Highlights

The reproductions of the original authors' published findings are listed below.

<style type="text/css">
</style>
<table id="T_0a118">
  <caption>Experimental results for TGAT model.</caption>
  <thead>
    <tr>
      <th class="index_name level0" ></th>
      <th id="T_0a118_level0_col0" class="col_heading level0 col0" colspan="2">Wikipedia</th>
      <th id="T_0a118_level0_col2" class="col_heading level0 col2" colspan="2">Reddit</th>
      <th id="T_0a118_level0_col4" class="col_heading level0 col4" colspan="2">Simulate V1</th>
      <th id="T_0a118_level0_col6" class="col_heading level0 col6" colspan="2">Simulate V2</th>
    </tr>
    <tr>
      <th class="index_name level1" ></th>
      <th id="T_0a118_level1_col0" class="col_heading level1 col0" >Best FID</th>
      <th id="T_0a118_level1_col1" class="col_heading level1 col1" >AUFSC</th>
      <th id="T_0a118_level1_col2" class="col_heading level1 col2" >Best FID</th>
      <th id="T_0a118_level1_col3" class="col_heading level1 col3" >AUFSC</th>
      <th id="T_0a118_level1_col4" class="col_heading level1 col4" >Best FID</th>
      <th id="T_0a118_level1_col5" class="col_heading level1 col5" >AUFSC</th>
      <th id="T_0a118_level1_col6" class="col_heading level1 col6" >Best FID</th>
      <th id="T_0a118_level1_col7" class="col_heading level1 col7" >AUFSC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_0a118_level0_row0" class="row_heading level0 row0" >ATTN</th>
      <td id="T_0a118_row0_col0" class="data row0 col0" >0.529735</td>
      <td id="T_0a118_row0_col1" class="data row0 col1" >0.082103</td>
      <td id="T_0a118_row0_col2" class="data row0 col2" >0.040894</td>
      <td id="T_0a118_row0_col3" class="data row0 col3" >-0.115032</td>
      <td id="T_0a118_row0_col4" class="data row0 col4" >0.872525</td>
      <td id="T_0a118_row0_col5" class="data row0 col5" >0.594758</td>
      <td id="T_0a118_row0_col6" class="data row0 col6" >0.475007</td>
      <td id="T_0a118_row0_col7" class="data row0 col7" >-0.908423</td>
    </tr>
    <tr>
      <th id="T_0a118_level0_row1" class="row_heading level0 row1" >PBONE</th>
      <td id="T_0a118_row1_col0" class="data row1 col0" >0.939908</td>
      <td id="T_0a118_row1_col1" class="data row1 col1" >0.537294</td>
      <td id="T_0a118_row1_col2" class="data row1 col2" >0.658535</td>
      <td id="T_0a118_row1_col3" class="data row1 col3" >0.346819</td>
      <td id="T_0a118_row1_col4" class="data row1 col4" >1.258814</td>
      <td id="T_0a118_row1_col5" class="data row1 col5" >0.862181</td>
      <td id="T_0a118_row1_col6" class="data row1 col6" >1.225851</td>
      <td id="T_0a118_row1_col7" class="data row1 col7" >0.873869</td>
    </tr>
    <tr>
      <th id="T_0a118_level0_row2" class="row_heading level0 row2" >PG</th>
      <td id="T_0a118_row2_col0" class="data row2 col0" >0.619685</td>
      <td id="T_0a118_row2_col1" class="data row2 col1" >-0.321997</td>
      <td id="T_0a118_row2_col2" class="data row2 col2" >0.718188</td>
      <td id="T_0a118_row2_col3" class="data row2 col3" >0.210220</td>
      <td id="T_0a118_row2_col4" class="data row2 col4" >0.714999</td>
      <td id="T_0a118_row2_col5" class="data row2 col5" >-0.411107</td>
      <td id="T_0a118_row2_col6" class="data row2 col6" >0.478570</td>
      <td id="T_0a118_row2_col7" class="data row2 col7" >-0.821495</td>
    </tr>
    <tr>
      <th id="T_0a118_level0_row3" class="row_heading level0 row3" >PGNavigator</th>
      <td id="T_0a118_row3_col0" class="data row3 col0" >1.155183</td>
      <td id="T_0a118_row3_col1" class="data row3 col1" >0.842484</td>
      <td id="T_0a118_row3_col2" class="data row3 col2" >0.788783</td>
      <td id="T_0a118_row3_col3" class="data row3 col3" >0.719582</td>
      <td id="T_0a118_row3_col4" class="data row3 col4" >1.513354</td>
      <td id="T_0a118_row3_col5" class="data row3 col5" >1.142643</td>
      <td id="T_0a118_row3_col6" class="data row3 col6" >1.155439</td>
      <td id="T_0a118_row3_col7" class="data row3 col7" >0.444023</td>
    </tr>
    <tr>
      <th id="T_0a118_level0_row4" class="row_heading level0 row4" >MLPNavigator</th>
      <td id="T_0a118_row4_col0" class="data row4 col0" >1.181733</td>
      <td id="T_0a118_row4_col1" class="data row4 col1" >0.776888</td>
      <td id="T_0a118_row4_col2" class="data row4 col2" >0.794740</td>
      <td id="T_0a118_row4_col3" class="data row4 col3" >0.605059</td>
      <td id="T_0a118_row4_col4" class="data row4 col4" >1.394806</td>
      <td id="T_0a118_row4_col5" class="data row4 col5" >0.881115</td>
      <td id="T_0a118_row4_col6" class="data row4 col6" >1.162250</td>
      <td id="T_0a118_row4_col7" class="data row4 col7" >0.367933</td>
    </tr>
    <tr>
      <th id="T_0a118_level0_row5" class="row_heading level0 row5" >DotProductNavigator</th>
      <td id="T_0a118_row5_col0" class="data row5 col0" >0.987423</td>
      <td id="T_0a118_row5_col1" class="data row5 col1" >0.469339</td>
      <td id="T_0a118_row5_col2" class="data row5 col2" >0.783412</td>
      <td id="T_0a118_row5_col3" class="data row5 col3" >0.712787</td>
      <td id="T_0a118_row5_col4" class="data row5 col4" >1.252745</td>
      <td id="T_0a118_row5_col5" class="data row5 col5" >0.597811</td>
      <td id="T_0a118_row5_col6" class="data row5 col6" >1.223335</td>
      <td id="T_0a118_row5_col7" class="data row5 col7" >0.596084</td>
    </tr>
  </tbody>
</table>


<style type="text/css">
</style>
<table id="T_c3d57">
  <caption>Experimental results for TGN model.</caption>
  <thead>
    <tr>
      <th class="index_name level0" ></th>
      <th id="T_c3d57_level0_col0" class="col_heading level0 col0" colspan="2">Wikipedia</th>
      <th id="T_c3d57_level0_col2" class="col_heading level0 col2" colspan="2">Reddit</th>
      <th id="T_c3d57_level0_col4" class="col_heading level0 col4" colspan="2">Simulate V1</th>
      <th id="T_c3d57_level0_col6" class="col_heading level0 col6" colspan="2">Simulate V2</th>
    </tr>
    <tr>
      <th class="index_name level1" ></th>
      <th id="T_c3d57_level1_col0" class="col_heading level1 col0" >Best FID</th>
      <th id="T_c3d57_level1_col1" class="col_heading level1 col1" >AUFSC</th>
      <th id="T_c3d57_level1_col2" class="col_heading level1 col2" >Best FID</th>
      <th id="T_c3d57_level1_col3" class="col_heading level1 col3" >AUFSC</th>
      <th id="T_c3d57_level1_col4" class="col_heading level1 col4" >Best FID</th>
      <th id="T_c3d57_level1_col5" class="col_heading level1 col5" >AUFSC</th>
      <th id="T_c3d57_level1_col6" class="col_heading level1 col6" >Best FID</th>
      <th id="T_c3d57_level1_col7" class="col_heading level1 col7" >AUFSC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_c3d57_level0_row0" class="row_heading level0 row0" >ATTN</th>
      <td id="T_c3d57_row0_col0" class="data row0 col0" >1.422790</td>
      <td id="T_c3d57_row0_col1" class="data row0 col1" >0.788410</td>
      <td id="T_c3d57_row0_col2" class="data row0 col2" >1.648984</td>
      <td id="T_c3d57_row0_col3" class="data row0 col3" >-0.974337</td>
      <td id="T_c3d57_row0_col4" class="data row0 col4" >0.596853</td>
      <td id="T_c3d57_row0_col5" class="data row0 col5" >0.418108</td>
      <td id="T_c3d57_row0_col6" class="data row0 col6" >0.180901</td>
      <td id="T_c3d57_row0_col7" class="data row0 col7" >-1.456566</td>
    </tr>
    <tr>
      <th id="T_c3d57_level0_row1" class="row_heading level0 row1" >PBONE</th>
      <td id="T_c3d57_row1_col0" class="data row1 col0" >1.677653</td>
      <td id="T_c3d57_row1_col1" class="data row1 col1" >0.750877</td>
      <td id="T_c3d57_row1_col2" class="data row1 col2" >2.988232</td>
      <td id="T_c3d57_row1_col3" class="data row1 col3" >0.137898</td>
      <td id="T_c3d57_row1_col4" class="data row1 col4" >0.734512</td>
      <td id="T_c3d57_row1_col5" class="data row1 col5" >0.432308</td>
      <td id="T_c3d57_row1_col6" class="data row1 col6" >0.265230</td>
      <td id="T_c3d57_row1_col7" class="data row1 col7" >-0.616406</td>
    </tr>
    <tr>
      <th id="T_c3d57_level0_row2" class="row_heading level0 row2" >PG</th>
      <td id="T_c3d57_row2_col0" class="data row2 col0" >1.318994</td>
      <td id="T_c3d57_row2_col1" class="data row2 col1" >-0.011089</td>
      <td id="T_c3d57_row2_col2" class="data row2 col2" >0.990243</td>
      <td id="T_c3d57_row2_col3" class="data row2 col3" >-2.312849</td>
      <td id="T_c3d57_row2_col4" class="data row2 col4" >0.549872</td>
      <td id="T_c3d57_row2_col5" class="data row2 col5" >-0.418763</td>
      <td id="T_c3d57_row2_col6" class="data row2 col6" >0.150199</td>
      <td id="T_c3d57_row2_col7" class="data row2 col7" >-2.179182</td>
    </tr>
    <tr>
      <th id="T_c3d57_level0_row3" class="row_heading level0 row3" >PGNavigator</th>
      <td id="T_c3d57_row3_col0" class="data row3 col0" >1.820665</td>
      <td id="T_c3d57_row3_col1" class="data row3 col1" >1.467331</td>
      <td id="T_c3d57_row3_col2" class="data row3 col2" >2.825218</td>
      <td id="T_c3d57_row3_col3" class="data row3 col3" >1.769516</td>
      <td id="T_c3d57_row3_col4" class="data row3 col4" >0.920611</td>
      <td id="T_c3d57_row3_col5" class="data row3 col5" >0.680338</td>
      <td id="T_c3d57_row3_col6" class="data row3 col6" >0.265451</td>
      <td id="T_c3d57_row3_col7" class="data row3 col7" >-1.056114</td>
    </tr>
    <tr>
      <th id="T_c3d57_level0_row4" class="row_heading level0 row4" >MLPNavigator</th>
      <td id="T_c3d57_row4_col0" class="data row4 col0" >1.907501</td>
      <td id="T_c3d57_row4_col1" class="data row4 col1" >1.493589</td>
      <td id="T_c3d57_row4_col2" class="data row4 col2" >2.493876</td>
      <td id="T_c3d57_row4_col3" class="data row4 col3" >0.819607</td>
      <td id="T_c3d57_row4_col4" class="data row4 col4" >0.934737</td>
      <td id="T_c3d57_row4_col5" class="data row4 col5" >0.491064</td>
      <td id="T_c3d57_row4_col6" class="data row4 col6" >0.255969</td>
      <td id="T_c3d57_row4_col7" class="data row4 col7" >-1.460341</td>
    </tr>
    <tr>
      <th id="T_c3d57_level0_row5" class="row_heading level0 row5" >DotProductNavigator</th>
      <td id="T_c3d57_row5_col0" class="data row5 col0" >1.300528</td>
      <td id="T_c3d57_row5_col1" class="data row5 col1" >0.398448</td>
      <td id="T_c3d57_row5_col2" class="data row5 col2" >2.945419</td>
      <td id="T_c3d57_row5_col3" class="data row5 col3" >0.653993</td>
      <td id="T_c3d57_row5_col4" class="data row5 col4" >0.908063</td>
      <td id="T_c3d57_row5_col5" class="data row5 col5" >0.371170</td>
      <td id="T_c3d57_row5_col6" class="data row5 col6" >0.264682</td>
      <td id="T_c3d57_row5_col7" class="data row5 col7" >-1.284565</td>
    </tr>
  </tbody>
</table>



### Full Results

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

### Generate simulated dataset

The simulated datasets can be generated (note: this requires the [tick](https://https://github.com/X-DataInitiative/tick/issues) library):

```Bash
cd  $ROOT/tgnnexplainer/xgraph/dataset
python generate_simulate_dataset.py -d simulate_v1(simulate_v2)
```

### Preprocess real-world datasets

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


### Generate indices to-be-explained

This will generate the indices of the edges to be explained for each dataset.

```Bash
cd $ROOT/tgnnexplainer/xgraph/dataset
python tg_dataset.py -d wikipedia(reddit, simulate_v1, simulate_v2) -c index
```

