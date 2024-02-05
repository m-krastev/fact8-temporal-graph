#!/bin/bash

# if ls $pwd does not contain tgnnexplainer, and benchmarks, then exit
if [ ! -d "tgnnexplainer" ] || [ ! -d "benchmarks" ]; then
    echo "Please run this script from the root of the repository"
    exit 1
fi
export ROOT="$PWD"
export PYTHONPATH="$ROOT:$PYTHONPATH:."

# create empty directories
mkdir -p "$ROOT/tgnnexplainer/xgraph/dataset/data"
mkdir -p "$ROOT/tgnnexplainer/xgraph/models/ext/tgat/processed"
mkdir -p "$ROOT/tgnnexplainer/xgraph/dataset/explain_index"
mkdir -p "$ROOT/tgnnexplainer/xgraph/explainer_ckpts"
mkdir -p "$ROOT/tgnnexplainer/xgraph/saved_mcts_results"
mkdir -p "$ROOT/tgnnexplainer/xgraph/models/checkpoints"

# create environment, if it doesn't exist
if [ ! -d "$ROOT/.venv" ]; then
    python3 -m venv "$ROOT/.venv"
    python3 -m pip install --upgrade pip
    python3 -m pip install "$ROOT"
fi

echo "Installation complete"
echo "To activate the environment, run: source $ROOT/.venv/bin/activate"
echo "To deactivate the environment, run: deactivate"
echo
echo "Next, please make sure the supplementary material is available locally as a zip file."
echo "Finally, please run $ROOT/scripts/unpack.sh to unzip and organize the data, weights, and results."