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

# download the raw- and the processed data
# TODO: also needs to processed scripts, not just the raw data
scripts/download_and_process.sh

# download the checkpoints for the target models
scripts/download_checkpoints.sh
