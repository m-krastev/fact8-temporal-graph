#!/bin/bash

# if ls $pwd does not contain tgnnexplainer, and benchmarks, then exit
if [ ! -d "tgnnexplainer" ] || [ ! -d "benchmarks" ]; then
    echo "Please run this script from the root of the repository"
    exit 1
fi
export ROOT="$PWD"
export PYTHONPATH="$ROOT:$PYTHONPATH:."

echo "Creating additional directories and installing the package"
# create empty directories
mkdir -p "$ROOT/tgnnexplainer/xgraph/dataset/data"
mkdir -p "$ROOT/tgnnexplainer/xgraph/models/ext/tgat/processed"
mkdir -p "$ROOT/tgnnexplainer/xgraph/dataset/explain_index"
mkdir -p "$ROOT/tgnnexplainer/xgraph/explainer_ckpts"
mkdir -p "$ROOT/tgnnexplainer/xgraph/saved_mcts_results"
mkdir -p "$ROOT/tgnnexplainer/xgraph/models/checkpoints"
echo
echo "Directories created"
echo

cd "$ROOT"
# create environment, if it doesn't exist
if [ ! -d "$ROOT/.venv" ]; then
    python3 -m venv "$ROOT/.venv"
    echo "New virtual environment created"
fi
echo
echo "Activating virtual environment and installing the package"
echo
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install "$ROOT"

echo
echo "Installation complete"