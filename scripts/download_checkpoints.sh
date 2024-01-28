#!/bin/bash

# if ls $pwd does not contain tgnnexplainer, and benchmarks, then exit
if [ ! -d "tgnnexplainer" ] || [ ! -d "benchmarks" ]; then
    echo "Please run this script from the root of the repository"
    exit 1
fi

export ROOT="$PWD"
export PYTHONPATH="$ROOT:$PYTHONPATH:."

# download the checkpoints
# TODO: upload them somewhere