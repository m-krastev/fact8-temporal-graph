#!/bin/bash

# check if $ROOT is set
if [ -z "$ROOT" ]; then
    echo "Please set ROOT eviroment variable to the root directory of the project. Exiting..."
    exit 1
fi

DATA_RAW="$ROOT/tgnnexplainer/xgraph/dataset/data"
DATA_PROCESSED="$ROOT/tgnnexplainer/xgraph/models/ext/tgat/processed"
DATA_IDX="$ROOT/tgnnexplainer/xgraph/dataset/explain_index"

MODEL_WEIGHTS="$ROOT/tgnnexplainer/xgraph/models/checkpoints"
EXPLAINER_WEIGHTS="$ROOT/tgnnexplainer/xgraph/explainer_ckpts"

RESULTS="$ROOT/benchmarks/results"
RESULTS_MCTS="$ROOT/tgnnexplainer/xgraph/saved_mcts_results"

TMP="/tmp/tgnnexplainer"
# check if /tmp exists
if [ ! -d /tmp ]; then
    TMP="./tmp"
fi
mkdir -p $TMP

# CLI
# ./script.sh -s source [--data|-d] [--weights|-w] [--results|-r] [--help|-h]
# --source|-s: a zip file with the data, weights, and results
# --data|-d: copy data files
# --weights|-w: copy weights files
# --results|-r: copy results files
# --help|-h: show help
function show_help {
    echo "Usage: $0 -s source [--data|-d] [--weights|-w] [--results|-r] [--help|-h]"
    echo "  -s, --source: a zip file with the data, weights, and results"
    echo "  -d, --data: copy data files"
    echo "  -w, --weights: copy weights files"
    echo "  -r, --results: copy results files"
    echo "  -h, --help: show help"
    exit 1
}

function verify_and_extract_data {
    # if data is not a file or it is not a zip, exit
    if [ ! -f $source ]; then
        echo "  - The source file does not exist"
        exit 1
    fi
    mkdir -p $TMP
    # if it's a zip, extract it
    if [ ${source: -4} == ".zip" ]; then
        # unzip to a temporary folder
        unzip -q $source -d $TMP
    # otherwise if it's a tar.gz then extract it
    elif [ ${source: -7} == ".tar.gz" ]; then
        # untar to a temporary folder
        tar -xzf $source -C $TMP
    else
        echo "  - The source file is not a zip or a tar.gz"
        exit 1
    fi
}

function copy_data {
    # if the data flag is set, copy the data files
    if [ $data ]; then
        # make sure the relevant folders exist
        mkdir -p "$DATA_RAW"
        mkdir -p "$DATA_PROCESSED"
        mkdir -p "$DATA_IDX"
        # copy the data files
        cp -r $TMP/explain_index/* "$DATA_IDX" && echo "  - Index files copied to $DATA_IDX"
        cp -r $TMP/data/* "$DATA_RAW" && echo "  - Raw data files copied to $DATA_RAW"
        cp -r $TMP/processed/* "$DATA_PROCESSED" && echo "  - Processed data files copied to $DATA_PROCESSED"
    fi
}

function copy_weights {
    # if the weights flag is set, copy the weights files
    if [ $weights ]; then
        # make sure the relevant folders exist
        mkdir -p "$MODEL_WEIGHTS"
        mkdir -p "$EXPLAINER_WEIGHTS"
        # copy the weights files
        cp -r $TMP/checkpoints/* "$MODEL_WEIGHTS" && echo "  - Model weights copied to $MODEL_WEIGHTS"
        cp -r $TMP/explainer_ckpts/* "$EXPLAINER_WEIGHTS" && echo "  - Explainer weights copied to $EXPLAINER_WEIGHTS"
    fi
}

function copy_results {
    # if the results flag is set, copy the results files
    if [ $results ]; then
        # make sure the relevant folders exist
        mkdir -p "$RESULTS"
        mkdir -p "$RESULTS_MCTS"
        # copy the results files
        cp -r $TMP/results/* "$RESULTS" && echo "  - Results copied to $RESULTS"
        cp -r $TMP/saved_mcts_results/* "$RESULTS_MCTS" && echo "  - MCTS checkpoints copied to $RESULTS_MCTS"
    fi
}

# print help if there are no arguments
if [ $# -eq 0 ]; then
    show_help
fi

# Parse CLI
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -s|--source)
            source="$2"
            shift
            shift
            ;;
        -d|--data)
            data=true
            shift
            ;;
        -w|--weights)
            weights=true
            shift
            ;;
        -r|--results)
            results=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# summarise what will happen and propmt the user to continue
echo "______________________________________________________"
echo "Warning the contents of the following directories may be overwritten:"
if [ $data ]; then
    echo "  - $DATA_RAW"
    echo "  - $DATA_PROCESSED"
    echo "  - $DATA_IDX"
fi
if [ $weights ]; then
    echo "  - $MODEL_WEIGHTS"
    echo "  - $EXPLAINER_WEIGHTS"
fi
if [ $results ]; then
    echo "  - $RESULTS"
    echo "  - $RESULTS_MCTS"
fi
echo "______________________________________________________"
read -p "Do you want to continue? (y/n) " -n 1 -r
echo 
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Exiting..."
    exit 1
fi

echo "Extracting files from $source"
# Verify and extract data
verify_and_extract_data
success=$?
# if unsuccessful, delete tmp folder and exit
if [ $success -ne 0 ]; then
    echo "Failed to extract files from $source. Exiting..."
    rm -rf $TMP
    exit 1
fi

echo "Files successfully extracted from $source. Copying files..."
# Copy data, weights, and results
if [ $data ]; then
    echo "Copying data files ..."
    copy_data
    success=$?
    # if unsuccessful, delete tmp folder and exit
    if [ $success -ne 0 ]; then
        echo "Failed to copy data. Exiting..."
        rm -rf $TMP
        exit 1
    fi
    echo "Data files copied"
fi

if [ $weights ]; then
    echo "Copying weights files ..."
    copy_weights
    success=$?
    # if unsuccessful, delete tmp folder and exit
    if [ $success -ne 0 ]; then
        echo "Failed to copy weights. Exiting..."
        rm -rf $TMP
        exit 1
    fi
    echo "Weights files copied"
fi

if [ $results ]; then
    echo "Copying results files ..."
    copy_results
    success=$?
    # if unsuccessful, delete tmp folder and exit
    if [ $success -ne 0 ]; then
        echo "Failed to copy results. Exiting..."
        rm -rf $TMP
        exit 1
    fi
    echo "Results files copied"
fi

# delete tmp folder
rm -rf $TMP
echo "Done"