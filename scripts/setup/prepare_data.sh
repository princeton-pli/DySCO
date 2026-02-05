#!/bin/bash
set -e

HF_BASE="https://huggingface.co/datasets/PrincetonPLI/DySCO/resolve/main"
TARGET_DIR="data_eval/longproc"

if [ -d "$TARGET_DIR" ]; then
    echo "data_eval/longproc/ already exists, skipping download."
    exit 0
fi

echo "Downloading longproc evaluation data from HuggingFace..."
wget -q --show-progress "$HF_BASE/data_eval_longproc.zip" -O data_eval_longproc.zip

echo "Extracting..."
mkdir -p data_eval
unzip -q data_eval_longproc.zip -d data_eval/
rm data_eval_longproc.zip

echo "Done. Data is ready at $TARGET_DIR/"
