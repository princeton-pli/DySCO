#!/bin/bash

# test on 20 samples for path_walking with Qwen3-8B

echo "======= Running Vanilla generation ======="
python run_eval.py --model "models/Qwen3-8B" --decoding_method "flash" --dataset path_walking_16k --max_tokens 128 --max_model_len 131072 --test_size 20
echo "Expected accuracy: around 25.0"

echo "======= Running DySCO generation ======="
python run_eval.py --model "models/Qwen3-8B" --decoding_method "dysco" --dataset path_walking_16k --max_tokens 128 --max_model_len 131072 --test_size 20 --dysco_cfgs_path "dysco_cfgs/qwen3_8b.yaml" --dysco_top_p 0 --dysco_top_k 1024 --dysco_strength 2.5
echo "Expected accuracy: around 40.0"