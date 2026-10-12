#!/bin/bash
# ./run_vllm.sh <gpus> <log name> [vllm_bench.py args]
export HIP_VISIBLE_DEVICES=$1 CUDA_VISIBLE_DEVICES=$1
mkdir -p logs
python vllm_bench.py "${@:3}" > logs/$2.log 2>&1
grep -E "RESULT|IDENTITY" logs/$2.log
