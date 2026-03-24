#!/bin/bash
#SBATCH --job-name=attr_evidence
#SBATCH --output=slurm-%j.out
#SBATCH --partition=gpu
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:4

export HF_HUB_CACHE=/shared/4/models/

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
TP="${TP:-4}"
MAX_USERS="${MAX_USERS:-100}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Job started: $(date) ==="
echo "Model: $MODEL | TP: $TP | Max users: $MAX_USERS"

python "$SCRIPT_DIR/generate_attr_evidence.py" \
    --model        "$MODEL" \
    --tensor-parallel-size "$TP" \
    --max-users    "$MAX_USERS"

echo "=== Job finished: $(date) ==="
