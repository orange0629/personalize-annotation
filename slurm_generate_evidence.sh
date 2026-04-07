#!/bin/bash
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA40x4
#SBATCH --account=bfuj-delta-gpu
#SBATCH --job-name=run_evaluation_llama_conversations
#SBATCH --time=24:00:00
#SBATCH --constraint="scratch"
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
### GPU options ###
#SBATCH --gpu-bind=closest
#SBATCH --gpus-per-node=2
#SBATCH --mail-user=lechenz3@illinois.edu
#SBATCH --mail-type="BEGIN,END"


export HF_HOME=/work/hdd/bfuj/lzhang49/huggingface
source ~/.bashrc

conda activate /projects/bfuj/lzhang49/llm-personalization/env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH


MODEL="${MODEL:-Qwen/Qwen3-32B}"
TP="${TP:-2}"
MAX_USERS="${MAX_USERS:-100}"

echo "=== Job started: $(date) ==="
echo "Model: $MODEL | TP: $TP | Max users: $MAX_USERS"

python "generate_attr_evidence.py" \
    --model        "$MODEL" \
    --tensor-parallel-size "$TP" \
    --max-users    "$MAX_USERS" \
    --max-model-len 40960 \
    --force

echo "=== Job finished: $(date) ==="
