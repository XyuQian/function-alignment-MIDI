#!/bin/bash

# Create a directory for Slurm logs if it doesn't exist
mkdir -p logs

#SBATCH --job-name=midi-perf-2-score
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Activate your Python environment (e.g., conda or venv)
source activate midiLM

# Navigate to your project directory if your script is not in the submission directory
# cd causal_group/peiyuan.zhu/xiaoyu/function-alignment-MIDI

export PYTHONPATH=.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create logs directory if it doesn't exist
mkdir -p logs

# Set the node's hostname as the master address for DDP.
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

echo "Starting training on with single NVIDIA A100 GPU..."
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"

# --- Launch Training Script ---
python -m shoelace.actual_shoelace.midi_train