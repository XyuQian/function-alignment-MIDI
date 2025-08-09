#!/bin/bash

# Navigate to your project directory if your script is not in the submission directory
# cd causal_group/peiyuan.zhu/xiaoyu/function-alignment-MIDI

export PYTHONPATH=.

export OMP_NUM_THREADS=16

# Set the node's hostname as the master address for DDP.
export MASTER_ADDR=$(hostname)
export MASTER_PORT=36900

echo "Starting training on a single node with 4 NVIDIA H100 GPUs..."
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"

# --- Launch Training Script ---
echo "Executing torchrun command now..."
torchrun --nnodes=1 --nproc_per_node=4 shoelace/actual_shoelace/midi_train_DDP.py \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --epochs 100 \
    --suffix perf_2_score \
    --num_workers 4 \
    --experiment_folder exp \
    --exp_name perf_2_score \
    --task_type midi_conversion \
    --n_prompts 5 \
    --mask_config '{"ScoreLM": True, "PerformanceLM": False}'


echo "Job finished."
