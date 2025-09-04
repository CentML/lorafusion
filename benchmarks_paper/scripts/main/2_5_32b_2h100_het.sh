#!/bin/bash

MODEL="Qwen/Qwen2.5-32B-Instruct"
GRADIENT_CHECKPOINTING=True
PROFILE=False

# ===============================
# (5) HETEROGENEOUS - 32B Model - 2 GPU
# ===============================

# HETEROGENEOUS - Multi-LoRA (mLoRA) Baseline
echo " - Running Heterogeneous mLoRA - 2 GPU"
# Note: Microbatch size doesn't matter for this baseline
echo " - Generating Heterogeneous mLoRA baseline schedule"
python imbalance_simulator/main_with_datasets.py \
    --dataset_path datasets/dataset_distributions.json \
    --num_adapters 4 \
    --adapter_to_dataset_idx 0,4,8,12 \
    --adapter_to_global_batch_size 8,8,8,8 \
    --perform_mlora_schedule_only \
    --output_name "interleaved_ada4_gbz8"

# Run baseline Multi-LoRA
bash scripts/run_single.sh \
    --model "$MODEL" \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --profile "$PROFILE" \
    --dataset_path "datasets/schedules/baseline_dataset_mlora_interleaved_ada4_gbz8.json" \
    --dataset_name "heterogeneous" \
    --seed_idx 0 \
    --nnodes 1 \
    --nproc_per_node 2 \
    --global_batch_size 4 \
    --per_device_train_batch_size 4 \
    --pipeline_parallel_size 2 \
    --gradient_accumulation_steps 1 \
    --benchmark_baseline_mlora_schedule True \
    --multi_lora_global_batch_sizes "8 8 8 8" \
    --max_steps 100

# HETEROGENEOUS - LoRAFusion with Pipeline Parallelism
echo " - Running Heterogeneous LoRAFusion - 2 GPU"

# Generate schedule for Multi-LoRA (mixed datasets)
echo " - Generating Heterogeneous Multi-LoRA schedule"
python imbalance_simulator/main_with_datasets.py \
    --dataset_path datasets/dataset_distributions.json \
    --capacity 7168 \
    --num_adapters 4 \
    --num_pipeline_stages 2 \
    --adapter_to_dataset_idx 0,4,8,12 \
    --adapter_to_global_batch_size 8,8,8,8 \
    --fwd_times 1,1 \
    --bwd_times 1,1 \
    --output_name "heterogeneous_cap7168_ada4_pp2_gbz8"

# Run LoRAFusion with Multi-LoRA
bash scripts/run_single.sh \
    --model "$MODEL" \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --profile "$PROFILE" \
    --dataset_name "heterogeneous" \
    --multi_lora_dataset_schedule_path "datasets/schedules/heterogeneous_cap7168_ada4_pp2_gbz8.pkl" \
    --nnodes 1 \
    --nproc_per_node 2 \
    --global_batch_size 1 \
    --per_device_train_batch_size 1 \
    --pipeline_parallel_size 2 \
    --gradient_accumulation_steps 1 \
    --apply_fused_lora True \
    --use_multi_lora True \
    --num_multi_loras 4 \
    --multi_lora_alpha "32.0 32.0 32.0 32.0" \
    --multi_lora_r "16 16 16 16" \
    --multi_lora_dropout "0.1 0.1 0.1 0.1" \
    --multi_lora_max_microbatch_tokens 7168 \
    --multi_lora_global_batch_sizes "8 8 8 8" \
    --max_steps 100
