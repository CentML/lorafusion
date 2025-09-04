#!/bin/bash

MODEL="Qwen/Qwen2.5-32B-Instruct"
GRADIENT_CHECKPOINTING=True
PROFILE=False

# ===============================
# (1) XSUM - 32B Model - 2 GPU
# ===============================

# XSUM - FSDP (Fully Sharded Data Parallel)
echo " - Running XSUM FSDP - 2 GPU"
bash scripts/run_single.sh \
    --model "$MODEL" \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --profile "$PROFILE" \
    --dataset_name "xsum" \
    --seed_idx 0 \
    --nnodes 1 \
    --nproc_per_node 2 \
    --global_batch_size 8 \
    --per_device_train_batch_size 4 \
    --pipeline_parallel_size 1 \
    --gradient_accumulation_steps 1 \
    --max_steps 30 \
    --use_fsdp True

# XSUM - Pipeline Parallelism (PP)
echo " - Running XSUM PP - 2 GPU"
bash scripts/run_single.sh \
    --model "$MODEL" \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --profile "$PROFILE" \
    --dataset_name "xsum" \
    --seed_idx 0 \
    --nnodes 1 \
    --nproc_per_node 2 \
    --global_batch_size 8 \
    --per_device_train_batch_size 4 \
    --pipeline_parallel_size 2 \
    --gradient_accumulation_steps 2 \
    --max_steps 30

# XSUM - Multi-LoRA (mLoRA) Baseline
echo " - Running XSUM mLoRA - 2 GPU"
bash scripts/run_single.sh \
    --model "$MODEL" \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --profile "$PROFILE" \
    --dataset_name "xsum" \
    --seed_idx 0 \
    --nnodes 1 \
    --nproc_per_node 2 \
    --global_batch_size 8 \
    --per_device_train_batch_size 8 \
    --pipeline_parallel_size 2 \
    --gradient_accumulation_steps 1 \
    --benchmark_baseline_mlora_schedule True \
    --multi_lora_global_batch_sizes "8 8 8 8" \
    --max_steps 100

# XSUM - LoRAFusion with Pipeline Parallelism
echo " - Running XSUM LoRAFusion - 2 GPU"

# Generate schedule for Multi-LoRA
echo " - Generating XSUM Multi-LoRA schedule"
python imbalance_simulator/main_with_datasets.py \
    --dataset_path datasets/dataset_distributions.json \
    --capacity 4096 \
    --num_adapters 4 \
    --num_pipeline_stages 2 \
    --adapter_to_dataset_idx 0,1,2,3 \
    --adapter_to_global_batch_size 8,8,8,8 \
    --fwd_times 1,1 \
    --bwd_times 1,1 \
    --output_name "xsum_cap4096_ada4_pp2_gbz8"

# Run LoRAFusion with Multi-LoRA
bash scripts/run_single.sh \
    --model "$MODEL" \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --profile "$PROFILE" \
    --dataset_name "xsum" \
    --multi_lora_dataset_schedule_path "datasets/schedules/xsum_cap4096_ada4_pp2_gbz8.pkl" \
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
    --multi_lora_max_microbatch_tokens 4096 \
    --multi_lora_global_batch_sizes "8 8 8 8" \
    --max_steps 100
