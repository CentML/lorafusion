#!/bin/bash

MODEL="meta-llama/Llama-3.1-8B-Instruct"
GRADIENT_CHECKPOINTING=False
PROFILE=False

# ===============================
# (3) WIKISUM - 8B Model - 1 GPU
# ===============================

# WIKISUM - Baseline LoRA
# Note: Using per_device_train_batch_size = 1 to avoid OOM
echo " - Running WIKISUM Baseline - ${MODEL} - 1 GPU"
bash scripts/run_single.sh \
    --model "$MODEL" \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --profile "$PROFILE" \
    --dataset_name "wikisum" \
    --seed_idx 0 \
    --nnodes 1 \
    --nproc_per_node 1 \
    --global_batch_size 8 \
    --per_device_train_batch_size 1 \
    --pipeline_parallel_size 1 \
    --gradient_accumulation_steps 8

# WIKISUM - Fused LoRA (LoRAFusion without Multi-LoRA)
echo " - Running WIKISUM LoRAFusion - ${MODEL} - 1 GPU"
bash scripts/run_single.sh \
    --model "$MODEL" \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --profile "$PROFILE" \
    --dataset_name "wikisum" \
    --seed_idx 0 \
    --nnodes 1 \
    --nproc_per_node 1 \
    --global_batch_size 8 \
    --per_device_train_batch_size 1 \
    --pipeline_parallel_size 1 \
    --gradient_accumulation_steps 8 \
    --apply_fused_lora True

# Generate schedule for Multi-LoRA
echo " - Generating WIKISUM Multi-LoRA schedule"
python imbalance_simulator/main_with_datasets.py \
    --dataset_path datasets/dataset_distributions.json \
    --capacity 6144 \
    --num_adapters 4 \
    --num_pipeline_stages 1 \
    --adapter_to_dataset_idx 8,9,10,11 \
    --adapter_to_global_batch_size 8,8,8,8 \
    --fwd_times 1 \
    --bwd_times 1 \
    --output_name "wikisum_cap6144_ada4_pp1_gbz8"

# WIKISUM - Multi-LoRA with LoRAFusion
echo " - Running WIKISUM Multi-LoRA LoRAFusion - ${MODEL} - 1 GPU"
bash scripts/run_single.sh \
    --model "$MODEL" \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --profile "$PROFILE" \
    --dataset_name "wikisum" \
    --multi_lora_dataset_schedule_path "datasets/schedules/wikisum_cap6144_ada4_pp1_gbz8.pkl" \
    --nnodes 1 \
    --nproc_per_node 1 \
    --global_batch_size 1 \
    --per_device_train_batch_size 1 \
    --pipeline_parallel_size 1 \
    --gradient_accumulation_steps 1 \
    --apply_fused_lora True \
    --use_multi_lora True \
    --num_multi_loras 4 \
    --multi_lora_alpha "32.0 32.0 32.0 32.0" \
    --multi_lora_r "16 16 16 16" \
    --multi_lora_dropout "0.1 0.1 0.1 0.1" \
    --multi_lora_max_microbatch_tokens 6144 \
    --multi_lora_global_batch_sizes "8 8 8 8"
