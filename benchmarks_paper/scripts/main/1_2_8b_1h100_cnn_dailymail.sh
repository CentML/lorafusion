#!/bin/bash

MODEL="meta-llama/Llama-3.1-8B-Instruct"
GRADIENT_CHECKPOINTING=False
PROFILE=False

# ===============================
# (2) CNN_DAILYMAIL - 8B Model - 1 GPU
# ===============================

# CNN_DAILYMAIL - Baseline LoRA
echo " - Running CNN_DAILYMAIL Baseline - ${MODEL} - 1 GPU"
bash scripts/run_single.sh \
    --model "$MODEL" \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --profile "$PROFILE" \
    --dataset_name "cnn_dailymail" \
    --seed_idx 0 \
    --nnodes 1 \
    --nproc_per_node 1 \
    --global_batch_size 8 \
    --per_device_train_batch_size 4 \
    --pipeline_parallel_size 1 \
    --gradient_accumulation_steps 2

# CNN_DAILYMAIL - Fused LoRA (LoRAFusion without Multi-LoRA)
echo " - Running CNN_DAILYMAIL LoRAFusion - ${MODEL} - 1 GPU"
bash scripts/run_single.sh \
    --model "$MODEL" \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --profile "$PROFILE" \
    --dataset_name "cnn_dailymail" \
    --seed_idx 0 \
    --nnodes 1 \
    --nproc_per_node 1 \
    --global_batch_size 8 \
    --per_device_train_batch_size 4 \
    --pipeline_parallel_size 1 \
    --gradient_accumulation_steps 2 \
    --apply_fused_lora True

# Generate schedule for Multi-LoRA
echo " - Generating CNN_DAILYMAIL Multi-LoRA schedule"
python imbalance_simulator/main_with_datasets.py \
    --dataset_path datasets/dataset_distributions.json \
    --capacity 6144 \
    --num_adapters 4 \
    --num_pipeline_stages 1 \
    --adapter_to_dataset_idx 4,5,6,7 \
    --adapter_to_global_batch_size 8,8,8,8 \
    --fwd_times 1 \
    --bwd_times 1 \
    --output_name "cnn_dailymail_cap6144_ada4_pp1_gbz8"

# CNN_DAILYMAIL - Multi-LoRA with LoRAFusion
echo " - Running CNN_DAILYMAIL Multi-LoRA LoRAFusion - ${MODEL} - 1 GPU"
bash scripts/run_single.sh \
    --model "$MODEL" \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --profile "$PROFILE" \
    --dataset_name "cnn_dailymail" \
    --multi_lora_dataset_schedule_path "datasets/schedules/cnn_dailymail_cap6144_ada4_pp1_gbz8.pkl" \
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
