#!/bin/bash

MODEL="meta-llama/Llama-3.1-70B-Instruct"
GRADIENT_CHECKPOINTING=True
PROFILE=False

# ===============================
# (2) CNN_DAILYMAIL - 70B Model - 4 GPU
# ===============================

# CNN_DAILYMAIL - FSDP (Fully Sharded Data Parallel)
echo "Running CNN_DAILYMAIL FSDP - 4 GPU"
bash scripts/run_single.sh \
    --model "$MODEL" \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --profile "$PROFILE" \
    --dataset_name "cnn_dailymail" \
    --seed_idx 0 \
    --nnodes 1 \
    --nproc_per_node 4 \
    --global_batch_size 16 \
    --per_device_train_batch_size 4 \
    --pipeline_parallel_size 1 \
    --gradient_accumulation_steps 1 \
    --max_steps 30 \
    --use_fsdp True

# CNN_DAILYMAIL - Pipeline Parallelism (PP)
echo "Running CNN_DAILYMAIL PP - 4 GPU"
bash scripts/run_single.sh \
    --model "$MODEL" \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --profile "$PROFILE" \
    --dataset_name "cnn_dailymail" \
    --seed_idx 0 \
    --nnodes 1 \
    --nproc_per_node 4 \
    --global_batch_size 16 \
    --per_device_train_batch_size 4 \
    --pipeline_parallel_size 4 \
    --gradient_accumulation_steps 4 \
    --max_steps 30

# CNN_DAILYMAIL - Multi-LoRA (mLoRA) Baseline
echo "Running CNN_DAILYMAIL mLoRA - 4 GPU"
bash scripts/run_single.sh \
    --model "$MODEL" \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --profile "$PROFILE" \
    --dataset_name "cnn_dailymail" \
    --seed_idx 0 \
    --nnodes 1 \
    --nproc_per_node 4 \
    --global_batch_size 16 \
    --per_device_train_batch_size 4 \
    --pipeline_parallel_size 4 \
    --gradient_accumulation_steps 4 \
    --benchmark_baseline_mlora_schedule True \
    --multi_lora_global_batch_sizes "16 16 16 16" \
    --max_steps 100

# CNN_DAILYMAIL - LoRAFusion with Pipeline Parallelism
echo " - Generating CNN_DAILYMAIL Multi-LoRA schedule"
python imbalance_simulator/main_with_datasets.py \
    --dataset_path datasets/dataset_distributions.json \
    --capacity 6144 \
    --num_adapters 4 \
    --num_pipeline_stages 4 \
    --adapter_to_dataset_idx 4,5,6,7 \
    --adapter_to_global_batch_size 16,16,16,16 \
    --fwd_times 1,1,1,1 \
    --bwd_times 1,1,1,1 \
    --output_name "cnn_dailymail_cap6144_ada4_pp4_gbz16"

# Run LoRAFusion with Multi-LoRA
echo " - Running CNN_DAILYMAIL LoRAFusion - 4 GPU"
bash scripts/run_single.sh \
    --model "$MODEL" \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --profile "$PROFILE" \
    --dataset_name "cnn_dailymail" \
    --multi_lora_dataset_schedule_path "datasets/schedules/cnn_dailymail_cap6144_ada4_pp4_gbz16.pkl" \
    --nnodes 1 \
    --nproc_per_node 4 \
    --global_batch_size 1 \
    --per_device_train_batch_size 1 \
    --pipeline_parallel_size 4 \
    --gradient_accumulation_steps 1 \
    --apply_fused_lora True \
    --use_multi_lora True \
    --num_multi_loras 4 \
    --multi_lora_alpha "32.0 32.0 32.0 32.0" \
    --multi_lora_r "16 16 16 16" \
    --multi_lora_dropout "0.1 0.1 0.1 0.1" \
    --multi_lora_max_microbatch_tokens 6144 \
    --multi_lora_global_batch_sizes "16 16 16 16" \
    --max_steps 100
