#!/bin/bash

MODEL="meta-llama/Llama-3.1-70B-Instruct"
GRADIENT_CHECKPOINTING=True
PROFILE=False

# ===============================
# (5) HETEROGENEOUS - 70B Model - 4 GPU
# ===============================

# HETEROGENEOUS - Multi-LoRA (mLoRA) Baseline
# Note: Microbatch size doesn't matter for this baseline
echo "Running Het - mLoRA - 4 GPU"
python imbalance_simulator/main_with_datasets.py \
    --dataset_path datasets/dataset_distributions.json \
    --num_adapters 4 \
    --adapter_to_dataset_idx 0,4,8,12 \
    --adapter_to_global_batch_size 16,16,16,16 \
    --perform_mlora_schedule_only \
    --output_name "interleaved_ada4_gbz16"

# echo "Running HET mLoRA - 4 GPU"
bash scripts/run_single.sh \
    --model "$MODEL" \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --profile "$PROFILE" \
    --dataset_path "datasets/schedules/baseline_dataset_mlora_interleaved_ada4_gbz16.json" \
    --dataset_name "heterogeneous" \
    --seed_idx 0 \
    --nnodes 1 \
    --nproc_per_node 4 \
    --global_batch_size 16 \
    --per_device_train_batch_size 2 \
    --pipeline_parallel_size 4 \
    --gradient_accumulation_steps 8 \
    --benchmark_baseline_mlora_schedule True \
    --multi_lora_global_batch_sizes "16 16 16 16" \
    --max_steps 100

# HETEROGENEOUS - LoRAFusion with Pipeline Parallelism
echo " - Generating Heterogeneous Multi-LoRA schedule"
python imbalance_simulator/main_with_datasets.py \
    --dataset_path datasets/dataset_distributions.json \
    --capacity 6144 \
    --num_adapters 4 \
    --num_pipeline_stages 4 \
    --adapter_to_dataset_idx 0,4,8,12 \
    --adapter_to_global_batch_size 16,16,16,16 \
    --fwd_times 1,1,1,1 \
    --bwd_times 1,1,1,1 \
    --output_name "het_cap6144_ada4_pp4_gbz16"

# Run LoRAFusion with Multi-LoRA
echo " - Running Heterogeneous LoRAFusion - 4 GPU"
bash scripts/run_single.sh \
  --model "$MODEL" \
  --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
  --profile "$PROFILE" \
  --dataset_name "heterogeneous" \
  --multi_lora_dataset_schedule_path "datasets/schedules/het_cap6144_ada4_pp4_gbz16.pkl" \
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
  --multi_lora_max_microbatch_tokens 7168 \
  --multi_lora_global_batch_sizes "16 16 16 16" \
  --max_steps 100
