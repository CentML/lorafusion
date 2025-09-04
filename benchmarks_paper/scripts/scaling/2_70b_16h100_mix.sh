#!/bin/bash

MODEL="meta-llama/Llama-3.1-70B-Instruct"
GRADIENT_CHECKPOINTING=True
PROFILE=False

MASTER_ADDR="192.168.0.80"
MASTER_PORT=29500
NODE_RANK=0

# Please update the NCCL_SOCKET_IFNAME and GLOO_SOCKET_IFNAME to the correct network interface name.
export NCCL_SOCKET_IFNAME="zhanda-tmp-mult"
export GLOO_SOCKET_IFNAME="zhanda-tmp-mult"

# For scaling results, we only need to run extra 5 experiments.
# For Megatron-LM-FSDP (Job Scaling), mLoRA (Job Scaling), and LoRAFusion (Job Scaling),
# we can directly scale the throughput from 1x4 to 1x8 or 2x8 by multiplying the throughput 
# of 1x4 by 2 or 4. Because we assume the performance does not change much across global batch sizes.

# MIX - PP - 4 GPUs (GBS 64)
bash scripts/run_single.sh \
  --model "$MODEL" \
  --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
  --profile "$PROFILE" \
  --dataset_name "mix" \
  --seed_idx 0 \
  --nnodes 1 \
  --nproc_per_node 4 \
  --global_batch_size 64 \
  --per_device_train_batch_size 4 \
  --pipeline_parallel_size 4 \
  --gradient_accumulation_steps 16 \
  --max_steps 10

# MIX - FSDP - 2x8 GPU (GBS 64)
bash scripts/run_single.sh \
  --model "$MODEL" \
  --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
  --profile "$PROFILE" \
  --dataset_name "mix" \
  --seed_idx 0 \
  --nnodes 2 \
  --nproc_per_node 8 \
  --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --global_batch_size 64 \
  --per_device_train_batch_size 4 \
  --pipeline_parallel_size 1 \
  --gradient_accumulation_steps 1 \
  --max_steps 30 \
  --use_fsdp True

# MIX - PP - 2x8 GPU (GBS 64)
bash scripts/run_single.sh \
    --model "$MODEL" \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --profile "$PROFILE" \
    --dataset_name "mix" \
    --seed_idx 0 \
    --nnodes 2 \
    --nproc_per_node 8 \
    --node_rank "$NODE_RANK" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    --global_batch_size 64 \
    --per_device_train_batch_size 4 \
    --pipeline_parallel_size 4 \
    --gradient_accumulation_steps 4 \
    --max_steps 30

# MIX - mLoRA - 2x8 GPU (GBS 64)
bash scripts/run_single.sh \
    --model "$MODEL" \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --profile "$PROFILE" \
    --dataset_name "mix" \
    --seed_idx 0 \
    --nnodes 2 \
    --nproc_per_node 8 \
    --node_rank "$NODE_RANK" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    --global_batch_size 64 \
    --per_device_train_batch_size 4 \
    --pipeline_parallel_size 4 \
    --gradient_accumulation_steps 4 \
    --benchmark_baseline_mlora_schedule True \
    --multi_lora_global_batch_sizes "64 64 64 64" \
    --max_steps 100

# MIX - LoRAFusion - PP - 2x8 GPU (GBS 64)
python imbalance_simulator/main_with_datasets.py \
    --dataset_path datasets/dataset_distributions.json \
    --capacity 6144 \
    --num_adapters 4 \
    --num_pipeline_stages 8 \
    --adapter_to_dataset_idx 12,13,14,15 \
    --adapter_to_global_batch_size 64,64,64,64 \
    --fwd_times 1,1,1,1,1,1,1,1 \
    --bwd_times 1,1,1,1,1,1,1,1 \
    --output_name "mix_cap6144_ada4_pp8_gbz64"

bash scripts/run_single.sh \
    --model "$MODEL" \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --profile "$PROFILE" \
    --dataset_name "mix" \
    --multi_lora_dataset_schedule_path "datasets/schedules/mix_cap6144_ada4_pp8_gbz64.pkl" \
    --nnodes 2 \
    --nproc_per_node 8 \
    --node_rank "$NODE_RANK" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    --global_batch_size 4 \
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
    --multi_lora_global_batch_sizes "64 64 64 64" \
    --max_steps 100
