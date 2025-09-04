#!/bin/bash

models=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen2.5-32B-Instruct"
  "meta-llama/Llama-3.1-70B-Instruct"
)

batch_sizes="4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19"

for model in "${models[@]}"; do
  python bench_transformer_layer.py \
    --model_name_or_path "$model" \
    --seq_len 512 \
    --batch_sizes "$batch_sizes" \
    --folder_to_save "layer-results/"

  sleep 1

  python bench_transformer_layer.py \
    --model_name_or_path "$model" \
    --seq_len 512 \
    --batch_sizes "$batch_sizes" \
    --folder_to_save "layer-results/" \
    --apply_fused_lora

  sleep 1

  python bench_transformer_layer.py \
    --model_name_or_path "$model" \
    --seq_len 512 \
    --batch_sizes "$batch_sizes" \
    --folder_to_save "layer-results/" \
    --apply_fused_lora \
    --use_multi_lora \
    --num_multi_loras 4 \
    --multi_lora_alpha "32.0 32.0 32.0 32.0" \
    --multi_lora_r "16 16 16 16" \
    --multi_lora_dropout "0.1 0.1 0.1 0.1" \
    --multi_lora_global_batch_sizes "1 1 1 1"

  sleep 1

done
