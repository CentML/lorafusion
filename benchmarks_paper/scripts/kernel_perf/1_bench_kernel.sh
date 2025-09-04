#!/bin/bash

dim_list=(4096 5120 8192)

# ===============================================
# Check linear lora and fused linear lora performance
# ===============================================
for dim in "${dim_list[@]}"; do
    echo "Running kernel only for dim=${dim}"
    python bench_lora_kernels.py \
        --in_features ${dim} \
        --out_features ${dim} \
        --r 16 \
        --seq_len 256 \
        --batch_sizes "8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32"
done

# ===============================================
# Check fused linear multi lora performance
# ===============================================
for dim in "${dim_list[@]}"; do
    for batch_size in 8 10 12 14 16 18 20 22 24 26 28 30 32; do
        echo "Running kernel only for dim=${dim} and batch_size=${batch_size}"
        python bench_lora_kernels.py \
            --in_features ${dim} \
            --out_features ${dim} \
            --r 16 \
            --seq_len 256 \
            --batch_sizes ${batch_size} \
            --not-bench-flash-lora \
            --bench-multi-lora-only
        sleep 2
    done
done

# ===============================================
# Notes (Some runs will trigger unexpected errors)
# If so, please comment out the above for-loop and fill them in using the following commands
# e.g.,
# ===============================================
# python bench_lora_kernels.py \
#     --in_features 4096 \
#     --out_features 4096 \
#     --r 16 \
#     --seq_len 256 \
#     --batch_sizes 32 \
#     --not-bench-flash-lora \
#     --bench-multi-lora-only

# python bench_lora_kernels.py \
#     --in_features 4096 \
#     --out_features 4096 \
#     --r 16 \
#     --seq_len 256 \
#     --batch_sizes 32
