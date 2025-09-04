#!/bin/bash
# ===============================================
# NCU profile
# ===============================================

export CUDA_LAUNCH_BLOCKING=1

features=(4096 5120 8192)

# Create base directory for all reports
mkdir -p ncu_reports

for feat in "${features[@]}"; do
  # Create dimension-specific directory
  mkdir -p "ncu_reports/dim_${feat}x${feat}"

  echo "Running NCU profiling for dimension ${feat}x${feat}"

  # ================================================
  # 1. Raw linear
  echo " *** Running Raw Linear *** "
  ncu \
    --nvtx \
    --nvtx-include "RawLinear" \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum \
    --section MemoryWorkloadAnalysis \
    -f -o "ncu_reports/dim_${feat}x${feat}/raw_linear_ncu" \
    python bench_lora_kernels.py \
    --in_features ${feat} \
    --out_features ${feat} \
    --r 16 \
    --seq_len 256 \
    --batch_sizes 32 \
    --ncu-profile

  sleep 3

  ncu --import "ncu_reports/dim_${feat}x${feat}/raw_linear_ncu.ncu-rep" \
    --page raw \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum \
    --csv | tee -a "ncu_reports/dim_${feat}x${feat}/raw_linear_ncu.csv"
  # ================================================

  # ================================================
  # 2. Raw LoRA
  echo " *** Running Raw LoRA *** "
  ncu \
    --nvtx \
    --nvtx-include "RawLoRA" \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum \
    --section MemoryWorkloadAnalysis \
    -f -o "ncu_reports/dim_${feat}x${feat}/raw_lora_ncu" \
    python bench_lora_kernels.py \
    --in_features ${feat} \
    --out_features ${feat} \
    --r 16 \
    --seq_len 256 \
    --batch_sizes 32 \
    --ncu-profile

  sleep 3

  ncu --import "ncu_reports/dim_${feat}x${feat}/raw_lora_ncu.ncu-rep" \
    --page raw \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum \
    --csv | tee -a "ncu_reports/dim_${feat}x${feat}/raw_lora_ncu.csv"
  # ================================================

  # ================================================
  # 3. Fused linear lora
  echo " *** Running Fused Linear LoRA *** "
  ncu \
    --nvtx \
    --nvtx-include "FusedLoRA" \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum \
    --section MemoryWorkloadAnalysis \
    -f -o "ncu_reports/dim_${feat}x${feat}/fused_linear_lora_ncu" \
    python bench_lora_kernels.py \
    --in_features ${feat} \
    --out_features ${feat} \
    --r 16 \
    --seq_len 256 \
    --batch_sizes 32 \
    --ncu-profile

  sleep 3

  ncu --import "ncu_reports/dim_${feat}x${feat}/fused_linear_lora_ncu.ncu-rep" \
    --page raw \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum \
    --csv | tee -a "ncu_reports/dim_${feat}x${feat}/fused_linear_lora_ncu.csv"
  # ================================================

  # ================================================
  # 4. Fused linear multi lora
  echo " *** Running Fused Linear Multi LoRA *** "
  ncu \
    --nvtx \
    --nvtx-include "FusedMultiLoRA" \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum \
    --section MemoryWorkloadAnalysis \
    -f -o "ncu_reports/dim_${feat}x${feat}/fused_linear_multi_lora_ncu" \
    python bench_lora_kernels.py \
    --in_features ${feat} \
    --out_features ${feat} \
    --r 16 \
    --seq_len 256 \
    --batch_sizes 32 \
    --not-bench-flash-lora \
    --bench-multi-lora-only \
    --ncu-profile

  sleep 3

  ncu --import "ncu_reports/dim_${feat}x${feat}/fused_linear_multi_lora_ncu.ncu-rep" \
    --page raw \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum \
    --csv | tee -a "ncu_reports/dim_${feat}x${feat}/fused_linear_multi_lora_ncu.csv"
  # ================================================

  echo "Completed NCU profiling for dimension ${feat}x${feat}"
done

# Run summarization for each group after all benchmarks are done
echo "Running summarization scripts for all benchmark groups"

# Run summarization for each dimension
for feat in "${features[@]}"; do
  echo "Summarizing results for dimension ${feat}x${feat}"
  python scripts/kernel_perf/summarize_kernel_ncu_profile.py "ncu_reports/dim_${feat}x${feat}"
done

echo "NCU profiling and summarization complete"
