#!/bin/bash

# Default values
# Hardware
NNODES=1
NPROC_PER_NODE=4
# Environment variables for distributed training
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29500}
# Model Arguments
MODEL="meta-llama/Llama-3.1-8B-Instruct"
LIGER_KERNEL_LEVEL="all"
TORCH_COMPILE_LEVEL="disable"
NUM_LAYERS_FOR_DEBUGGING=-1
LORA_R=16
LORA_ALPHA=16
LORA_DROPOUT=0.1
DISABLE_OPTIMIZER_FOR_DEBUGGING=False
APPLY_FUSED_LORA=False
MERGE_QKV_PROJ=False
USE_MULTI_LORA=False
NUM_MULTI_LORAS=4
MULTI_LORA_ALPHA="32.0 32.0 32.0 32.0"
MULTI_LORA_R="16 16 16 16"
MULTI_LORA_DROPOUT="0.1 0.1 0.1 0.1"
MULTI_LORA_MAX_MICROBATCH_TOKENS=4096
MULTI_LORA_GLOBAL_BATCH_SIZES="8 8 8 8"
# Training Arguments
GLOBAL_BATCH_SIZE=16
PER_DEVICE_BATCH_SIZE=1
USE_FSDP=False
PIPELINE_PARALLEL_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
GRADIENT_CHECKPOINTING=False
GRADIENT_CHECKPOINTING_LAYERS=None
PROFILE=False
USE_TIMERS=False
BENCHMARK_BASELINE_MLORA_SCHEDULE=False
# Mock Data Arguments
DATASET_PATH="datasets/dataset_distributions.json"
DATASET_NAME="cnn_dailymail"
NUM_SAMPLES=1000
SEED_IDX=0
PERMUTATION_IDX=0
MULTI_LORA_DATASET_SCHEDULE_PATH="datasets/schedules/schedule.pkl"
USE_DUMMY_FIXED_LENGTH_DATASET=False
DUMMY_FIXED_LENGTH_DATASET_LENGTH=1024
# Other Arguments
LOG_FOLDER_NAME="benchmark"
MAX_SEQ_LENGTH=2048
MAX_STEPS=100
USE_CUDA_GRAPH=False
SAVE_MODEL=False
EVAL_STRATEGY="no"
# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
  --nnodes)
    NNODES="$2"
    shift 2
    ;;
  --nproc_per_node)
    NPROC_PER_NODE="$2"
    shift 2
    ;;
  --node_rank)
    NODE_RANK="$2"
    shift 2
    ;;
  --master_addr)
    MASTER_ADDR="$2"
    shift 2
    ;;
  --master_port)
    MASTER_PORT="$2"
    shift 2
    ;;
  --model)
    MODEL="$2"
    shift 2
    ;;
  --num_layers_for_debugging)
    NUM_LAYERS_FOR_DEBUGGING="$2"
    shift 2
    ;;
  --disable_optimizer_for_debugging)
    DISABLE_OPTIMIZER_FOR_DEBUGGING="$2"
    shift 2
    ;;
  --apply_fused_lora)
    APPLY_FUSED_LORA="$2"
    shift 2
    ;;
  --merge_qkv_proj)
    MERGE_QKV_PROJ="$2"
    shift 2
    ;;
  --dataset_path)
    DATASET_PATH="$2"
    shift 2
    ;;
  --use_multi_lora)
    USE_MULTI_LORA="$2"
    shift 2
    ;;
  --num_multi_loras)
    NUM_MULTI_LORAS="$2"
    shift 2
    ;;
  --multi_lora_alpha)
    MULTI_LORA_ALPHA="$2"
    shift 2
    ;;
  --multi_lora_r)
    MULTI_LORA_R="$2"
    shift 2
    ;;
  --multi_lora_dropout)
    MULTI_LORA_DROPOUT="$2"
    shift 2
    ;;
  --multi_lora_max_microbatch_tokens)
    MULTI_LORA_MAX_MICROBATCH_TOKENS="$2"
    shift 2
    ;;
  --multi_lora_global_batch_sizes)
    MULTI_LORA_GLOBAL_BATCH_SIZES="$2"
    shift 2
    ;;
  --dataset_name)
    DATASET_NAME="$2"
    shift 2
    ;;
  --num_samples)
    NUM_SAMPLES="$2"
    shift 2
    ;;
  --seed_idx)
    SEED_IDX="$2"
    shift 2
    ;;
  --permutation_idx)
    PERMUTATION_IDX="$2"
    shift 2
    ;;
  --max_seq_length)
    MAX_SEQ_LENGTH="$2"
    shift 2
    ;;
  --epochs)
    EPOCHS="$2"
    shift 2
    ;;
  --packing)
    PACKING="$2"
    shift 2
    ;;
  --global_batch_size)
    GLOBAL_BATCH_SIZE="$2"
    shift 2
    ;;
  --gradient_accum_steps | --gradient_accumulation_steps)
    GRADIENT_ACCUMULATION_STEPS="$2"
    shift 2
    ;;
  --per_device_batch_size | --per_device_train_batch_size)
    PER_DEVICE_BATCH_SIZE="$2"
    shift 2
    ;;
  --use_fsdp)
    USE_FSDP="$2"
    shift 2
    ;;
  --pipeline_parallel_size)
    PIPELINE_PARALLEL_SIZE="$2"
    shift 2
    ;;
  --gradient_checkpointing)
    GRADIENT_CHECKPOINTING="$2"
    shift 2
    ;;
  --gradient_checkpointing_layers)
    GRADIENT_CHECKPOINTING_LAYERS="$2"
    shift 2
    ;;
  --lora_r)
    LORA_R="$2"
    shift 2
    ;;
  --lora_alpha)
    LORA_ALPHA="$2"
    shift 2
    ;;
  --lora_dropout)
    LORA_DROPOUT="$2"
    shift 2
    ;;
  --max_steps)
    MAX_STEPS="$2"
    shift 2
    ;;
  --eval_strategy)
    EVAL_STRATEGY="$2"
    shift 2
    ;;
  --liger_kernel_level)
    LIGER_KERNEL_LEVEL="$2"
    shift 2
    ;;
  --torch_compile_level)
    TORCH_COMPILE_LEVEL="$2"
    shift 2
    ;;
  --use_cuda_graph)
    USE_CUDA_GRAPH="$2"
    shift 2
    ;;
  --save_model)
    SAVE_MODEL="$2"
    shift 2
    ;;
  --multi_lora_dataset_schedule_path)
    MULTI_LORA_DATASET_SCHEDULE_PATH="$2"
    shift 2
    ;;
  --use_dummy_fixed_length_dataset)
    USE_DUMMY_FIXED_LENGTH_DATASET="$2"
    shift 2
    ;;
  --dummy_fixed_length_dataset_length)
    DUMMY_FIXED_LENGTH_DATASET_LENGTH="$2"
    shift 2
    ;;
  --log_folder_name)
    LOG_FOLDER_NAME="$2"
    shift 2
    ;;
  --profile)
    PROFILE="$2"
    shift 2
    ;;
  --benchmark_baseline_mlora_schedule)
    BENCHMARK_BASELINE_MLORA_SCHEDULE="$2"
    shift 2
    ;;
  --use_timers)
    USE_TIMERS="$2"
    shift 2
    ;;
  *)
    echo "Unknown parameter: $1"
    exit 1
    ;;
  esac
done

# Set up the log file
HOME_BENCHMARK="$(pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_HOME="${HOME_BENCHMARK}/logs/${LOG_FOLDER_NAME}"
mkdir -p "${LOG_HOME}"
LOG_DIR="${LOG_HOME}/${TIMESTAMP}"
LOG_FILE="${LOG_DIR}/run.log"
mkdir -p "${LOG_DIR}"

# Construct the arguments
model_arguments=(
  --model_name_or_path "$MODEL"
  --num_layers_for_debugging "$NUM_LAYERS_FOR_DEBUGGING"
  --disable_optimizer_for_debugging "$DISABLE_OPTIMIZER_FOR_DEBUGGING"
  --liger_kernel_level "$LIGER_KERNEL_LEVEL"
  --torch_compile_level "$TORCH_COMPILE_LEVEL"
  --lora_r "$LORA_R"
  --lora_alpha "$LORA_ALPHA"
  --lora_dropout "$LORA_DROPOUT"
  --lora_target_modules "all-linear"
  --use_flash_attn True
  --use_multi_lora "$USE_MULTI_LORA"
  --num_multi_loras "$NUM_MULTI_LORAS"
  --multi_lora_alpha "$MULTI_LORA_ALPHA"
  --multi_lora_r "$MULTI_LORA_R"
  --multi_lora_dropout "$MULTI_LORA_DROPOUT"
)
if [ "$APPLY_FUSED_LORA" = "True" ]; then
  model_arguments+=(
    --apply_fused_lora
    --merge_qkv_proj
  )
fi
training_arguments=(
  --global_batch_size "$GLOBAL_BATCH_SIZE"
  --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE"
  --use_fsdp "$USE_FSDP"
  --pipeline_parallel_size "$PIPELINE_PARALLEL_SIZE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --gradient_checkpointing "$GRADIENT_CHECKPOINTING"
  --profile "$PROFILE"
  --use_timers "$USE_TIMERS"
  --max_steps "$MAX_STEPS"
  --multi_lora_max_microbatch_tokens "$MULTI_LORA_MAX_MICROBATCH_TOKENS"
  --multi_lora_global_batch_sizes "$MULTI_LORA_GLOBAL_BATCH_SIZES"
)
if [ "$GRADIENT_CHECKPOINTING_LAYERS" != "None" ]; then
  training_arguments+=(
    --gradient_checkpointing_layers "$GRADIENT_CHECKPOINTING_LAYERS"
  )
fi
if [ "$BENCHMARK_BASELINE_MLORA_SCHEDULE" = "True" ]; then
  training_arguments+=(
    --benchmark_baseline_mlora_schedule
  )
fi
mock_data_arguments=(
  --dataset_path "$DATASET_PATH"
  --dataset_name "$DATASET_NAME"
  --num_samples "$NUM_SAMPLES"
  --seed_idx "$SEED_IDX"
  --permutation_idx "$PERMUTATION_IDX"
  --multi_lora_dataset_schedule_path "$MULTI_LORA_DATASET_SCHEDULE_PATH"
  --use_dummy_fixed_length_dataset "$USE_DUMMY_FIXED_LENGTH_DATASET"
  --dummy_fixed_length_dataset_length "$DUMMY_FIXED_LENGTH_DATASET_LENGTH"
)

# Set up torchrun arguments for distributed training
torchrun_args=(--nproc-per-node "$NPROC_PER_NODE")
if [ "$NNODES" -ne 1 ]; then
  # For multi-node setup, NODE_RANK, MASTER_ADDR, MASTER_PORT must be valid
  if [ -z "$NODE_RANK" ]; then
    echo "NODE_RANK is not set but required for multi-node setup"
    exit 1
  fi
  if [ -z "$MASTER_ADDR" ]; then
    echo "MASTER_ADDR is not set but required for multi-node setup"
    exit 1
  fi
  if [ -z "$MASTER_PORT" ]; then
    echo "MASTER_PORT is not set but required for multi-node setup"
    exit 1
  fi
  torchrun_args+=(
    --nnodes "$NNODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
  )
  echo "torchrun arguments: ${torchrun_args[*]}"
fi

torchrun "${torchrun_args[@]}" bench_multi_gpu.py \
  "${model_arguments[@]}" \
  "${training_arguments[@]}" \
  "${mock_data_arguments[@]}" \
  2>&1 | tee "$LOG_FILE"

# First check whether it is OOM or not
if grep -q "torch.OutOfMemoryError: CUDA out of memory." "${LOG_FILE}"; then
  echo "OOM detected. Skipping throughput calculation."
  THROUGHPUT_PER_GPU=OOM
  PEAK_MEMORY=OOM
else
  # [Rank 0] Global Benchmarking results: Total tokens: 188565, Total time: 20.19 s, Global Throughput (tokens/s): 9338.31, Per GPU Throughput (tokens/s): 9338.31
  THROUGHPUT_PER_GPU=$(grep "Global Benchmarking results:" "${LOG_FILE}" | sed -E 's/.*Per GPU Throughput \(tokens\/s\): ([0-9.]+).*/\1/' | head -n 1)
  PEAK_MEMORY=$(grep "\[After training\] Peak Memory:" "${LOG_FILE}" | sed -E 's/.*Peak Memory: ([0-9.]+).*/\1/' | head -n 1)
fi

# Set defaults for parameters that might not be properly captured
# TODO(zhanda): Resupport this for multi-gpu (FSDP and PP).
if [ -z "$TRAINABLE_PARAMS" ]; then
  TRAINABLE_PARAMS="N/A"
fi
if [ -z "$ALL_PARAMS" ]; then
  ALL_PARAMS="N/A"
fi
if [ -z "$TRAINABLE_PERCENT" ]; then
  TRAINABLE_PERCENT="N/A"
fi

# Print the results
{
  echo ""
  echo "===== Run Summary ====="
  echo "Timestamp: ${TIMESTAMP}"
  echo "Nnodes: ${NNODES}"
  echo "Nproc per node: ${NPROC_PER_NODE}"
  if [ "$NNODES" -ne 1 ]; then
    echo "Node rank: ${NODE_RANK}"
    echo "Master address: ${MASTER_ADDR}"
    echo "Master port: ${MASTER_PORT}"
  fi
  echo "Model: ${MODEL}"
  echo "LoRA dimensions: ${LORA_R}"
  echo "LoRA alpha: ${LORA_ALPHA}"
  echo "LoRA dropout: ${LORA_DROPOUT}"
  echo "Trainable params: ${TRAINABLE_PARAMS}"
  echo "All params: ${ALL_PARAMS}"
  echo "Trainable percent: ${TRAINABLE_PERCENT}"
  echo "Global batch size: ${GLOBAL_BATCH_SIZE}"
  echo "Pipeline parallel size: ${PIPELINE_PARALLEL_SIZE}"
  echo "Per device batch size: ${PER_DEVICE_BATCH_SIZE}"
  echo "Gradient accumulation steps: ${GRADIENT_ACCUMULATION_STEPS}"
  echo "Dataset path: ${DATASET_PATH}"
  echo "Dataset name: ${DATASET_NAME}"
  echo "Use fused lora: ${APPLY_FUSED_LORA}"
  echo "Use multi lora: ${USE_MULTI_LORA}"
  echo "Multi LoRA Dataset Schedule Path: ${MULTI_LORA_DATASET_SCHEDULE_PATH}"
  echo "Num samples: ${NUM_SAMPLES}"
  echo "Seed idx: ${SEED_IDX}"
  echo "Permutation idx: ${PERMUTATION_IDX}"
  echo "Context length: ${MAX_SEQ_LENGTH}"
  echo "Batch size: ${PER_DEVICE_BATCH_SIZE}"
  echo "Gradient accumulation steps: ${GRADIENT_ACCUMULATION_STEPS}"
  echo "Max train step: ${MAX_STEPS}"
  echo "Gradient checkpointing: ${GRADIENT_CHECKPOINTING}"
  echo "Gradient checkpointing layers: ${GRADIENT_CHECKPOINTING_LAYERS}"
  echo "Use liger kernel: ${LIGER_KERNEL_LEVEL}"
  echo "Torch compile level: ${TORCH_COMPILE_LEVEL}"
  echo "Use CUDA graph: ${USE_CUDA_GRAPH}"
  echo "Num layers for debugging: ${NUM_LAYERS_FOR_DEBUGGING}"
  echo "Disable optimizer for debugging: ${DISABLE_OPTIMIZER_FOR_DEBUGGING}"
  echo ""
  echo "===== Metrics ====="
  echo "Peak memory: ${PEAK_MEMORY} MB"
  echo "Throughput Per GPU: ${THROUGHPUT_PER_GPU} tokens/second per GPU"
} | tee -a "${LOG_FILE}"

# Save the results to a csv file
# Check the csv file exists. If it does, append the results to it.
# If it doesn't, create it and add the header.
echo ""
if [ -f "${LOG_HOME}/results.csv" ]; then
  echo "Appending results to ${LOG_HOME}/results.csv"
  echo "${TIMESTAMP},${MODEL},${USE_FSDP},${PIPELINE_PARALLEL_SIZE},${BENCHMARK_BASELINE_MLORA_SCHEDULE},${APPLY_FUSED_LORA},${USE_MULTI_LORA},${LORA_R},${LORA_ALPHA},${LORA_DROPOUT},${TRAINABLE_PARAMS},${ALL_PARAMS},${TRAINABLE_PERCENT},${GLOBAL_BATCH_SIZE},${PIPELINE_PARALLEL_SIZE},${PER_DEVICE_BATCH_SIZE},${GRADIENT_ACCUMULATION_STEPS},${DATASET_PATH},${DATASET_NAME},${NUM_SAMPLES},${SEED_IDX},${PERMUTATION_IDX},${MAX_SEQ_LENGTH},${PER_DEVICE_BATCH_SIZE},${GRADIENT_ACCUMULATION_STEPS},${MAX_STEPS},${GRADIENT_CHECKPOINTING},${LIGER_KERNEL_LEVEL},${TORCH_COMPILE_LEVEL},${USE_CUDA_GRAPH},${NUM_LAYERS_FOR_DEBUGGING},${DISABLE_OPTIMIZER_FOR_DEBUGGING},${NNODES},${NPROC_PER_NODE},${MASTER_ADDR},${MASTER_PORT},${PEAK_MEMORY},${THROUGHPUT_PER_GPU}" >>"${LOG_HOME}/results.csv"
else
  echo "Creating ${LOG_HOME}/results.csv"
  echo "timestamp,model,use_fsdp,pp_size,mlora,apply_fused_lora,use_multi_lora,lora_r,lora_alpha,lora_dropout,trainable_params,all_params,trainable_percent,global_batch_size,pipeline_parallel_size,per_device_batch_size,gradient_accum_steps,dataset_path,dataset_name,num_samples,seed_idx,permutation_idx,context_length,batch_size,gradient_accum_steps,max_train_step,gradient_checkpointing,liger_kernel_level,torch_compile_level,use_cuda_graph,num_layers_for_debugging,disable_optimizer_for_debugging,nnodes,nproc_per_node,master_addr,master_port,peak_memory,throughput_per_gpu" > "${LOG_HOME}/results.csv"
  echo "${TIMESTAMP},${MODEL},${USE_FSDP},${PIPELINE_PARALLEL_SIZE},${BENCHMARK_BASELINE_MLORA_SCHEDULE},${APPLY_FUSED_LORA},${USE_MULTI_LORA},${LORA_R},${LORA_ALPHA},${LORA_DROPOUT},${TRAINABLE_PARAMS},${ALL_PARAMS},${TRAINABLE_PERCENT},${GLOBAL_BATCH_SIZE},${PIPELINE_PARALLEL_SIZE},${PER_DEVICE_BATCH_SIZE},${GRADIENT_ACCUMULATION_STEPS},${DATASET_PATH},${DATASET_NAME},${NUM_SAMPLES},${SEED_IDX},${PERMUTATION_IDX},${MAX_SEQ_LENGTH},${PER_DEVICE_BATCH_SIZE},${GRADIENT_ACCUMULATION_STEPS},${MAX_STEPS},${GRADIENT_CHECKPOINTING},${LIGER_KERNEL_LEVEL},${TORCH_COMPILE_LEVEL},${USE_CUDA_GRAPH},${NUM_LAYERS_FOR_DEBUGGING},${DISABLE_OPTIMIZER_FOR_DEBUGGING},${NNODES},${NPROC_PER_NODE},${MASTER_ADDR},${MASTER_PORT},${PEAK_MEMORY},${THROUGHPUT_PER_GPU}" >>"${LOG_HOME}/results.csv"
fi

sleep 5
