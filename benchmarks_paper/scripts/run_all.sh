#!/bin/bash
# This script must be run from benchmarks_paper/ directory.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
cd ${SCRIPT_DIR}/..

# ------------------------------

BENCH_MODE="${1:-"all"}"
bench_main=False
bench_main_single_gpu=False
bench_kernel=False
bench_layer=False

if [ "$BENCH_MODE" != "all" ] && [ "$BENCH_MODE" != "main" ] && [ "$BENCH_MODE" != "kernel" ] && [ "$BENCH_MODE" != "layer" ] && [ "$BENCH_MODE" != "all_single_gpu" ]; then
    echo "Invalid benchmark mode: $BENCH_MODE"
    echo "Valid modes: all, main, kernel, layer, all_single_gpu"
    exit 1
fi

if [ "$BENCH_MODE" == "all" ]; then
    bench_main=True
    bench_kernel=True
    bench_layer=True
fi

if [ "$BENCH_MODE" == "main" ]; then
    bench_main=True
fi

if [ "$BENCH_MODE" == "kernel" ]; then
    bench_kernel=True
fi

if [ "$BENCH_MODE" == "layer" ]; then
    bench_layer=True
fi

if [ "$BENCH_MODE" == "all_single_gpu" ]; then
    bench_layer=True
    bench_kernel=True
    bench_main_single_gpu=True
fi

# ===============================
# You can also manually override the bench_modes here.
# e.g., bench_main=False
# ===============================

# Take ~10 mins to run.
if [ "$bench_layer" == True ]; then
    # Run the layer benchmarks
    bash scripts/layer_perf/run_layer.sh

    # Draw the figures for layer performance
    python plots/evaluation-fig-6-layer-perf.py
fi


# Take ~40 mins to run.
if [ "$bench_kernel" == True ]; then
    bash scripts/kernel_perf/1_bench_kernel.sh

    # Draw the figures for kernel performance
    python plots/evaluation-fig-5-kernel-perf.py

    # Run the NCU profiling for kernel
    bash scripts/kernel_perf/2_kernel_ncu_profile.sh

    # Draw the figures for kernel NCU profiling
    python plots/evaluation-fig-7-kernel-ncu-profile.py
    
fi

if [ "$bench_main" == True ]; then
    # This takes ~35 mins to run.
    bash scripts/main/1_1_8b_1h100_xsum.sh
    bash scripts/main/1_2_8b_1h100_cnn_dailymail.sh
    bash scripts/main/1_3_8b_1h100_wikisum.sh
    bash scripts/main/1_4_8b_1h100_mix.sh
    bash scripts/main/1_5_8b_1h100_het.sh

    # This takes ~1 hour to run.
    bash scripts/main/2_1_32b_2h100_xsum.sh
    bash scripts/main/2_2_32b_2h100_cnn_dailymail.sh
    bash scripts/main/2_3_32b_2h100_wikisum.sh
    bash scripts/main/2_4_32b_2h100_mix.sh
    bash scripts/main/2_5_32b_2h100_het.sh

    # This takes ~75 mins to run.
    bash scripts/main/3_1_70b_4h100_xsum.sh
    bash scripts/main/3_2_70b_4h100_cnn_dailymail.sh
    bash scripts/main/3_3_70b_4h100_wikisum.sh
    bash scripts/main/3_4_70b_4h100_mix.sh
    bash scripts/main/3_5_70b_4h100_het.sh

    # Parse the results
    # The results are saved to parsed_results.json
    python parse_main_results.py

    # Generate the plots
    # The plots are saved under results/
    python plots/evaluation-fig-1-end-to-end.py
fi

if [ "$bench_main_single_gpu" == True ]; then
    # This takes ~35 mins to run - only single GPU configurations
    bash scripts/main/1_1_8b_1h100_xsum.sh
    bash scripts/main/1_2_8b_1h100_cnn_dailymail.sh
    bash scripts/main/1_3_8b_1h100_wikisum.sh
    bash scripts/main/1_4_8b_1h100_mix.sh
    bash scripts/main/1_5_8b_1h100_het.sh

    # Parse the results
    # The results are saved to parsed_results.json
    python parse_main_results.py

    # Generate the plots
    # The plots are saved under results/
    python plots/evaluation-fig-1-end-to-end.py
fi

echo "The figures are saved under 'results/'."
