#!/bin/bash

# This takes ~1 hour to run.
# Main script that runs all 32B model benchmarks across different datasets

bash scripts/2_1_32b_2h100_xsum.sh
bash scripts/2_2_32b_2h100_cnn_dailymail.sh
bash scripts/2_3_32b_2h100_wikisum.sh
bash scripts/2_4_32b_2h100_mix.sh
bash scripts/2_5_32b_2h100_het.sh