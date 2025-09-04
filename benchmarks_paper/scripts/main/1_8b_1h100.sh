#!/bin/bash

# This takes 30~40 mins to run.
# Main script that runs all 8B model benchmarks across different datasets

bash scripts/1_1_8b_1h100_xsum.sh
bash scripts/1_2_8b_1h100_cnn_dailymail.sh
bash scripts/1_3_8b_1h100_wikisum.sh
bash scripts/1_4_8b_1h100_mix.sh
bash scripts/1_5_8b_1h100_het.sh