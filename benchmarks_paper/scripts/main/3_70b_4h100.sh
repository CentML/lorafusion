#!/bin/bash

# This takes ~75 mins to run.

bash scripts/3_1_70b_4h100_xsum.sh
bash scripts/3_2_70b_4h100_cnn_dailymail.sh
bash scripts/3_3_70b_4h100_wikisum.sh
bash scripts/3_4_70b_4h100_mix.sh
bash scripts/3_5_70b_4h100_het.sh