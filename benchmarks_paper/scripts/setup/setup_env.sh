#!/bin/bash

# Go to the project root directory
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
cd ${SCRIPT_DIR}/../../..

# Install PyTorch
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .

# Install CUDAToolkit
conda install nvidia/label/cuda-12.6.0::cuda-toolkit

# Install PuLP and CoincBC for the proposed solver
pip install pulp
conda install -y -c conda-forge coincbc