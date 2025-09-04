# Installation

## Prerequisites

### Step 1 Conda Environment

Create a new conda environment and install the prerequisites.

```bash
conda create -y -n lorafusion python=3.12
conda activate lorafusion
```

### [Optional] Step 2 Install Prerequisites

Follow the instructions in the [Development](./development.md) guide to install the prerequisites.

### Step 3 Install Torch, LoRAFusion and Dependencies

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
```

If you want to use NCU, perhaps you need to install the CUDAToolkit:
```bash
conda install -y nvidia/label/cuda-12.6.0::cuda-toolkit
```

Additionally, if you want to use the proposed solver, you need to install the following dependencies:

```bash
pip install pulp
# Install through conda-forge
conda install -y -c conda-forge coincbc
```
