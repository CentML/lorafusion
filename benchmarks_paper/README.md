# LoRAFusion Artifact Evaluation

> Note: Here the figure numbers can be a bit confusing, we use the "0" to represent the data distribution plot, and "1" is the 
main end-to-end plot in the evaluation section. Figure "2" and "3" are the two subfigures for L40 GPU evaluation. Figure "4" 
is the scaling of the different methods. Figure "5" is the kernel performance forward/backward. Figure "6" is the layer 
performance normalized. Figure "7" is the kernel NCU profile. Figure "8" is the pipeline bubbles for the different 
configurations.

We provide the source code of LoRAFusion and scripts to reproduce the major experimental results from the paper.
This appendix shows how to generate the plots in Figure 0 (data distributions), Figure 1 (end-to-end results), Figure 5 (kernel performance), Figure 6 (layer-wise performance), and Figure 7 (memory traffic reduction).
We provide installation instructions and scripts to set up the environment.
To reproduce results, you need at least 192 GB RAM, 256 GB disk space, and 4 NVIDIA H100 GPUs.

## Description & Requirements

### Hardware dependencies
You need a Linux machine with at least 192 GB RAM, 256 GB free disk space, and 4 NVIDIA H100 GPUs with NVLinks.

We also provide instructions to set up the kernel and layer benchmarks for other hardware (1 GPU is enough). However, the results may not be exactly the same as those in the paper. Technically, our results will be better on the latest hardware with a larger TFLOPS/Memory Bandwidth ratio, so if you are using older hardware, the results may be slightly worse.

### Software dependencies
You need Conda to set up the environment. The environment includes CUDA 12.6, PyTorch v2.6.0, megatron-core v0.11.0, and Triton v3.2.0.

## Setup

1. **Clone the GitHub repository:**
   ```bash
   git clone https://github.com/CentML/lorafusion.git
   cd lorafusion
   git checkout eurosys-ae
   ```

2. **Install the requirements by running this command or following `../docs/installation.md`:**
   ```bash
   conda create -y -n lorafusion python=3.12
   conda activate lorafusion
   cd benchmarks_paper
   bash scripts/setup/setup_env.sh
   ```

3. **Download the Hugging Face models and datasets. Make sure you are logged in and have access to them:**
   ```bash
   # huggingface-cli login
   python prepare_models.py
   python gen_sample_distribution.py
   ```

4. **Check whether your hardware has been tuned for Triton**: Our kernels need to be tuned to choose the best tiling strategy for your hardware. Please check `lorafusion/ops/triton_ops/config.py` to see whether your hardware has been tuned. Here we have included the tuned configs for the following GPUs:
   - h100-80gb-hbm3 (recommended)
   - a100-sxm4-80gb
   - a100-80gb-pcie
   - geforce-rtx-3090

   If not, you need to tune it by running:
   ```bash
   cd /PATH/TO/lorafusion/
   python tools/tune_kernels.py
   ```

   And then update the `lorafusion/ops/triton_ops/config.py` file with the output tuned configurations.

## Evaluation Workflow

### Major Claims

- **(C1)**: LoRAFusion is up to 1.96× faster (average 1.47×) than Megatron-LM, and up to 1.46× faster (average 1.29×) than mLoRA. See Section 4.1 and Figure 1.

- **(C2)**: Our fused kernels are up to 1.39× faster (average 1.27×) and can replace existing LoRA kernels. See Section 4.2 and Figure 5, Figure 6, and Figure 7.

### Experiments (4 GPUs, each with 80GB memory)

1. **Make sure you are in the `benchmarks_paper` directory.**

2. **Run the experiments:**
   ```bash
   bash scripts/run_all.sh all
   ```
   
   a. This runs all the main experiments and kernel performance tests. It takes about 4 hours.
   
   b. Check `scripts/run_all.sh` for the exact commands and timing for each experiment.
   
   c. You can easily modify it to run only some experiments.

3. **Check the results in the `results` directory. The script automatically creates plots like those in Figure 0, Figure 1, Figure 5, Figure 6, and Figure 7.**

### Single GPU Kernel and Layer Benchmarks

If you only have 1 GPU, you can run the single GPU benchmark, and kernel and layer benchmarks by running:
```bash
bash scripts/run_all.sh all_single_gpu
```

Check the results in the `results` directory. The script automatically creates plots like those in sub figure 1 of Figure 1, Figure 5, Figure 6, and potentially Figure 7 (if you have NCU profiling enabled).

## Notes on Reusability

1. To customize experiments, edit `scripts/run_all.sh` and the related sub-scripts. We provide detailed scripts for each experiment and corresponding Python scripts to generate the plots.
2. For other old GPUs compared to H100, the results may be worse. In theory, if the ratio of TFLOPS over Memory Bandwidth is smaller, the results will be worse. Another possible reason is that we observe the power consumption issue during the kernel config tuning process, where the limit of the power consumption affects the correctness of the tuned configurations and the benchmarking results. In this case, we recommend you to set a GPU frequency lower than the default frequency for fair performance comparison.
   ```bash
   # Disable Automatic Frequency Scaling:
   sudo nvidia-smi -pm 1
   sudo nvidia-smi --auto-boost-default=0

   # Find supported frequencies by:
   nvidia-smi -q -d SUPPORTED_CLOCKS
   
   # Set the frequencies by, e.g.
   # sudo nvidia-smi -ac <memory clock,graphics clock>
   sudo nvidia-smi -ac 6251,1050
   ```
