"""Benchmark different Batched LoRA models."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import torch
from loguru import logger
from peft import LoraConfig, PeftMixedModel
from torch import nn
from torch.cuda import nvtx

from lorafusion.ops.lora_v1 import fused_linear_lora as fused_linear_lora_v1
from lorafusion.ops.triton_ops.config import get_lora_kernel_config
from lorafusion.utils.benchmark import (
    benchmark,
    format_time,
    set_warmup_and_number,
    tabulate_2d_benchmark_results,
)
from lorafusion.utils.module import CompileModelLevel, compile_model

torch.backends.cudnn.benchmark = True

if TYPE_CHECKING:
    from peft import PeftConfig

# For benchmarking
WARMUP = 200
NUMBER = 200
USE_CUDA_GRAPH = True
USE_CUDA_EVENT = True
NCU_PROFILE = False


def set_ncu_profile() -> None:
    """Set the NCU profile.

    Args:
        ncu_profile: Whether to profile the model with NCU.
    """
    global WARMUP, NUMBER, USE_CUDA_GRAPH, USE_CUDA_EVENT, NCU_PROFILE
    WARMUP = 0
    NUMBER = 1
    USE_CUDA_GRAPH = False
    USE_CUDA_EVENT = True
    NCU_PROFILE = True


###############################################################################
#                           Base Models (No LoRA)                             #
###############################################################################
class LinearModel(nn.Module):
    """A simple linear model.."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Initialize the linear model.

        Args:
            in_features: The number of input features.
            out_features: The number of output features.
            bias: Whether to include a bias term.
            device: The device to run the model on.
            dtype: The dtype to run the model on.
        """
        super().__init__()
        self.linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.linear.weight.requires_grad = False
        if bias:
            self.linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the linear model.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.
        """
        return self.linear(x)


###############################################################################
#                     Benchmark Base and Standard Linear                      #
###############################################################################
@dataclass
class DataConfig:
    """The configuration for the data."""

    batch_size: int
    seq_len: int


@dataclass
class LoRAFamilyConfig:
    """The configuration for the LoRA family."""

    r: int
    lora_alpha: int
    lora_dropout: float
    lora_type: str
    dtype: torch.dtype

    def to_peft_config(self) -> PeftConfig:
        """Convert the LoRA family config to a PeftConfig."""
        return LoraConfig(
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            r=self.r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["linear"],
        )


@dataclass
class BenchmarkConfig:
    """The configuration for the benchmark."""

    in_features: int
    out_features: int
    adapter_config: LoRAFamilyConfig
    data_config: DataConfig
    compile_level: CompileModelLevel = CompileModelLevel.DISABLE


class BenchmarkBase:
    """The base class for all benchmarks."""

    def __init__(self, config: BenchmarkConfig) -> None:
        """Initialize the benchmark.

        Args:
            config: The configuration for the benchmark.
        """
        self.config = config
        self.model = LinearModel(
            in_features=config.in_features,
            out_features=config.out_features,
            bias=False,
            device="cuda",
            dtype=torch.bfloat16,
        )

    def prepare_inputs(self) -> dict[str, Any]:
        """Prepare the inputs for the benchmark.

        Returns:
            The inputs for the benchmark.
        """
        msg = "Subclasses must implement this method."
        raise NotImplementedError(msg)

    def prepare_inputs_and_grad_outputs(self) -> dict[str, Any]:
        """Prepare the inputs and grad outputs for the benchmark.

        Returns:
            The inputs and grad outputs for the benchmark.
        """
        msg = "Subclasses must implement this method."
        raise NotImplementedError(msg)

    def forward_func(self, inputs: list[torch.Tensor]) -> None:
        """Forward pass of the model.

        Args:
            inputs: The inputs for the forward pass.
        """
        msg = "Subclasses must implement this method."
        raise NotImplementedError(msg)

    def forward_backward_func(
        self, inputs: list[torch.Tensor], grad_outputs: list[torch.Tensor]
    ) -> None:
        """Forward and backward pass of the model.

        Args:
            inputs: The inputs for the forward pass.
            grad_outputs: The gradients for the backward pass.
        """
        msg = "Subclasses must implement this method."
        raise NotImplementedError(msg)

    @property
    def worker_name(self) -> str:
        """The name of the worker."""
        r = "0"
        if self.config.adapter_config is not None:
            r = self.config.adapter_config.r
        return (
            f"{self.__class__.__name__}-"
            f"bsz{self.config.data_config.batch_size}-"
            f"seq{self.config.data_config.seq_len}-"
            f"r{r}-"
            f"{time.strftime('%Y%m%d-%H%M%S')}"
        )

    def benchmark_forward(
        self,
        *,
        profile: bool = False,
        output_dir: str = "./profiling-results",
        worker_name: str | None = None,
    ) -> float:
        """Benchmark the forward pass of the model.

        Args:
            profile: Whether to profile the model.
            output_dir: The directory to save the profiling results.
            worker_name: The name of the worker.
        """
        return benchmark(
            self.forward_func,
            prepare_func=self.prepare_inputs,
            use_cuda_graph=USE_CUDA_GRAPH,
            use_cuda_event=USE_CUDA_EVENT,
            profile=profile,
            output_dir=Path(output_dir) / "forward",
            worker_name=self.worker_name if worker_name is None else worker_name,
        )

    def benchmark_forward_backward(
        self,
        *,
        profile: bool = False,
        output_dir: str = "./profiling-results",
        worker_name: str | None = None,
    ) -> float:
        """Benchmark the forward and backward pass of the model.

        Args:
            profile: Whether to profile the model.
            output_dir: The directory to save the profiling results.
            worker_name: The name of the worker.
        """
        return benchmark(
            self.forward_backward_func,
            prepare_func=self.prepare_inputs_and_grad_outputs,
            use_cuda_graph=USE_CUDA_GRAPH,
            use_cuda_event=USE_CUDA_EVENT,
            profile=profile,
            output_dir=Path(output_dir) / "forward_backward",
            worker_name=self.worker_name if worker_name is None else worker_name,
        )

    def __repr__(self) -> str:
        """Return the string representation of the benchmark."""
        return f"{self.__class__.__name__}(config={self.config})"


###############################################################################
#                              Base Linear Model                              #
###############################################################################


class RawLinearBenchmark(BenchmarkBase):
    """The benchmark for the raw linear model."""

    def prepare_inputs(self) -> dict[str, Any]:
        """Prepare the inputs for the benchmark.

        Returns:
            The inputs for the benchmark.
        """
        data_config = self.config.data_config
        return {
            "input_tensor": torch.randn(
                data_config.batch_size,
                data_config.seq_len,
                self.config.in_features,
                dtype=self.model.linear.weight.dtype,
                device=self.model.linear.weight.device,
                requires_grad=True,
            )
        }

    def prepare_inputs_and_grad_outputs(self) -> dict[str, Any]:
        """Prepare the inputs and grad outputs for the benchmark.

        Returns:
            The inputs and grad outputs for the benchmark.
        """
        fwd_inputs = self.prepare_inputs()
        fwd_inputs["grad_output"] = torch.randn_like(fwd_inputs["input_tensor"])
        return fwd_inputs

    def forward_func(self, input_tensor: torch.Tensor) -> None:
        """Forward pass of the model.

        Args:
            input_tensor: The input tensor for the forward pass.
        """
        self.model(input_tensor)

    def forward_backward_func(
        self, input_tensor: torch.Tensor, grad_output: torch.Tensor
    ) -> None:
        """Forward and backward pass of the model.

        Args:
            input_tensor: The inputs for the forward pass.
            grad_output: The gradients for the backward pass.
        """
        rid = nvtx.range_start("RawLinear")
        output = self.model(input_tensor)
        torch.autograd.backward(output, grad_output)
        nvtx.range_end(rid)


###############################################################################
#                              LoRA Models                                    #
###############################################################################


class LoRAFamilyBenchmark(BenchmarkBase):
    """The benchmark for the LoRA family."""

    def __init__(self, config: BenchmarkConfig) -> None:
        """Initialize the benchmark.

        Args:
            config: The configuration for the benchmark.
        """
        super().__init__(config)
        self.lora_model = PeftMixedModel(
            self.model,
            self.config.adapter_config.to_peft_config(),
            adapter_name="lora_0",
        )
        self.lora_model = compile_model(
            self.lora_model, level=self.config.compile_level
        )

    def prepare_inputs(self) -> dict[str, Any]:
        """Prepare the inputs for the benchmark.

        Returns:
            The inputs for the benchmark.
        """
        return {
            "input_tensor": torch.randn(
                self.config.data_config.batch_size,
                self.config.data_config.seq_len,
                self.config.in_features,
                dtype=self.model.linear.weight.dtype,
                device=self.model.linear.weight.device,
                requires_grad=True,
            )
        }

    def prepare_inputs_and_grad_outputs(self) -> dict[str, Any]:
        """Prepare the inputs and grad outputs for the benchmark.

        Returns:
            The inputs and grad outputs for the benchmark.
        """
        fwd_inputs = self.prepare_inputs()
        fwd_inputs["grad_output"] = torch.randn_like(fwd_inputs["input_tensor"])
        return fwd_inputs

    def forward_func(self, input_tensor: torch.Tensor) -> None:
        """Forward pass of the model.

        Args:
            input_tensor: The input tensor for the forward pass.
        """
        self.lora_model.set_adapter("lora_0")
        self.lora_model(input_tensor)

    def forward_backward_func(
        self, input_tensor: torch.Tensor, grad_output: torch.Tensor
    ) -> None:
        """Forward and backward pass of the model.

        Args:
            input_tensor: The inputs for the forward pass.
            grad_output: The gradients for the backward pass.
        """
        rid = nvtx.range_start("RawLoRA")
        self.lora_model.set_adapter("lora_0")
        # ================================================
        # I don't know why but it triggers CUBLAS error during ncu profiling
        # > output = self.lora_model(input_tensor)
        # ================================================
        x = input_tensor
        linear_w = self.lora_model.base_model.model.linear.base_layer.weight
        lora_a = self.lora_model.base_model.model.linear.lora_A["lora_0"].weight
        lora_b = self.lora_model.base_model.model.linear.lora_B["lora_0"].weight
        alpha = self.lora_model.base_model.model.linear.lora_alpha["lora_0"]
        dropout_p = self.lora_model.base_model.model.linear.lora_dropout["lora_0"].p
        x_dropout = torch.dropout(x, p=dropout_p, train=True)
        output = x @ linear_w.T + (x_dropout @ lora_a.T @ lora_b.T) * alpha
        torch.autograd.backward(output, grad_output)
        nvtx.range_end(rid)


###############################################################################
#                              FlashLoRA Models                               #
###############################################################################


def flash_lora_peft_mixed_model(
    model: PeftMixedModel, input_tensor: torch.Tensor
) -> torch.Tensor:
    """Flash LoRA PEFT Mixed Model.

    Args:
        model: The model to apply Flash LoRA to.
        input_tensor: The input tensor to apply Flash LoRA to.
    """
    if not isinstance(model, PeftMixedModel):
        msg = f"{model} must be a PeftMixedModel, got {type(model)}"
        raise TypeError(msg)
    fn = fused_linear_lora_v1
    return fn(
        x=input_tensor,
        linear_w=model.base_model.model.linear.base_layer.weight,
        lora_a=model.base_model.model.linear.lora_A["lora_0"].weight,
        lora_b=model.base_model.model.linear.lora_B["lora_0"].weight,
        alpha=model.base_model.model.linear.lora_alpha["lora_0"],
        dropout_p=model.base_model.model.linear.lora_dropout["lora_0"].p,
        seed=42,
    )


class FlashLoRAFamilyBenchmark(LoRAFamilyBenchmark):
    """The benchmark for the Flash LoRA family."""

    def forward_func(self, input_tensor: torch.Tensor) -> None:
        """Forward pass of the model.

        Args:
            input_tensor: The input tensor for the forward pass.
        """
        self.lora_model.set_adapter("lora_0")
        return flash_lora_peft_mixed_model(self.lora_model, input_tensor)

    def forward_backward_func(
        self, input_tensor: torch.Tensor, grad_output: torch.Tensor
    ) -> None:
        """Forward and backward pass of the model.

        Args:
            input_tensor: The input tensor for the forward pass.
            grad_output: The gradients for the backward pass.
        """
        rid = nvtx.range_start("FusedLoRA")
        output = self.forward_func(input_tensor)
        torch.autograd.backward(output, grad_output)
        nvtx.range_end(rid)


###############################################################################
#                              MultiLoRA Models                               #
###############################################################################


class MultiLoRAFamilyBenchmark(BenchmarkBase):
    """The benchmark for the Multi-LoRA family."""

    def __init__(self, config: BenchmarkConfig) -> None:
        """Initialize the benchmark.

        Args:
            config: The configuration for the benchmark.
        """
        super().__init__(config)
        # We'll create a standard model but use direct multi-LoRA functions
        # instead of going through PeftMixedModel
        self.num_adapters = 4

        # Prepare adapter configurations
        self.lora_rank_list = [config.adapter_config.r] * self.num_adapters
        self.dropout_p_list = [config.adapter_config.lora_dropout] * self.num_adapters
        self.alpha_list = [float(config.adapter_config.lora_alpha)] * self.num_adapters

        # Create LoRA A and B matrices for each adapter
        self.lora_a_list = [
            torch.randn(
                r,
                config.in_features,
                dtype=config.adapter_config.dtype,
                device="cuda",
                requires_grad=True,
            )
            for r in self.lora_rank_list
        ]

        self.lora_b_list = [
            torch.randn(
                config.out_features,
                r,
                dtype=config.adapter_config.dtype,
                device="cuda",
                requires_grad=True,
            )
            for r in self.lora_rank_list
        ]

        # Constants from multi_lora.py
        self.block_size_m = get_lora_kernel_config("fused_multi_lora_block_size_m")

    def prepare_inputs(self) -> dict[str, Any]:
        """Prepare the inputs for the benchmark.

        Returns:
            The inputs for the benchmark.
        """
        total_tokens = (
            self.config.data_config.batch_size * self.config.data_config.seq_len
        )
        # Evenly split tokens between adapters
        tokens_per_adapter = total_tokens // self.num_adapters
        remaining_tokens = total_tokens % self.num_adapters

        self.seq_len_list = [tokens_per_adapter] * self.num_adapters
        # Distribute any remaining tokens
        for i in range(remaining_tokens):
            self.seq_len_list[i] += 1

        self.lora_idx_list = list(range(self.num_adapters))

        # Prepare batch info for multi-LoRA
        from lorafusion.ops.multi_lora import prepare_multi_lora_batch_info

        self.multi_lora_batch_info = prepare_multi_lora_batch_info(
            seq_len_list=self.seq_len_list,
            lora_idx_list=self.lora_idx_list,
            lora_rank_list=self.lora_rank_list,
            dropout_p_list=self.dropout_p_list,
            alpha_list=self.alpha_list,
            block_size_m=self.block_size_m,
            output_dtype=self.config.adapter_config.dtype,
        )

        # Create total merged input tensor
        return {
            "input_tensor": torch.randn(
                1,  # batch dimension
                sum(self.seq_len_list),  # total sequence length
                self.config.in_features,
                dtype=self.config.adapter_config.dtype,
                device="cuda",
                requires_grad=True,
            ),
        }

    def prepare_inputs_and_grad_outputs(self) -> dict[str, Any]:
        """Prepare the inputs and grad outputs for the benchmark.

        Returns:
            The inputs and grad outputs for the benchmark.
        """
        fwd_inputs = self.prepare_inputs()
        fwd_inputs["grad_output"] = torch.randn_like(fwd_inputs["input_tensor"])
        return fwd_inputs

    def forward_func(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            input_tensor: The input tensor for the forward pass.
        """
        from lorafusion.ops.multi_lora import fused_linear_multi_lora

        return fused_linear_multi_lora(
            padded_x=input_tensor,
            linear_w=self.model.linear.weight,
            lora_a_list=self.lora_a_list,
            lora_b_list=self.lora_b_list,
            seq_len_list=self.seq_len_list,
            padded_seq_len_list=self.multi_lora_batch_info.padded_seq_len_list,
            block_to_lookup_table=self.multi_lora_batch_info.block_to_lookup_table,
            block_to_dropout_p=self.multi_lora_batch_info.block_to_dropout_p,
            block_to_alpha=self.multi_lora_batch_info.block_to_alpha,
            enable_dropout=self.multi_lora_batch_info.enable_dropout,
            same_dropout_p_value=self.multi_lora_batch_info.same_dropout_p_value,
            max_r=self.multi_lora_batch_info.max_r,
            linear_bias=None,
        )

    def forward_backward_func(
        self, input_tensor: torch.Tensor, grad_output: torch.Tensor
    ) -> None:
        """Forward and backward pass of the model.

        Args:
            input_tensor: The inputs for the forward pass.
            grad_output: The gradients for the backward pass.
        """
        rid = nvtx.range_start("FusedMultiLoRA")
        output = self.forward_func(input_tensor)
        torch.autograd.backward(output, grad_output)
        nvtx.range_end(rid)


###############################################################################
#                              Benchmark                                      #
###############################################################################
@click.command()
@click.option(
    "--in_features", type=int, default=4096, help="The number of input features."
)
@click.option(
    "--out_features", type=int, default=4096, help="The number of output features."
)
@click.option("--seq_len", type=int, default=2048, help="The sequence length.")
@click.option(
    "--batch_sizes",
    type=str,
    default="1,2,3,4,6",
    help="The batch sizes to benchmark.",
)
@click.option("--r", type=int, default=16, help="The rank of the LoRA matrices.")
@click.option(
    "--torch-compile", is_flag=True, default=False, help="Whether to use torch compile."
)
@click.option(
    "--path-to-save",
    type=str,
    default="./kernel-results/lora-kernels.csv",
    help="The path to save the results.",
)
@click.option(
    "--not-bench-flash-lora",
    is_flag=True,
    default=False,
    help="Whether to not benchmark Flash LoRA.",
)
@click.option(
    "--bench-multi-lora-only",
    is_flag=True,
    default=False,
    help="Whether to only benchmark Multi LoRA.",
)
@click.option(
    "--profile", is_flag=True, default=False, help="Whether to profile the model."
)
@click.option(
    "--ncu-profile",
    is_flag=True,
    default=False,
    help="Whether to profile the model with NCU.",
)
def main(  # noqa: C901, PLR0912, PLR0915
    *,
    in_features: int,
    out_features: int,
    seq_len: int,
    batch_sizes: str,
    r: int,
    torch_compile: bool,
    path_to_save: str | None,
    not_bench_flash_lora: bool,
    bench_multi_lora_only: bool,
    profile: bool,
    ncu_profile: bool,
) -> None:
    """Benchmark the LoRA family."""
    if ncu_profile:
        set_ncu_profile()

    lora_alpha = 16
    lora_dropout = 0.1
    lora_type = "LoRA"
    dtype = torch.bfloat16

    batch_size_choices = [int(s) for s in batch_sizes.split(",")]
    torch_compile_level = (
        CompileModelLevel.MODEL if torch_compile else CompileModelLevel.DISABLE
    )

    # Initialize standard benchmarks
    raw_linear_benchmarks = [
        RawLinearBenchmark(
            config=BenchmarkConfig(
                in_features=in_features,
                out_features=out_features,
                adapter_config=None,
                data_config=DataConfig(batch_size=batch_size, seq_len=seq_len),
            )
        )
        for batch_size in batch_size_choices
    ]

    single_lora_benchmarks = [
        LoRAFamilyBenchmark(
            config=BenchmarkConfig(
                in_features=in_features,
                out_features=out_features,
                adapter_config=LoRAFamilyConfig(
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    lora_type=lora_type,
                    dtype=dtype,
                ),
                data_config=DataConfig(batch_size=batch_size, seq_len=seq_len),
                compile_level=torch_compile_level,
            )
        )
        for batch_size in batch_size_choices
    ]

    flash_lora_benchmarks = [
        FlashLoRAFamilyBenchmark(
            config=BenchmarkConfig(
                in_features=in_features,
                out_features=out_features,
                adapter_config=LoRAFamilyConfig(
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    lora_type=lora_type,
                    dtype=dtype,
                ),
                data_config=DataConfig(batch_size=batch_size, seq_len=seq_len),
            )
        )
        for batch_size in batch_size_choices
    ]

    # Initialize multi-LoRA benchmarks
    multi_lora_benchmarks = [
        MultiLoRAFamilyBenchmark(
            config=BenchmarkConfig(
                in_features=in_features,
                out_features=out_features,
                adapter_config=LoRAFamilyConfig(
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    lora_type=lora_type,
                    dtype=dtype,
                ),
                data_config=DataConfig(batch_size=batch_size, seq_len=seq_len),
            )
        )
        for batch_size in batch_size_choices
    ]

    do_forward = not NCU_PROFILE
    do_forward_backward = True
    set_warmup_and_number(WARMUP, NUMBER)

    # Handle multi-LoRA benchmarks separately
    if bench_multi_lora_only:
        logger.info("Benchmarking Multi-LoRA kernel (4 adapters)")

        # Run benchmarks for multi-LoRA separately
        multi_lora_results = []
        multi_lora_first_col_choices = []

        # Define operations to benchmark
        benchmark_operations = {
            "FWD": (do_forward, "benchmark_forward"),
            "FWD+BWD": (do_forward_backward, "benchmark_forward_backward"),
        }

        # Run benchmarks for each operation
        for operation_name, (should_run, method_name) in benchmark_operations.items():
            if not should_run:
                continue

            for benchmark in multi_lora_benchmarks:
                batch_size = benchmark.config.data_config.batch_size
                m = batch_size * seq_len
                k = in_features
                n = out_features
                name = f"MultiLoRA-4x[{m}x{k}x{n}x{r}]"

                # Log benchmark configuration
                logger.info(
                    f"MultiLoRA Benchmark - Batch size: {batch_size}, "
                    f"Total tokens: {m}, K: {k}, N: {n}, R: {r}"
                )

                # Run benchmark
                time_result = getattr(benchmark, method_name)(profile=profile)
                multi_lora_results.append([time_result])
                multi_lora_first_col_choices.append(f"{name} {operation_name}")

                # Log result
                logger.info(
                    f" - [{operation_name}] MultiLoRA: {format_time(time_result)}"
                )

        # Generate summary table for multi-LoRA
        if multi_lora_results:
            multi_lora_table = tabulate_2d_benchmark_results(
                multi_lora_results,
                ["MultiLoRA"],
                multi_lora_first_col_choices,
                first_col_name="Shape",
                path_to_save=(
                    path_to_save.replace(".csv", "_multi_lora.csv")
                    if path_to_save
                    else None
                ),
            )
            logger.info(f"\nMulti-LoRA Results:\n{multi_lora_table}")

        return

    # Benchmark standard models
    if not not_bench_flash_lora:
        benchmark_groups = zip(
            single_lora_benchmarks,
            raw_linear_benchmarks,
            flash_lora_benchmarks,
            strict=False,
        )
        impl_names = [
            "Torch LoRA",
            "Torch Linear",
            "FlashLoRA",
        ]
    else:
        benchmark_groups = zip(
            single_lora_benchmarks,
            raw_linear_benchmarks,
            strict=False,
        )
        impl_names = [
            "Torch LoRA",
            "Torch Linear",
        ]

    for benchmark_group in benchmark_groups:
        batch_size = benchmark_group[0].config.data_config.batch_size
        m = batch_size * seq_len
        k = in_features
        n = out_features
        name = f"[{m}x{k}x{n}x{r}]"
        logger.info(
            f"Batch size: {batch_size}, M (num_tokens): {m}, K: {k}, N: {n}, R: {r}"
        )

        # Collect benchmark results
        results = []
        fwd_speedups = []
        fwd_bwd_speedups = []
        first_col_choices = []

        # Define operations to benchmark
        benchmark_operations = {
            "FWD": (do_forward, "benchmark_forward", fwd_speedups),
            "FWD+BWD": (
                do_forward_backward,
                "benchmark_forward_backward",
                fwd_bwd_speedups,
            ),
        }

        # Run benchmarks for each operation
        for operation_name, (
            should_run,
            method_name,
            speedup_tracker,
        ) in benchmark_operations.items():
            if not should_run:
                continue

            # Run benchmarks for all implementations
            times = [
                getattr(bench, method_name)(profile=profile)
                for bench in benchmark_group
            ]
            # Unpack times based on available implementations
            expected_len = 2
            if len(times) == expected_len:  # Only Torch LoRA and Torch Linear
                lora_time, linear_time = times
                has_flash_lora = False
            else:  # All three implementations
                lora_time, linear_time, flash_time = times
                has_flash_lora = True

            # Record results
            results.append(times)
            first_col_choices.append(f"{name} Compile={torch_compile} {operation_name}")

            # Calculate speedups only if FlashLoRA is available
            if has_flash_lora:
                speedup_tracker.append(
                    [[linear_time / flash_time, lora_time / flash_time]]
                )

                # Log performance comparison with FlashLoRA
                logger.info(
                    f" - [{operation_name}] "
                    f"Torch LoRA: {format_time(lora_time)}, "
                    f"Torch Linear: {format_time(linear_time)}, "
                    f"FlashLoRA: {format_time(flash_time)} "
                    f"Slowdown: {flash_time / linear_time * 100:.2f}% "
                    f"-> {lora_time / linear_time * 100:.2f}% "
                    f"Speedup: {linear_time / flash_time:.2f}x vs linear, "
                    f"{lora_time / flash_time:.2f}x vs LoRA"
                )
            else:
                # Log performance comparison without FlashLoRA
                logger.info(
                    f" - [{operation_name}] "
                    f"Torch LoRA: {format_time(lora_time)}, "
                    f"Torch Linear: {format_time(linear_time)} "
                    f"Slowdown: {lora_time / linear_time * 100:.2f}%"
                )

        # Generate summary table
        if results:
            # Generate and log the table
            table = tabulate_2d_benchmark_results(
                results,
                impl_names,
                first_col_choices,
                first_col_name="Shape",
                path_to_save=path_to_save,
            )
            logger.info(f"\n{table}")


if __name__ == "__main__":
    main()
