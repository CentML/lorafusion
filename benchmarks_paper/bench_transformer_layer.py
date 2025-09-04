"""Benchmark a single transformer layer for comparing LoRA variants."""

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from transformers import AutoConfig, HfArgumentParser

from lorafusion.ops.multi_lora import (
    MULTI_LORA_BLOCK_SIZE,
    get_multi_lora_manager,
    prepare_multi_lora_batch_info,
)
from lorafusion.train.training_utils import (
    MockDataArguments,
    ModelArguments,
    TrainingArguments,
    create_model,
)
from lorafusion.utils.benchmark import (
    benchmark,
)
from lorafusion.utils.common import list_of_ints
from lorafusion.utils.hf import create_packed_dummy_inputs
from lorafusion.utils.module import (
    SUPPORTED_LAYER_TYPES,
    CompileModelLevel,
    LigerKernelLevel,
    compile_model,
    get_submodules_by_type,
)

# Default benchmark configuration
WARMUP = 50
NUMBER = 50
USE_CUDA_GRAPH = True
USE_CUDA_EVENT = True

# Default batch sizes and sequence lengths for comprehensive benchmarking
DEFAULT_BATCH_SIZES = [1, 2, 4, 8]
DEFAULT_SEQ_LENGTHS = [512, 1024, 2048, 4096]


@dataclass
class ExtraArgs:
    """Extra arguments for the benchmark."""

    seq_len: int = field(
        default=1024,
        metadata={"help": "Sequence length for single configuration benchmark"},
    )
    batch_sizes: str = field(
        default="1,2,4,8",
        metadata={"help": "Batch size for single configuration benchmark"},
    )
    folder_to_save: str = field(
        default="layer-results/",
        metadata={"help": "Folder to save the benchmark results"},
    )
    output_csv: str = field(
        default="layer_benchmark_results.csv",
        metadata={"help": "CSV file to save the benchmark results"},
    )


def run_single_lora_benchmarks(
    layer: torch.nn.Module,
    batch_sizes: list[int],
    seq_len: int,
    dtype: torch.dtype,
    rotary_emb_module: torch.nn.Module,
    hf_config: AutoConfig,
) -> dict[int, float]:
    """Run the single transformer layer benchmark.

    Args:
        layer: The transformer layer to benchmark
        batch_sizes: The batch sizes to benchmark
        seq_len: The sequence length to benchmark
        results_path: The path to save the results
        dtype: The data type to use
        rotary_emb_module: The rotary embedding module
        hf_config: The HuggingFace model configuration

    Returns:
        Dictionary mapping batch size to benchmark time
    """
    results = {}

    def prepare_func() -> dict[str, Any]:
        inputs_ = create_packed_dummy_inputs(
            hidden_size=hf_config.hidden_size,
            seq_len_list=[seq_len] * batch_size,
            dtype=dtype,
            device="cuda",
            return_input_ids=False,
            return_inputs_embeds=True,
            return_labels=False,
        )
        inputs_["hidden_states"] = inputs_.pop("inputs_embeds")
        inputs_["position_embeddings"] = rotary_emb_module(
            inputs_["hidden_states"], inputs_["position_ids"]
        )
        inputs_["grad_output"] = torch.ones_like(inputs_["hidden_states"])
        return inputs_

    def forward_backward_func(**inputs) -> None:
        """Forward and backward function for the layer."""
        grad_output = inputs.pop("grad_output")
        output = layer(**inputs)[0]
        output.backward(grad_output)

    for batch_size in batch_sizes:
        mean_time = benchmark(
            forward_backward_func,
            prepare_func=prepare_func,
            use_cuda_graph=USE_CUDA_GRAPH,
            use_cuda_event=USE_CUDA_EVENT,
            warmup=WARMUP,
            number=NUMBER,
            profile=training_args.profile,
            output_dir="./profiling-results/layer-forward-backward",
            worker_name=f"layer-single-lora-forward-backward-{time.strftime('%Y%m%d-%H%M%S')}",
            msg=f"Layer ({batch_size} x {seq_len}) forward+backward",
        )
        results[batch_size] = mean_time

    return results


def run_multi_lora_benchmarks(
    layer: torch.nn.Module,
    batch_sizes: list[int],
    seq_len: int,
    dtype: torch.dtype,
    rotary_emb_module: torch.nn.Module,
    hf_config: AutoConfig,
    num_adapters: int = 4,
) -> dict[int, float]:
    """Run the multi-LoRA transformer layer benchmark.

    Args:
        layer: The transformer layer to benchmark
        batch_sizes: The batch sizes to benchmark
        seq_len: The sequence length to benchmark
        dtype: The data type to use
        rotary_emb_module: The rotary embedding module
        hf_config: The HuggingFace model configuration
        num_adapters: Number of adapters to simulate in multi-LoRA

    Returns:
        Dictionary mapping batch size to benchmark time
    """
    results = {}

    # Get the multi-LoRA manager
    multi_lora_manager = get_multi_lora_manager()
    if multi_lora_manager is None:
        logger.error("Multi-LoRA manager is not initialized")
        return results

    lora_idx_list = list(range(num_adapters))

    for batch_size in batch_sizes:
        # For multi-LoRA, we evenly distribute the tokens among adapters
        total_tokens = batch_size * seq_len
        tokens_per_adapter = total_tokens // num_adapters
        remaining_tokens = total_tokens % num_adapters

        # Distribute tokens among adapters
        seq_len_list = [tokens_per_adapter] * num_adapters
        # Distribute any remaining tokens
        for i in range(remaining_tokens):
            seq_len_list[i] += 1

        def prepare_func() -> dict[str, Any]:
            # Initialize configurations per benchmark run to avoid in-place
            # modifications
            lora_rank_list = [
                lora_config.r
                for lora_config in multi_lora_manager.lora_configs[:num_adapters]
            ]
            dropout_p_list = [
                lora_config.lora_dropout
                for lora_config in multi_lora_manager.lora_configs[:num_adapters]
            ]
            alpha_list = [
                float(lora_config.lora_alpha)
                for lora_config in multi_lora_manager.lora_configs[:num_adapters]
            ]

            # Create batch info for this run
            multi_lora_batch_info = prepare_multi_lora_batch_info(
                seq_len_list=seq_len_list,  # noqa: B023
                lora_idx_list=lora_idx_list,
                lora_rank_list=lora_rank_list,
                dropout_p_list=dropout_p_list,
                alpha_list=alpha_list,
                block_size_m=MULTI_LORA_BLOCK_SIZE,
                output_dtype=dtype,
                allow_empty_micro_batch_info=True,
            )

            # Create packed inputs for the total batch
            inputs_ = create_packed_dummy_inputs(
                hidden_size=hf_config.hidden_size,
                seq_len_list=seq_len_list,  # noqa: B023
                dtype=dtype,
                device="cuda",
                return_input_ids=False,
                return_inputs_embeds=True,
                return_labels=False,
            )
            inputs_["hidden_states"] = inputs_.pop("inputs_embeds")
            inputs_["position_embeddings"] = rotary_emb_module(
                inputs_["hidden_states"], inputs_["position_ids"]
            )
            inputs_["grad_output"] = torch.ones_like(inputs_["hidden_states"])
            inputs_["multi_lora_batch_info"] = multi_lora_batch_info
            return inputs_

        def forward_backward_func(**inputs) -> None:
            """Forward and backward function for the layer with multi-LoRA."""
            grad_output = inputs.pop("grad_output")
            multi_lora_batch_info = inputs.pop("multi_lora_batch_info")

            # Add the batch info to the manager
            multi_lora_manager.add_batch_info(multi_lora_batch_info)

            # Forward pass
            multi_lora_manager.mark_forward_pass_started()
            output = layer(**inputs)[0]

            # Backward pass
            multi_lora_manager.mark_backward_pass_started()
            output.backward(grad_output)

            # Always clean up, even if an error occurs
            multi_lora_manager.pop_oldest_batch_info()

        # Run the benchmark
        mean_time = benchmark(
            forward_backward_func,
            prepare_func=prepare_func,
            use_cuda_graph=USE_CUDA_GRAPH,
            use_cuda_event=USE_CUDA_EVENT,
            warmup=WARMUP,
            number=NUMBER,
            profile=training_args.profile,
            output_dir="./profiling-results/layer-forward-backward",
            worker_name=f"layer-multi-lora-forward-backward-{time.strftime('%Y%m%d-%H%M%S')}",
            msg=(
                f"Multi-LoRA Layer ({num_adapters} adapters, {batch_size} x {seq_len}) "
                "forward+backward"
            ),
        )
        results[batch_size] = mean_time

    return results


def save_benchmark_results(
    results: dict[int, float],
    model_name: str,
    mode: str,
    seq_len: int,
    output_csv: str,
    folder_to_save: str,
    num_adapters: int | None = None,
) -> None:
    """Save benchmark results to a CSV file.

    Args:
        results: Dictionary mapping batch size to benchmark time
        model_name: Name of the model being benchmarked
        mode: Benchmark mode (e.g., "Vanilla", "FusedLoRA", "FusedMultiLoRA")
        seq_len: Sequence length used in the benchmark
        output_csv: Name of the CSV file to save results
        folder_to_save: Folder to save the CSV file
        num_adapters: Number of adapters (for multi-LoRA benchmarks)
    """
    # Create directory if it doesn't exist
    Path(folder_to_save).mkdir(parents=True, exist_ok=True)

    # Full path to the CSV file
    csv_path = Path(folder_to_save) / output_csv

    # Check if file exists to determine if we need to write headers
    file_exists = csv_path.exists()

    # Headers for the CSV file
    headers = [
        "Mode",
        "Model",
        "BatchSize",
        "SeqLength",
        "NumAdapters",
        "TimeMs",
        "ThroughputTokensPerSec",
    ]

    with csv_path.open("a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)

        # Write headers if file doesn't exist
        if not file_exists:
            writer.writeheader()

        # Write benchmark results
        for batch_size, mean_time in results.items():
            # Calculate throughput
            tokens_per_batch = batch_size * seq_len
            throughput = tokens_per_batch / mean_time if mean_time > 0 else 0

            writer.writerow(
                {
                    "Mode": mode,
                    "Model": model_name,
                    "BatchSize": batch_size,
                    "SeqLength": seq_len,
                    "NumAdapters": num_adapters if num_adapters is not None else "N/A",
                    "TimeMs": mean_time * 1000,  # Convert to milliseconds
                    "ThroughputTokensPerSec": throughput,
                }
            )

    logger.info(f"Benchmark results saved to {csv_path}")


def main(
    model_args: ModelArguments,
    training_args: TrainingArguments,
    mock_data_args: MockDataArguments,
    extra_args: ExtraArgs,
) -> None:
    """Run the single transformer layer benchmark.

    Args:
        model_args: Model arguments
        training_args: Training arguments
        mock_data_args: Mock data arguments
        extra_args: Extra arguments
    """
    # Apply Liger kernel if specified
    if model_args.liger_kernel_level != LigerKernelLevel.DISABLE:
        from lorafusion.utils.module import apply_liger_kernel

        apply_liger_kernel(model_args.liger_kernel_level)

    # Load the model
    model, _ = create_model(
        model_args=model_args,
        training_args=training_args,
        device_map="cpu",  # Load on CPU first to avoid OOM
        num_hidden_layers=1,
    )
    model = model.to("cuda")
    model = model.to(model_args.dtype)
    model_hf_config = model.config
    # Get the inner model (PeftModel -> LoRAModel -> LlamaForCausalLM)
    model = model.model.model

    # Extract the specified layer
    layer_dict = get_submodules_by_type(model, SUPPORTED_LAYER_TYPES)
    layer = next(iter(layer_dict.values()))
    layer = layer.to("cuda")

    # Apply torch.compile if specified
    if model_args.torch_compile_level != CompileModelLevel.DISABLE:
        logger.info(
            f"Compiling layer with torch.compile level {model_args.torch_compile_level}"
        )
        layer = compile_model(layer, model_args.torch_compile_level)

    # Print information about the layer
    logger.info(f"Layer to benchmark: {type(layer)}")

    # Run comprehensive benchmarks
    batch_sizes = list_of_ints(extra_args.batch_sizes)
    seq_len = extra_args.seq_len

    logger.info(f"Batch sizes: {batch_sizes}")
    logger.info(f"Sequence length: {seq_len}")

    # Determine model name from model_args
    model_name = model_args.model_name_or_path.split("/")[-1]

    if not model_args.use_multi_lora:
        # Run single LoRA benchmarks
        mode = "FusedLoRA" if model_args.apply_fused_lora else "Vanilla"
        results = run_single_lora_benchmarks(
            layer=layer,
            batch_sizes=batch_sizes,
            seq_len=seq_len,
            dtype=model_args.dtype,
            rotary_emb_module=model.rotary_emb,
            hf_config=model_hf_config,
        )
        # Save benchmark results
        save_benchmark_results(
            results=results,
            model_name=model_name,
            mode=mode,
            seq_len=seq_len,
            output_csv=extra_args.output_csv,
            folder_to_save=extra_args.folder_to_save,
        )
    else:
        # Run multi-LoRA benchmarks
        num_adapters = 4
        mode = "FusedMultiLoRA" if model_args.apply_fused_lora else "MultiLoRA"
        results = run_multi_lora_benchmarks(
            layer=layer,
            batch_sizes=batch_sizes,
            seq_len=seq_len,
            dtype=model_args.dtype,
            rotary_emb_module=model.rotary_emb,
            hf_config=model_hf_config,
            num_adapters=num_adapters,
        )
        # Save benchmark results
        save_benchmark_results(
            results=results,
            model_name=model_name,
            mode=mode,
            seq_len=seq_len,
            output_csv=extra_args.output_csv,
            folder_to_save=extra_args.folder_to_save,
            num_adapters=num_adapters,
        )


if __name__ == "__main__":
    # Parse arguments using the existing HfArgumentParser
    parser = HfArgumentParser(
        (ModelArguments, TrainingArguments, MockDataArguments, ExtraArgs),
        description="Benchmark a single transformer layer with LoRA variants.",
    )
    model_args, training_args, mock_data_args, extra_args = (
        parser.parse_args_into_dataclasses()
    )

    # Update the model args for benchmarking
    model_args.num_layers_for_debugging = 1

    logger.info(
        f"Model args: {model_args}, "
        f"Training args: {training_args}, "
        f"Mock data args: {mock_data_args}, "
        f"Extra args: {extra_args}"
    )

    main(model_args, training_args, mock_data_args, extra_args)
