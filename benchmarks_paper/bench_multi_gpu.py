"""Entrypoint for demonstrating the pipeline parallel training."""

import sys
import time
from collections.abc import Callable
from functools import partial

import torch
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed import TorchFullyShardedDataParallel as TorchFSDP
from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.enums import ModelType
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.pipeline_parallel.schedules import (
    get_forward_backward_func as megatron_get_forward_backward_func,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_attr_wrapped_model
from peft.tuners.lora import Linear as LoRALinear
from transformers import HfArgumentParser

from lorafusion.ops.multi_lora import get_multi_lora_manager
from lorafusion.pipeline_parallel.baseline_mlora_schedule import (
    baseline_mlora_forward_backward_pipelining_without_interleaving,
)
from lorafusion.pipeline_parallel.multi_lora_schedule import (
    EMPTY_BATCH_MAGIC_TOKEN,
    multi_lora_get_forward_backward_func,
)
from lorafusion.pipeline_parallel.pipe_module import PipeModel
from lorafusion.train.training_utils import (
    DataProvider,
    MockDataArguments,
    MockDataset,
    MockMultiLoRADataProvider,
    MockMultiLoRADataset,
    ModelArguments,
    TrainingArguments,
    create_model,
    get_timers,
    init_timers,
    prepare_model_for_training,
    print_model_param_info,
)
from lorafusion.utils.benchmark import create_profiler_context
from lorafusion.utils.common import (
    empty_cuda_cache,
    log_memory_usage,
    logging_if,
    logging_with_rank,
    maybe_setup_distributed,
)
from lorafusion.utils.module import (
    SUPPORTED_LAYER_TYPES,
    CompileModelLevel,
    LigerKernelLevel,
    compile_model,
)

SEQ_LEN_LIST = [1024, 1024]


def initialize_distributed(tp_size: int, pp_size: int) -> None:
    """Initialize the distributed environment."""
    maybe_setup_distributed()
    # Currently only LoRA is supported (full model finetuning is not supported in this
    # script). The reason is that in finalize_model_grads, we need to allreduce the
    # gradients of the embeddings and that requires specific attributes in the model,
    # i.e. pre_process, etc. Now we disable embedding groups to avoid allreduce.
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        get_embedding_ranks=lambda *args, **kwargs: [0],
        get_position_embedding_ranks=lambda *args, **kwargs: [0],
    )


def construct_model_parallel_config(
    model: PipeModel, dtype: torch.dtype
) -> TransformerConfig:
    """Construct the model parallel config.

    See: https://github.com/volcengine/verl/blob/a35c044627941163066ad9ad131b7009b913e5a2/verl/models/mcore/config_converter.py
    """
    hf_config = model.hf_config
    overlap_p2p_comm = (
        parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None
        and parallel_state.get_virtual_pipeline_model_parallel_world_size() > 1
    )
    batch_p2p_comm = False
    return TransformerConfig(
        # Model architecture
        num_layers=hf_config.num_hidden_layers,
        hidden_size=hf_config.hidden_size,
        num_attention_heads=hf_config.num_attention_heads,
        num_query_groups=hf_config.num_key_value_heads,
        ffn_hidden_size=hf_config.intermediate_size,
        # Parallelization / Optimization
        params_dtype=dtype,
        pipeline_dtype=dtype,
        variable_seq_lengths=True,
        perform_initialization=False,
        tensor_model_parallel_size=parallel_state.get_tensor_model_parallel_world_size(),
        pipeline_model_parallel_size=parallel_state.get_pipeline_model_parallel_world_size(),
        # Communication
        overlap_p2p_comm=overlap_p2p_comm,
        batch_p2p_comm=batch_p2p_comm,
        moe_token_dispatcher_type="alltoall",  # noqa: S106
        # Timers
        timers=get_timers(),
    )


def decorate_model_for_megatron_core(model: PipeModel, dtype: torch.dtype) -> PipeModel:
    """Decorate the model for megatron core."""
    model.config = construct_model_parallel_config(model, dtype)
    model.model_type = ModelType.encoder_or_decoder
    model.decoder = model.model
    return model


def prepare_ddp_model(
    model: PipeModel,
    tf_config: TransformerConfig | None = None,
    *,
    use_fsdp: bool = False,
) -> DistributedDataParallel:
    """Prepare the model for DDP training."""
    TORCH_FSDP_with_modules = partial(  # noqa: N806
        TorchFSDP, sub_modules_to_wrap=list(SUPPORTED_LAYER_TYPES)
    )
    ddp_cls = TORCH_FSDP_with_modules if use_fsdp else DistributedDataParallel
    bucket_size = max(
        4e7,
        1e6 * parallel_state.get_data_parallel_world_size(with_context_parallel=True),
    )
    if tf_config is None:
        tf_config = model.config
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=False,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        align_param_gather=False,
        use_distributed_optimizer=False,
        check_for_nan_in_grad=False,
        bucket_size=bucket_size,
    )
    # Update no_sync_func for the ModelParallelConfig
    ddp_model = ddp_cls(
        config=tf_config,
        ddp_config=ddp_config,
        module=model,
        disable_bucketing=False,
    )
    tf_config.no_sync_func = ddp_model.no_sync
    tf_config.finalize_model_grads_func = finalize_model_grads
    return ddp_model


def prepare_multi_lora_ddp_model(
    model: PipeModel,
    *,
    use_fsdp: bool = False,
) -> tuple[DistributedDataParallel, list[DistributedDataParallel]]:
    """Prepare the model for multi-LoRA training."""
    multi_lora_manager = get_multi_lora_manager()
    fake_modules: list[torch.nn.Module] = [
        torch.nn.Module() for _ in range(len(multi_lora_manager.lora_configs))
    ]
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            for adapter_idx, adapter_name in enumerate(module.active_adapters):
                setattr(
                    fake_modules[adapter_idx],
                    f"{name}.{adapter_name}.lora_A",
                    module.lora_A[adapter_name],
                )
                setattr(
                    fake_modules[adapter_idx],
                    f"{name}.{adapter_name}.lora_B",
                    module.lora_B[adapter_name],
                )
    ddp_modules = [
        prepare_ddp_model(fake_module, tf_config=model.config, use_fsdp=False)
        for fake_module in fake_modules
    ]

    # Store the original parameters that require grad
    original_param_names_require_grad = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            original_param_names_require_grad.add(name)
            param.requires_grad_(requires_grad=False)
    # Create the ddp model for the entire model
    ddp_model = prepare_ddp_model(model, use_fsdp=use_fsdp)
    # Restore the original parameters that require grad
    for name, param in model.named_parameters():
        if name in original_param_names_require_grad:
            param.requires_grad_(requires_grad=True)

    return ddp_model, ddp_modules


def create_optimizer(
    model: PipeModel | DistributedDataParallel,
    model_args: ModelArguments,
    training_args: TrainingArguments,
    *,
    use_distributed_optimizer: bool = False,
    use_torch_optimizer: bool = True,
) -> torch.optim.Optimizer:
    """Create the optimizer for the pipeline parallel."""
    if model_args.use_torch_optimizer != use_torch_optimizer:
        msg = (
            f"model_args.use_torch_optimizer={model_args.use_torch_optimizer} "
            f"use_torch_optimizer={use_torch_optimizer} are different. "
            "We by default use torch optimizer to be compatible with FSDP2. "
            "If you want to use Megatron/Apex Optimizer, please set manually "
            "in both model_args and function arguments."
        )
        raise ValueError(msg)
    if use_torch_optimizer:
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=getattr(training_args, "learning_rate", 1e-3),
            weight_decay=getattr(training_args, "weight_decay", 0.0),
            fused=True,
        )
    config = OptimizerConfig(
        optimizer="adam",
        lr=getattr(training_args, "learning_rate", 1e-3),
        clip_grad=getattr(training_args, "max_grad_norm", 1.0),
        weight_decay=getattr(training_args, "weight_decay", 0.0),
        bf16=model_args.dtype == torch.bfloat16,
        use_distributed_optimizer=use_distributed_optimizer,
    )
    return get_megatron_optimizer(config, [model])


def setup_model_and_optimizer(
    model_args: ModelArguments,
    training_args: TrainingArguments,
) -> tuple[PipeModel | DistributedDataParallel, torch.optim.Optimizer]:
    """Model provider for the pipeline parallel."""
    dtype = model_args.dtype
    # Create and prepare the Huggingface model
    hf_model, _ = create_model(
        model_args, training_args=training_args, device_map="cpu"
    )
    hf_model = prepare_model_for_training(hf_model, model_args, training_args)
    # Create the pipeline model for this stage
    pipe_model = PipeModel.from_model(
        hf_model,
        hf_model.config,
        stage_idx=parallel_state.get_pipeline_model_parallel_rank(),
        num_stages=parallel_state.get_pipeline_model_parallel_world_size(),
    )
    # For FSDP2, we don't allocate GPU memory here.
    if not training_args.use_fsdp:
        pipe_model.to("cuda")
    # Compile the model if specified
    if model_args.torch_compile_level is not None:
        compile_model(pipe_model, model_args.torch_compile_level)
    decorate_model_for_megatron_core(pipe_model, dtype)

    if not model_args.use_multi_lora:
        # Create the DDP model
        ddp_model = prepare_ddp_model(pipe_model, use_fsdp=training_args.use_fsdp)

        # Create the optimizer
        optimizer = create_optimizer(ddp_model, model_args, training_args)

    else:
        ddp_model, ddp_modules = prepare_multi_lora_ddp_model(
            pipe_model, use_fsdp=training_args.use_fsdp
        )
        optimizers = [
            create_optimizer(ddp_module, model_args, training_args)
            for ddp_module in ddp_modules
        ]
        multi_lora_manager = get_multi_lora_manager()
        multi_lora_manager.register_ddp_modules_and_optimizers(
            ddp_model, ddp_modules, optimizers
        )
        optimizer = None

    print_model_param_info(ddp_model)

    return ddp_model, optimizer


def create_data_provider(
    mock_data_args: MockDataArguments,
    model_args: ModelArguments,
    *,
    hidden_size: int,
) -> DataProvider:
    """Create the data provider for the pipeline parallel."""
    dp_rank = parallel_state.get_data_parallel_rank()
    dp_size = parallel_state.get_data_parallel_world_size(with_context_parallel=True)

    if model_args.use_multi_lora:
        mock_dataset = MockMultiLoRADataset.from_dataset_args(mock_data_args)
        logging_with_rank(
            f"Creating MockMultiLoRADataProvider with "
            f"dp_rank={dp_rank}, dp_size={dp_size}."
        )
        data_provider = MockMultiLoRADataProvider(
            dataset=mock_dataset,
            batch_size=1,
            hidden_size=hidden_size,
            parallel_state=parallel_state,
            dp_rank=dp_rank,
            dp_size=dp_size,
            verbose=True,
        )
    else:
        if mock_data_args.use_dummy_fixed_length_dataset:
            mock_dataset = MockDataset.from_a_fixed_length(
                mock_data_args.dummy_fixed_length_dataset_length,
                mock_data_args.num_samples,
            )
            logging_with_rank(
                f"Creating Fixed Length DataProvider with "
                f"length={mock_data_args.dummy_fixed_length_dataset_length}, "
                f"num_samples={mock_data_args.num_samples}, "
                f"dp_rank={dp_rank}, dp_size={dp_size}, "
                f"batch_size={training_args.per_device_train_batch_size}"
            )
        else:
            mock_dataset = MockDataset.from_dataset_args(mock_data_args)
            logging_with_rank(
                f"Creating DataProvider with "
                f"dataset_name={mock_data_args.dataset_name}, "
                f"num_samples={mock_data_args.num_samples}, "
                f"dp_rank={dp_rank}, dp_size={dp_size}, "
                f"batch_size={training_args.per_device_train_batch_size}"
            )
        data_provider = DataProvider(
            mock_dataset,
            hidden_size=hidden_size,
            parallel_state=parallel_state,
            batch_size=training_args.per_device_train_batch_size,
            dp_rank=dp_rank,
            dp_size=dp_size,
            verbose=True,
        )
    return data_provider


def forward_step_func(
    data_provider: DataProvider,
    model: PipeModel,
) -> tuple[torch.Tensor, Callable]:
    """Forward step function for the pipeline parallel."""
    input_ids, shared_inputs, multi_lora_batch_info = data_provider.next_batch()

    # If the batch is empty, do a no-op pipeline step
    if input_ids is None and shared_inputs is None and multi_lora_batch_info is None:
        logging_with_rank(
            "Empty batch detected, returning a no-op pipeline step.", level="info"
        )
        minimal_output = torch.zeros(
            (1, 1, EMPTY_BATCH_MAGIC_TOKEN), dtype=torch.bfloat16, device="cuda"
        )

        # Return the minimal output and a minimal loss function
        def _minimal_loss_func(
            output_tensor: torch.Tensor,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            # Return a zero loss
            return output_tensor, {"loss": minimal_output}

        return minimal_output, _minimal_loss_func

    # Normal processing for non-empty batches
    if parallel_state.is_pipeline_first_stage():
        output = model(input_ids, **shared_inputs)
    else:
        output = model(**shared_inputs)

    # Return the output and a loss function
    def _loss_func(
        output_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Simple loss function that returns the output as the loss
        # In a real scenario, you would compute an actual loss here
        return output_tensor, {"loss": output_tensor}

    return output[0], _loss_func


def log_benchmarking_results(benchmarking_results: dict[str, list]) -> None:
    """Log local and global benchmarking results.

    Args:
        benchmarking_results: Dictionary containing lists of tokens and times
    """
    # Local benchmarking results
    local_tokens_sum = sum(benchmarking_results["tokens"])
    local_time_sum = sum(benchmarking_results["times"])
    logging_with_rank(
        f"Local Benchmarking results: "
        f"Total tokens: {local_tokens_sum}, "
        f"Total time: {local_time_sum:.2f} s, "
        f"Throughput (tokens/s): {local_tokens_sum / local_time_sum:.2f}",
        level="success",
    )

    # Global benchmarking results
    global_tokens = torch.tensor(
        benchmarking_results["tokens"], dtype=torch.int32, device="cuda"
    )
    global_time = torch.tensor(
        benchmarking_results["times"], dtype=torch.float32, device="cuda"
    )
    torch.distributed.all_reduce(
        global_tokens,
        group=parallel_state.get_data_parallel_group(),
        op=torch.distributed.ReduceOp.SUM,
    )
    torch.distributed.all_reduce(
        global_time,
        group=parallel_state.get_data_parallel_group(),
        op=torch.distributed.ReduceOp.MAX,
    )
    global_tokens_sum = global_tokens.sum().item()
    global_time_sum = global_time.sum().item()
    global_throughput = global_tokens_sum / global_time_sum
    logging_with_rank(
        f"Global Benchmarking results: "
        f"Total tokens: {global_tokens_sum}, "
        f"Total time: {global_time_sum:.2f} s, "
        f"Global Throughput (tokens/s): {global_throughput:.2f}, "
        f"Per GPU Throughput (tokens/s): "
        f"{global_throughput / torch.distributed.get_world_size():.2f}",
        level="success",
    )

    # Check the bubble time and output them
    if (
        benchmarking_results["forward_compute_times"]
        and benchmarking_results["backward_compute_times"]
    ):
        forward_compute_times = benchmarking_results["forward_compute_times"]
        backward_compute_times = benchmarking_results["backward_compute_times"]
        local_times = benchmarking_results["times"]
        if not (
            len(forward_compute_times)
            == len(backward_compute_times)
            == len(local_times)
        ):
            msg = (
                f"The length of forward_compute_times, backward_compute_times, "
                f"and local_times are not the same. "
                f"forward_compute_times: {len(forward_compute_times)}, "
                f"backward_compute_times: {len(backward_compute_times)}, "
                f"local_times: {len(local_times)}"
            )
            raise ValueError(msg)

        # Calculate the bubble time
        bubble_times = [
            local_times[i] - forward_compute_times[i] - backward_compute_times[i]
            for i in range(len(local_times))
        ]
        bubble_ratios = [
            bubble_times[i] / local_times[i] for i in range(len(local_times))
        ]
        local_average_bubble_ratio = sum(bubble_ratios) / len(bubble_ratios)
        global_average_bubble_ratio = torch.tensor(
            local_average_bubble_ratio, dtype=torch.float32, device="cuda"
        )
        torch.distributed.all_reduce(
            global_average_bubble_ratio,
            group=torch.distributed.group.WORLD,
            op=torch.distributed.ReduceOp.AVG,
        )
        global_average_bubble_ratio = global_average_bubble_ratio.item()
        logging_with_rank(
            f"Bubble ratios: {', '.join(f'{ratio:.2f}' for ratio in bubble_ratios)}, "
            f"Local average bubble ratio: {local_average_bubble_ratio * 100:.2f}%, "
            f"Global average bubble ratio: {global_average_bubble_ratio * 100:.2f}%",
            level="success",
        )

    # Log memory usage
    log_memory_usage("After training", print_fn=logging_with_rank)


def standard_benchmarking(  # noqa: C901, PLR0912
    forward_step_func: Callable,
    data_provider: DataProvider,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    forward_backward_func: Callable,
    *,
    max_steps: int,
    gradient_accumulation_steps: int,
    per_device_train_batch_size: int,
    profile: bool = False,
    profile_skip_first: int = 1,
    profile_wait: int = 0,
    profile_warmup: int = 10,
    profile_active: int = 2,
    profile_repeat: int = 1,
    benchmark_warmup_iters: int = 16,
    log_results: bool = True,
    **kwargs,
) -> None:
    """Standard benchmarking function.

    This is for all except multi-LoRA pipeline parallel.
    """
    profiler_context = create_profiler_context(
        profile=profile,
        skip_first=profile_skip_first,
        wait=profile_wait,
        warmup=profile_warmup,
        active=profile_active,
        repeat=profile_repeat,
        output_dir="./profiling-results",
        worker_name=f"pp-rank-{parallel_state.get_pipeline_model_parallel_rank()}",
    )

    timers = get_timers()
    benchmarking_results = {
        "tokens": [],
        "times": [],
        "forward_compute_times": [],
        "backward_compute_times": [],
    }

    with profiler_context as prof:
        for i in range(max_steps):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            start_tokens = data_provider.processed_tokens

            # For CUDA Graphs if needed
            torch.compiler.cudagraph_mark_step_begin()

            if hasattr(model, "zero_grad_buffer"):
                model.zero_grad_buffer()
            if optimizer is not None:
                optimizer.zero_grad()

            forward_output = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_provider,
                model=model,
                num_microbatches=gradient_accumulation_steps,
                micro_batch_size=per_device_train_batch_size,
                seq_length=sum(data_provider.peek_batch()),
                decoder_seq_length=sum(data_provider.peek_batch()),
                forward_only=False,
                **kwargs,
            )

            if optimizer is not None:
                optimizer.step()

            if parallel_state.is_pipeline_last_stage() and forward_output:
                # Extract loss value safely
                loss_value = 0.0
                if isinstance(forward_output[0], dict) and "loss" in forward_output[0]:
                    loss_value = forward_output[0]["loss"].item()
                elif isinstance(forward_output[0], torch.Tensor):
                    loss_value = forward_output[0].item()

                del loss_value  # Just for debugging if needed

            if prof is not None:
                prof.step()

            torch.cuda.synchronize()
            end_time = time.perf_counter()
            end_tokens = data_provider.processed_tokens
            if timers is not None:
                forward_compute_time = timers("forward-compute").elapsed(reset=True)
                backward_compute_time = timers("backward-compute").elapsed(reset=True)
            if i >= benchmark_warmup_iters:
                benchmarking_results["tokens"].append(end_tokens - start_tokens)
                benchmarking_results["times"].append(end_time - start_time)
                if timers is not None:
                    benchmarking_results["forward_compute_times"].append(
                        forward_compute_time
                    )
                    benchmarking_results["backward_compute_times"].append(
                        backward_compute_time
                    )
                    if log_results:
                        logging_with_rank(
                            f"Iteration {i} time: {end_time - start_time:.4f} s, "
                            f"forward_compute_time: {forward_compute_time:.4f} s, "
                            f"backward_compute_time: {backward_compute_time:.4f} s."
                        )
            if log_results:
                logging_with_rank(
                    f"Iteration {i} time: {end_time - start_time:.4f} s, "
                    f"tokens: {end_tokens - start_tokens}, "
                    f"throughput (tokens/s): "
                    f"{(end_tokens - start_tokens) / (end_time - start_time):.2f}"
                )

    empty_cuda_cache()
    if log_results and benchmarking_results["tokens"]:
        log_benchmarking_results(benchmarking_results)


def main(
    model_args: ModelArguments,
    training_args: TrainingArguments,
    mock_data_args: MockDataArguments,
) -> None:
    """Run the pipeline parallel demo."""
    # Initialize timers
    if training_args.use_timers:
        init_timers()

    # Initialize models for each pipeline stage
    initialize_distributed(1, training_args.pipeline_parallel_size)
    model, optimizer = setup_model_and_optimizer(model_args, training_args)

    # Check training arguments
    hf_config = get_attr_wrapped_model(model, "hf_config", allow_none=False)
    hidden_size = hf_config.hidden_size
    global_batch_size = training_args.global_batch_size
    per_device_train_batch_size = training_args.per_device_train_batch_size
    gradient_accumulation_steps = training_args.gradient_accumulation_steps
    dp_size = parallel_state.get_data_parallel_world_size(with_context_parallel=True)
    training_args.validate_global_batch_size(dp_size)
    logging_if(
        (
            "Successfully validated training arguments. "
            f"Global batch size: {global_batch_size}, "
            f"Per device train batch size: {per_device_train_batch_size}, "
            f"Gradient accumulation steps: {gradient_accumulation_steps}"
        ),
        condition=parallel_state.is_pipeline_first_stage(),
        level="success",
    )

    # Create the data provider
    data_provider = create_data_provider(
        mock_data_args, model_args, hidden_size=hidden_size
    )
    data_provider_steps = data_provider.num_global_batches(global_batch_size)
    # If we use multi-lora, we divide the data provider steps by the number of adapters
    if model_args.use_multi_lora:
        data_provider_steps = data_provider_steps // model_args.num_multi_loras
    max_steps = (
        min(training_args.max_steps, data_provider_steps)
        if training_args.max_steps is not None and training_args.max_steps > 0
        else data_provider_steps
    )

    # Before execution, empty the cache
    empty_cuda_cache()

    # Get the forward backward function
    if model_args.use_multi_lora:
        # Set gradient accumulation steps for benchmarking
        gradient_accumulation_steps = max_steps
        if training_args.profile:
            gradient_accumulation_steps = 8

        # Choose appropriate multi-lora schedule based on pipeline size
        forward_backward_func = multi_lora_get_forward_backward_func()

        standard_benchmarking(
            forward_step_func=forward_step_func,
            data_provider=data_provider,
            model=model,
            optimizer=None,
            forward_backward_func=forward_backward_func,
            # only call the schedule once because we set the gradient accumulation
            # steps to be the max steps
            max_steps=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_train_batch_size=1,
            profile=training_args.profile,
            profile_skip_first=0,
            profile_warmup=0,
            profile_active=1,
            profile_repeat=1,
            benchmark_warmup_iters=1,
            log_results=False,  # the results are logged in the schedule function
        )
    # For baseline mLoRA schedule
    elif training_args.benchmark_baseline_mlora_schedule:
        # Set gradient accumulation steps for benchmarking
        gradient_accumulation_steps = max_steps
        if training_args.profile:
            gradient_accumulation_steps = 8

        # Choose appropriate forward backward function
        forward_backward_func = (
            baseline_mlora_forward_backward_pipelining_without_interleaving
        )

        standard_benchmarking(
            forward_step_func=forward_step_func,
            data_provider=data_provider,
            model=model,
            optimizer=None,
            forward_backward_func=forward_backward_func,
            # only call the schedule once because we set the gradient accumulation
            # steps to be the max steps
            max_steps=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            profile=training_args.profile,
            profile_skip_first=0,
            profile_warmup=0,
            profile_active=1,
            profile_repeat=1,
            benchmark_warmup_iters=1,
            mlora_fake_optimizer=optimizer,
            log_results=False,  # the results are logged in the schedule function
            global_batch_sizes=training_args.multi_lora_global_batch_sizes,
        )
    else:
        # Check global batch size
        benchmark_warmup_iters = 4 if global_batch_size >= 16 else 16  # noqa: PLR2004
        forward_backward_func = megatron_get_forward_backward_func()
        standard_benchmarking(
            forward_step_func=forward_step_func,
            data_provider=data_provider,
            model=model,
            optimizer=optimizer,
            forward_backward_func=forward_backward_func,
            max_steps=max_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            profile=training_args.profile,
            benchmark_warmup_iters=benchmark_warmup_iters,
        )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        parser = HfArgumentParser(
            (ModelArguments, TrainingArguments, MockDataArguments),
            description="Train a model using PEFT LoRA.",
        )
        model_args, training_args, mock_data_args = parser.parse_args_into_dataclasses()
        logging_with_rank(
            f"Model args: {model_args}, "
            f"Training args: {training_args}, "
            f"Mock data args: {mock_data_args}",
            level="success",
        )
    else:
        debug_mapping = {
            "dp": {
                "global_batch_size": 4,
                "pipeline_parallel_size": 1,
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 1,
            },
            "pp-2": {
                "global_batch_size": 16,
                "pipeline_parallel_size": 2,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 4,
            },
            "pp-4": {
                "global_batch_size": 16,
                "pipeline_parallel_size": 4,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 8,
            },
        }
        key = "dp"
        global_batch_size = debug_mapping[key]["global_batch_size"]
        per_device_train_batch_size = debug_mapping[key]["per_device_train_batch_size"]
        gradient_accumulation_steps = debug_mapping[key]["gradient_accumulation_steps"]
        pipeline_parallel_size = debug_mapping[key]["pipeline_parallel_size"]
        model_args = ModelArguments(
            model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
            dtype="bfloat16",
            liger_kernel_level=LigerKernelLevel.ALL,
            torch_compile_level=CompileModelLevel.DISABLE,
        )
        training_args = TrainingArguments(
            global_batch_size=global_batch_size,
            per_device_train_batch_size=per_device_train_batch_size,
            pipeline_parallel_size=pipeline_parallel_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=False,
            profile=True,
        )
        mock_data_args = MockDataArguments(
            dataset_path="datasets/dataset_distributions.json",
            dataset_name="cnn_dailymail",
        )
    main(model_args, training_args, mock_data_args)
