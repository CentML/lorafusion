"""1F1B schedule for multi-lora model."""

# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import contextlib
from collections.abc import Callable, Iterator

import torch
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import (
    backward_step,
    check_first_val_step,
    deallocate_output_tensor,
    forward_step,
    get_tensor_shapes,
    recv_backward,
    recv_forward,
    send_backward,
    send_backward_recv_forward,
    send_forward,
    send_forward_recv_backward,
)
from megatron.core.timers import Timers
from megatron.core.utils import (
    get_model_config,
    get_model_type,
    get_model_xattn,
)

from lorafusion.ops.multi_lora import get_multi_lora_manager
from lorafusion.utils.common import log_memory_usage, logging_if, logging_with_rank

# Types
Shape = list[int] | torch.Size

EMPTY_BATCH_MAGIC_TOKEN = 7


def logging_results(
    cuda_events_end: list[torch.cuda.Event],
    tokens_end: list[int],
    timers: Timers | None = None,
    warmup_microbatches: int = 24,
    num_pipeline_stages: int = 4,
) -> None:
    """Log local and global benchmarking results."""
    # Local benchmarking results
    torch.cuda.synchronize()
    local_tokens_sum = (
        tokens_end[-num_pipeline_stages] - tokens_end[warmup_microbatches]
    )
    local_time_sum = (
        cuda_events_end[warmup_microbatches].elapsed_time(
            cuda_events_end[-num_pipeline_stages]
        )
        / 1000
    )
    logging_with_rank(
        f"Local Benchmarking results: "
        f"Total tokens: {local_tokens_sum}, "
        f"Total time: {local_time_sum:.2f} s, "
        f"Throughput (tokens/s): {local_tokens_sum / local_time_sum:.2f}",
        level="success",
        depth=2,
    )

    # Global benchmarking results
    global_tokens = torch.tensor(local_tokens_sum, dtype=torch.int32, device="cuda")
    global_time = torch.tensor(local_time_sum, dtype=torch.float32, device="cuda")
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
        depth=2,
    )

    if timers is not None:
        forward_compute_time = timers("forward-compute").elapsed(reset=True)
        backward_compute_time = timers("backward-compute").elapsed(reset=True)
        forward_backward_time = timers("forward-backward").elapsed(reset=True)
        if "optimizer-step" in timers._timers:  # noqa: SLF001
            optimizer_step_time = timers("optimizer-step").elapsed(reset=True)
        else:
            optimizer_step_time = 0.0

        bubble_time = (
            forward_backward_time
            - forward_compute_time
            - backward_compute_time
            - optimizer_step_time
        )
        bubble_ratio = bubble_time / forward_backward_time
        local_average_bubble_ratio = bubble_ratio
        global_average_bubble_ratio = torch.tensor(
            local_average_bubble_ratio, dtype=torch.float32, device="cuda"
        )
        torch.distributed.all_reduce(
            global_average_bubble_ratio,
            group=torch.distributed.group.WORLD,
            op=torch.distributed.ReduceOp.AVG,
        )
        logging_with_rank(
            f"Local average bubble ratio: {local_average_bubble_ratio * 100:.2f}%, "
            f"Global average bubble ratio: {global_average_bubble_ratio * 100:.2f}%",
            level="success",
            depth=2,
        )

    # Log memory usage
    log_memory_usage("After training", print_fn=logging_with_rank)


def multi_lora_get_forward_backward_func() -> Callable:
    """Get multi-lora forward-backward function."""
    pipeline_model_parallel_size = (
        parallel_state.get_pipeline_model_parallel_world_size()
    )
    if pipeline_model_parallel_size > 1:
        forward_backward_func = (
            multi_lora_forward_backward_pipelining_without_interleaving
        )
    else:
        forward_backward_func = multi_lora_forward_backward_no_pipelining
    return forward_backward_func


def multi_lora_forward_backward_no_pipelining(  # noqa: C901, PLR0912, PLR0915
    *,
    forward_step_func: Callable,
    data_iterator: Iterator | list[Iterator],
    model: torch.nn.Module | list[torch.nn.Module],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int = 1,
    decoder_seq_length: int | None = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool | None = None,
    profile: bool = False,
) -> list[torch.Tensor]:
    """No pipeline forward-backward schedule."""
    if isinstance(model, list) and len(model) > 1:
        msg = "non-pipeline-parallel schedule does not support model chunking"
        raise ValueError(msg)
    if isinstance(data_iterator, list) and len(data_iterator) > 1:
        msg = "non-pipeline-parallel schedule does not support data chunking"
        raise ValueError(msg)

    config = get_model_config(model)
    if config.timers is not None:
        config.timers("forward-backward", log_level=1).start(
            barrier=config.barrier_with_L1_time
        )

    # ===============================================================
    # Multi-LoRA manager
    multi_lora_manager = get_multi_lora_manager()
    logging_interval = 8
    cuda_events_start = [
        torch.cuda.Event(enable_timing=True) for _ in range(num_microbatches)
    ]
    cuda_events_end = [
        torch.cuda.Event(enable_timing=True) for _ in range(num_microbatches)
    ]
    tokens_start = [None for _ in range(num_microbatches)]
    tokens_end = [None for _ in range(num_microbatches)]
    # ===============================================================

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync() -> None:
        """Disable asynchronous grad reductions."""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync() -> None:
        """Enable asynchronous grad reductions."""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    model_type = get_model_type(model)

    forward_data_store = []
    input_tensor, output_tensor_grad = None, None
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")

    for i in range(num_microbatches):
        # ===============================================================
        cuda_events_start[i].record()
        tokens_start[i] = data_iterator.processed_tokens
        multi_lora_manager.mark_forward_pass_started()
        # ===============================================================

        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            is_first_microbatch=check_first_val_step(
                first_val_step, forward_only, i == 0
            ),
            current_microbatch=i,
        )
        total_num_tokens += num_tokens.item()

        if not forward_only:
            multi_lora_manager.mark_backward_pass_started()
            backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )
            multi_lora_manager.pop_oldest_batch_info()
            if config.timers is not None:
                config.timers("optimizer-step").start()
            enable_grad_sync()
            multi_lora_manager.maybe_reduce_grad_and_optimizer_step()
            disable_grad_sync()
            if config.timers is not None:
                config.timers("optimizer-step").stop()

            # ===============================================================
            cuda_events_end[i].record()
            tokens_end[i] = data_iterator.processed_tokens

            if (i + 1) % logging_interval == 0:
                logging_start = i - logging_interval * 2
                logging_end = i - logging_interval
                if logging_start > 0:
                    curr_time = (
                        cuda_events_end[logging_start].elapsed_time(
                            cuda_events_end[logging_end]
                        )
                        / 1000
                    )
                    curr_tokens = tokens_end[logging_end] - tokens_end[logging_start]
                    logging_if(
                        f"Finished {i + 1} microbatches, "
                        f"for the last {logging_interval} microbatches: "
                        f"time: {curr_time}, "
                        f"tokens: {curr_tokens}, "
                        f"throughput (tokens/s): "
                        f"{curr_tokens / curr_time:.2f}",
                        condition=True,
                    )
            # ===============================================================

    # Launch any remaining grad reductions
    enable_grad_sync()
    if config.grad_sync_func is not None:
        config.grad_sync_func(model.parameters())

    if config.finalize_model_grads_func is not None and not forward_only:
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism and layernorm all-reduce for sequence parallelism).
        config.finalize_model_grads_func(
            [model], total_num_tokens if config.calculate_per_token_loss else None
        )
        # Also finalize grads for the multi-lora modules
        config.finalize_model_grads_func(
            multi_lora_manager.ddp_modules,
            total_num_tokens if config.calculate_per_token_loss else None,
        )

    if config.timers is not None:
        config.timers("forward-backward").stop()

    logging_results(cuda_events_end, tokens_end)

    return forward_data_store


def multi_lora_forward_backward_pipelining_without_interleaving(  # noqa: C901, PLR0912, PLR0915
    *,
    forward_step_func: Callable,
    data_iterator: Iterator | list[Iterator],
    model: torch.nn.Module | list[torch.nn.Module],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int = 1,
    decoder_seq_length: int | None = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool | None = None,
    profile: bool = False,
) -> list[torch.Tensor]:
    """1F1B schedule for multi-lora model."""
    if isinstance(model, list) and len(model) > 1:
        msg = (
            "non-interleaved pipeline-parallel schedule does not support model chunking"
        )
        raise ValueError(msg)
    if isinstance(data_iterator, list) and len(data_iterator) > 1:
        msg = (
            "non-interleaved pipeline-parallel schedule does not support data chunking"
        )
        raise ValueError(msg)

    config = get_model_config(model)
    if config.overlap_p2p_comm:
        msg = (
            "Non-interleaved pipeline parallelism does not support "
            "overlapping p2p communication"
        )
        raise ValueError(msg)

    if config.timers is not None:
        config.timers("forward-backward", log_level=1).start(
            barrier=config.barrier_with_L1_time
        )

    # ===============================================================
    # Multi-LoRA manager
    multi_lora_manager = get_multi_lora_manager()
    warmup_microbatches = 32
    logging_interval = 8
    cuda_events_start = [
        torch.cuda.Event(enable_timing=True) for _ in range(num_microbatches)
    ]
    cuda_events_end = [
        torch.cuda.Event(enable_timing=True) for _ in range(num_microbatches)
    ]
    tokens_start = [None for _ in range(num_microbatches)]
    tokens_end = [None for _ in range(num_microbatches)]
    start_idx = 0
    end_idx = 0
    # ===============================================================

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync() -> None:
        """Disable asynchronous grad reductions."""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync() -> None:
        """Enable asynchronous grad reductions."""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Compute number of warmup microbatches.
    num_warmup_microbatches = (
        parallel_state.get_pipeline_model_parallel_world_size()
        - parallel_state.get_pipeline_model_parallel_rank()
        - 1
    )
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    model_type = get_model_type(model)
    encoder_decoder_xattn = get_model_xattn(model)

    rank = parallel_state.get_pipeline_model_parallel_rank()
    recv_tensor_shapes = get_tensor_shapes(
        rank=rank - 1,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        encoder_decoder_xattn=encoder_decoder_xattn,
    )
    send_tensor_shapes = get_tensor_shapes(
        rank=rank,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        encoder_decoder_xattn=encoder_decoder_xattn,
    )

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    empty_batch_markers = None
    total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()

    if not forward_only:
        input_tensors = []
        output_tensors = []
        empty_batch_markers = []
    forward_data_store = []

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        # Decide to checkpoint all layers' activations of the current micro-batch
        checkpoint_activations_microbatch = None

        input_tensor = recv_forward(recv_tensor_shapes, config)

        # ===============================================================
        cuda_events_start[start_idx].record()
        tokens_start[start_idx] = data_iterator.processed_tokens
        start_idx += 1
        multi_lora_manager.mark_forward_pass_started()
        # ===============================================================

        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(first_val_step, forward_only, i == 0),
            current_microbatch=i,
            encoder_decoder_xattn=encoder_decoder_xattn,
        )

        # Check if this is an empty batch (marked by having exactly 1 token)
        is_empty_batch = output_tensor[0].numel() == EMPTY_BATCH_MAGIC_TOKEN

        # Always send forward to maintain pipeline synchronization
        send_forward(output_tensor, send_tensor_shapes, config)

        # Only count tokens for real batches
        if not is_empty_batch:
            total_num_tokens += num_tokens.item()

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            empty_batch_markers.append(is_empty_batch)
            # Safe deallocation with NULL checks
            if (
                config.deallocate_pipeline_outputs
                and output_tensor is not None
                and isinstance(output_tensor, list | tuple)
                and len(output_tensor) > 0
                and output_tensor[0] is not None
            ):
                deallocate_output_tensor(
                    output_tensor[0], deallocate_pipeline_outputs=True
                )

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        input_tensor = recv_forward(recv_tensor_shapes, config)

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        last_iteration = i == (num_microbatches_remaining - 1)

        # Decide to checkpoint all layers' activations of the current micro-batch
        checkpoint_activations_microbatch = None

        # ===============================================================
        cuda_events_start[start_idx].record()
        tokens_start[start_idx] = data_iterator.processed_tokens
        start_idx += 1
        multi_lora_manager.mark_forward_pass_started()
        # ===============================================================

        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(
                first_val_step,
                forward_only,
                (i == 0) and (num_warmup_microbatches == 0),
            ),
            current_microbatch=i + num_warmup_microbatches,
            encoder_decoder_xattn=encoder_decoder_xattn,
        )

        # Check if this is an empty batch (marked by having exactly 1 token)
        is_empty_batch = output_tensor[0].numel() == EMPTY_BATCH_MAGIC_TOKEN

        # Only count tokens for real batches
        if not is_empty_batch:
            total_num_tokens += num_tokens.item()

        if forward_only:
            send_forward(output_tensor, send_tensor_shapes, config)

            if not last_iteration:
                input_tensor = recv_forward(recv_tensor_shapes, config)

        else:
            output_tensor_grad = send_forward_recv_backward(
                output_tensor, send_tensor_shapes, config
            )

            # Add input_tensor and output_tensor to end of list.
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            empty_batch_markers.append(is_empty_batch)
            # Safe deallocation with NULL checks
            if (
                config.deallocate_pipeline_outputs
                and output_tensor is not None
                and isinstance(output_tensor, list | tuple)
                and len(output_tensor) > 0
                and output_tensor[0] is not None
            ):
                deallocate_output_tensor(
                    output_tensor[0], deallocate_pipeline_outputs=True
                )

            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)
            is_current_batch_empty = empty_batch_markers.pop(0)

            # Because right now we don't need to store the forward data
            if parallel_state.is_pipeline_last_stage():
                forward_data_store.pop(0)

            # Mark backward pass started for all batches for bookkeeping
            multi_lora_manager.mark_backward_pass_started()

            # For empty batches, we skip the actual backward computation but still
            # maintain pipeline synchronization
            if not is_current_batch_empty:
                # Normal backward pass for non-empty batches
                input_tensor_grad = backward_step(
                    input_tensor, output_tensor, output_tensor_grad, model_type, config
                )
            else:
                # For empty batches, we just pass through the gradient without
                # computation
                input_tensor_grad = torch.zeros(
                    (1, 1, EMPTY_BATCH_MAGIC_TOKEN),
                    dtype=config.pipeline_dtype,
                    device="cuda",
                )

            # Only perform gradient sync and optimizer step for non-empty batches
            if not is_current_batch_empty:
                multi_lora_manager.pop_oldest_batch_info()
                if config.timers is not None:
                    config.timers("optimizer-step").start()
                enable_grad_sync()
                multi_lora_manager.maybe_reduce_grad_and_optimizer_step()
                disable_grad_sync()
                if config.timers is not None:
                    config.timers("optimizer-step").stop()

            # ===============================================================
            cuda_events_end[end_idx].record()
            tokens_end[end_idx] = data_iterator.processed_tokens
            end_idx += 1

            if (end_idx - warmup_microbatches) % logging_interval == 0:
                # Every time, we log the n microbatches before the last n microbatches
                logging_start = end_idx - logging_interval * 2
                logging_end = end_idx - logging_interval
                if logging_start > 0:
                    curr_time = (
                        cuda_events_end[logging_start].elapsed_time(
                            cuda_events_end[logging_end]
                        )
                        / 1000
                    )
                    curr_tokens = tokens_end[logging_end] - tokens_end[logging_start]
                    logging_if(
                        f"Finished {end_idx} microbatches, "
                        f"for the last {logging_interval} microbatches: "
                        f"time: {curr_time}, "
                        f"tokens: {curr_tokens}, "
                        f"throughput (tokens/s): "
                        f"{curr_tokens / curr_time:.2f}",
                        condition=parallel_state.get_pipeline_model_parallel_rank()
                        == 0,
                    )
            # ===============================================================

            if last_iteration:
                input_tensor = None
                send_backward(input_tensor_grad, recv_tensor_shapes, config)
            else:
                input_tensor = send_backward_recv_forward(
                    input_tensor_grad, recv_tensor_shapes, config
                )

    # Run cooldown backward passes.
    if not forward_only:
        for _ in range(num_warmup_microbatches):
            # Enable async grad reduction in the last backward pass
            # Note: If grad sync function is provided, only enable
            # async grad reduction in first pipeline stage. Other
            # pipeline stages do grad reduction during pipeline
            # bubble.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)
            is_current_batch_empty = empty_batch_markers.pop(0)

            output_tensor_grad = recv_backward(send_tensor_shapes, config)

            multi_lora_manager.mark_backward_pass_started()

            # For empty batches, we skip the actual backward computation but still
            # maintain pipeline synchronization
            if not is_current_batch_empty:
                # Normal backward pass for non-empty batches
                input_tensor_grad = backward_step(
                    input_tensor, output_tensor, output_tensor_grad, model_type, config
                )
            else:
                # For empty batches, we just pass through the gradient without
                # computation
                input_tensor_grad = torch.zeros(
                    (1, 1, EMPTY_BATCH_MAGIC_TOKEN),
                    dtype=config.pipeline_dtype,
                    device="cuda",
                )

            # Only perform gradient sync and optimizer step for non-empty batches
            if not is_current_batch_empty:
                multi_lora_manager.pop_oldest_batch_info()
                if config.timers is not None:
                    config.timers("optimizer-step").start()
                enable_grad_sync()
                multi_lora_manager.maybe_reduce_grad_and_optimizer_step()
                disable_grad_sync()
                if config.timers is not None:
                    config.timers("optimizer-step").stop()

            # ===============================================================
            cuda_events_end[end_idx].record()
            tokens_end[end_idx] = data_iterator.processed_tokens
            end_idx += 1

            if (end_idx - warmup_microbatches) % logging_interval == 0:
                # Every time, we log the n microbatches before the last n microbatches
                logging_start = end_idx - logging_interval * 2
                logging_end = end_idx - logging_interval
                if logging_start > 0:
                    curr_time = (
                        cuda_events_end[logging_start].elapsed_time(
                            cuda_events_end[logging_end]
                        )
                        / 1000
                    )
                    curr_tokens = tokens_end[logging_end] - tokens_end[logging_start]
                    logging_if(
                        f"Finished {end_idx} microbatches, "
                        f"for the last {logging_interval} microbatches: "
                        f"time: {curr_time}, "
                        f"tokens: {curr_tokens}, "
                        f"throughput (tokens/s): "
                        f"{curr_tokens / curr_time:.2f}",
                        condition=parallel_state.get_pipeline_model_parallel_rank()
                        == 0,
                    )
            # ===============================================================

            send_backward(input_tensor_grad, recv_tensor_shapes, config)

        # Launch any remaining grad reductions.
        if no_sync_context is not None:
            enable_grad_sync()
            if config.grad_sync_func is not None:
                config.grad_sync_func(model.parameters())

    if config.finalize_model_grads_func is not None and not forward_only:
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        config.finalize_model_grads_func(
            [model], total_num_tokens if config.calculate_per_token_loss else None
        )
        config.finalize_model_grads_func(
            multi_lora_manager.ddp_modules,
            total_num_tokens if config.calculate_per_token_loss else None,
        )

    if config.timers is not None:
        config.timers("forward-backward").stop()

    logging_results(cuda_events_end, tokens_end, timers=config.timers)

    return forward_data_store
