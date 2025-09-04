"""Baseline mLoRA schedule."""

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
from megatron.core.utils import (
    get_model_config,
    get_model_type,
    get_model_xattn,
)

from lorafusion.pipeline_parallel.multi_lora_schedule import logging_results
from lorafusion.utils.common import logging_if

# Types
Shape = list[int] | torch.Size


def baseline_mlora_forward_backward_pipelining_without_interleaving(  # noqa: C901, PLR0912, PLR0915
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
    mlora_fake_optimizer: torch.optim.Optimizer | None = None,
    global_batch_sizes: list[int] | None = None,
) -> list[torch.Tensor]:
    """Baseline mLoRA schedule."""
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
    if mlora_fake_optimizer is None:
        msg = "mlora_fake_optimizer must be provided"
        raise ValueError(msg)
    if global_batch_sizes is None:
        msg = "global_batch_size must be provided"
        raise ValueError(msg)
    num_adapters = len(global_batch_sizes)
    warmup_microbatches = 16
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
    last_optimizer_step_idx = 0
    curr_optimizer_adapter_idx = 0
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

    def reduce_grad_and_optimizer_step() -> None:
        if config.timers is not None:
            config.timers("optimizer-step").start()
        enable_grad_sync()
        model.start_grad_sync()
        model.finish_grad_sync()
        if hasattr(mlora_fake_optimizer, "step_with_ready_grads"):
            mlora_fake_optimizer.step_with_ready_grads()
        else:
            mlora_fake_optimizer.step()
        if hasattr(model, "zero_grad_buffer"):
            model.zero_grad_buffer()
        mlora_fake_optimizer.zero_grad()
        disable_grad_sync()
        if config.timers is not None:
            config.timers("optimizer-step").stop()

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
    total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()

    if not forward_only:
        input_tensors = []
        output_tensors = []
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
        send_forward(output_tensor, send_tensor_shapes, config)
        total_num_tokens += num_tokens.item()

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(
                output_tensor[0], config.deallocate_pipeline_outputs
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
            deallocate_output_tensor(
                output_tensor[0], config.deallocate_pipeline_outputs
            )

            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            # Because right now we don't need to store the forward data
            if parallel_state.is_pipeline_last_stage():
                forward_data_store.pop(0)

            # Enable grad sync for the last microbatch in the batch if the full
            # backward pass completes in the 1F1B stage.
            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )
            # ===============================================================
            if (
                end_idx + 1 - last_optimizer_step_idx
                == global_batch_sizes[curr_optimizer_adapter_idx]
            ):
                reduce_grad_and_optimizer_step()
                last_optimizer_step_idx = end_idx + 1
                curr_optimizer_adapter_idx = (
                    curr_optimizer_adapter_idx + 1
                ) % num_adapters

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

            output_tensor_grad = recv_backward(send_tensor_shapes, config)

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )
            # ===============================================================
            if (
                end_idx + 1 - last_optimizer_step_idx
                == global_batch_sizes[curr_optimizer_adapter_idx]
            ):
                reduce_grad_and_optimizer_step()
                last_optimizer_step_idx = end_idx + 1
                curr_optimizer_adapter_idx = (
                    curr_optimizer_adapter_idx + 1
                ) % num_adapters

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

    if config.timers is not None:
        config.timers("forward-backward").stop()

    logging_results(cuda_events_end, tokens_end, timers=config.timers)

    return forward_data_store
