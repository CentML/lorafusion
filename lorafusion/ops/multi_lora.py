"""Multi-LoRA manager."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from loguru import logger
from megatron.core import parallel_state

from lorafusion.ops.triton_ops.blocked_dropout import blocked_seeded_dropout
from lorafusion.ops.triton_ops.dropout import seeded_dropout
from lorafusion.ops.triton_ops.fused_multi_lora_dyw_dsa import (
    fused_multi_lora_dyw_dsa,
)
from lorafusion.ops.triton_ops.fused_multi_lora_xw_sb import (
    construct_s_and_b_ptrs_list,
    fused_multi_lora_xw_sb,
)
from lorafusion.ops.triton_ops.fused_multi_lora_dys_dyb import (
    fused_multi_lora_dys_dyb,
)

if TYPE_CHECKING:
    from peft import LoraConfig
    from peft.tuners.lora import Linear as LoRALinear

    from lorafusion.ops.multi_lora import MultiLoRAManager
    from lorafusion.train.training_utils import MicroBatchInfo


multi_lora_manager: MultiLoRAManager | None = None
THREE_DIM = 3
MULTI_LORA_BLOCK_SIZE = 64


def init_multi_lora_manager(
    lora_configs: list[LoraConfig],
    num_pipeline_stages: int,
    multi_lora_max_microbatch_tokens: int | None,
    multi_lora_global_batch_sizes: list[int] | None = None,
) -> None:
    """Initialize the multi-LoRA manager."""
    global multi_lora_manager
    if multi_lora_manager is not None:
        msg = "Multi-LoRA manager has already been initialized"
        raise ValueError(msg)
    multi_lora_manager = MultiLoRAManager(
        lora_configs,
        num_pipeline_stages,
        multi_lora_max_microbatch_tokens,
        multi_lora_global_batch_sizes,
    )
    return multi_lora_manager


def get_multi_lora_manager() -> MultiLoRAManager | None:
    """Get the multi-LoRA manager."""
    return multi_lora_manager


class MultiLoRAManager:
    """Multi-LoRA manager."""

    def __init__(
        self,
        lora_configs: list[LoraConfig],
        num_pipeline_stages: int,
        multi_lora_max_microbatch_tokens: int | None,
        multi_lora_global_batch_sizes: list[int] | None = None,
    ) -> None:
        """Initialize the Multi-LoRA manager.

        Args:
            lora_configs: The information of the LoRA.
            num_pipeline_stages: The number of pipeline stages.
            multi_lora_max_microbatch_tokens: The maximum microbatch tokens for
                multi-LoRA.
            multi_lora_global_batch_sizes: Target global batch sizes for each LoRA
                adapter. If provided, manual gradient synchronization will be triggered
                when the sample count reaches these thresholds.
        """
        self.lora_configs = lora_configs
        self.num_pipeline_stages = num_pipeline_stages
        self.multi_lora_max_microbatch_tokens = multi_lora_max_microbatch_tokens
        self.multi_lora_global_batch_sizes = multi_lora_global_batch_sizes

        # Initialize tracking for manual gradient synchronization
        if multi_lora_global_batch_sizes is None:
            msg = "multi_lora_global_batch_sizes is not provided"
            raise ValueError(msg)
        if len(multi_lora_global_batch_sizes) != len(lora_configs):
            msg = (
                f"Expected {len(lora_configs)} batch sizes, "
                f"got {len(multi_lora_global_batch_sizes)}"
            )
            raise ValueError(msg)
        self.adapter_sample_counts = {i: 0 for i in range(len(lora_configs))}

        # Initialize deque to store batch information for pipeline stages
        # Size is num_pipeline_stages to accommodate all microbatches in the pipeline
        self.batch_info_deque = deque(maxlen=num_pipeline_stages)

        # Dictionary to track LoRALinear layers
        self.lora_linear_layers = {}
        self.lora_linear_counter = 0

        # Keep track of the microbatch index
        self.microbatch_idx = 0

        # Will be set later for DDP management
        # ddp_model is the entire model
        self.ddp_model = None
        # ddp_modules is the list of faked ddp modules for each adapter
        self.ddp_modules: list[torch.nn.Module] = []
        # optimizer is the optimizer for the entire model
        self.optimizers: list[torch.optim.Optimizer] = []

        # Keep the state
        self.is_in_backward = False

    def register_ddp_modules_and_optimizers(
        self,
        ddp_model: torch.nn.Module,
        ddp_modules: list[torch.nn.Module],
        optimizers: list[torch.optim.Optimizer],
    ) -> None:
        """Register the DDP model and modules with the manager."""
        self.ddp_model = ddp_model
        self.ddp_modules = ddp_modules
        self.optimizers = optimizers

    def maybe_reduce_grad_and_optimizer_step(self) -> None:
        """Check the adapter sample counts and reduce the grad and optimizer step."""
        # If we have data parallelism, we need to all-reduce the sample counts to make
        # sure the sync is correct.
        for adapter_idx, sample_count in self.adapter_sample_counts.items():
            if parallel_state.get_data_parallel_world_size() > 1:
                # All-reduce the sample count across data parallel ranks
                sample_count_tensor = torch.tensor(
                    sample_count, dtype=torch.int, device="cuda"
                )
                torch.distributed.all_reduce(
                    sample_count_tensor,
                    op=torch.distributed.ReduceOp.SUM,
                    group=parallel_state.get_data_parallel_group(),
                )
                global_sample_count = sample_count_tensor.item()
            else:
                global_sample_count = sample_count

            if global_sample_count > self.multi_lora_global_batch_sizes[adapter_idx]:
                msg = (
                    f"Adapter {adapter_idx} has reached the global batch size "
                    f"{self.multi_lora_global_batch_sizes[adapter_idx]} with "
                    f"sample count {global_sample_count}"
                )
                raise ValueError(msg)
            if global_sample_count == self.multi_lora_global_batch_sizes[adapter_idx]:
                self.reduce_grad_and_optimizer_step(adapter_idx)
                self.adapter_sample_counts[adapter_idx] = 0

    def reduce_grad_and_optimizer_step(self, adapter_idx: int) -> None:
        """Reduce the grad and optimizer step for the adapter."""
        ddp_module = self.ddp_modules[adapter_idx]
        optimizer = self.optimizers[adapter_idx]

        # Trigger gradient synchronization
        ddp_module.start_grad_sync()
        ddp_module.finish_grad_sync()

        # Update weights according to the optimizer
        if hasattr(optimizer, "step_with_ready_grads"):
            optimizer.step_with_ready_grads()
        else:
            optimizer.step()

        # Zero gradients
        if hasattr(ddp_module, "zero_grad_buffer"):
            ddp_module.zero_grad_buffer()
        optimizer.zero_grad()

    def register_lora_linear(self, lora_linear: LoRALinear) -> LoRALinear:
        """Register a LoRALinear layer with the manager and assign it an index."""
        layer_id = id(lora_linear)
        if layer_id not in self.lora_linear_layers:
            self.lora_linear_layers[layer_id] = self.lora_linear_counter

        self.lora_linear_counter += 1

        return self.lora_linear_layers[layer_id]

    def add_batch_info(self, batch_info: MultiLoRABatchInfo) -> None:
        """Add a new MultiLoRABatchInfo to the deque.

        Args:
            batch_info: The MultiLoRABatchInfo to add.
        """
        if len(self.batch_info_deque) == self.num_pipeline_stages:
            msg = (
                f"Batch info deque is full. Current size: {len(self.batch_info_deque)}"
            )
            raise ValueError(msg)
        batch_info.microbatch_idx = self.microbatch_idx
        self.microbatch_idx += 1
        self.batch_info_deque.append(batch_info)

        # Track sample counts for each adapter in this batch - only if we're using
        # adapter-specific batch sizes for gradient synchronization
        micro_batch_info = getattr(batch_info, "micro_batch_info", None)
        if batch_info.allow_empty_micro_batch_info and micro_batch_info is None:
            return
        if micro_batch_info is None:
            msg = "MicroBatchInfo is not available"
            raise ValueError(msg)

        # Use the adapter_num_samples_pairs directly from the micro_batch_info
        for (
            adapter_idx,
            num_samples,
        ) in micro_batch_info.adapter_num_samples_pairs.items():
            if adapter_idx in self.adapter_sample_counts:
                self.adapter_sample_counts[adapter_idx] += num_samples
            else:
                msg = f"Adapter {adapter_idx} not found in adapter_sample_counts"
                raise ValueError(msg)

    def get_newest_batch_info(self) -> MultiLoRABatchInfo | None:
        """Get the newest batch info from the deque.

        Returns:
            The newest MultiLoRABatchInfo or None if the deque is empty.
        """
        if not self.batch_info_deque:
            return None
        return self.batch_info_deque[-1]

    def get_oldest_batch_info(self) -> MultiLoRABatchInfo | None:
        """Get the oldest batch info from the deque without removing it.

        Returns:
            The oldest MultiLoRABatchInfo or None if the deque is empty.
        """
        if not self.batch_info_deque:
            return None
        return self.batch_info_deque[0]

    def pop_oldest_batch_info(self) -> MultiLoRABatchInfo | None:
        """Pop the oldest batch info from the deque.

        Returns:
            The oldest MultiLoRABatchInfo or None if the deque is empty.
        """
        if not self.batch_info_deque:
            return None
        return self.batch_info_deque.popleft()

    def clear_batch_info(self) -> None:
        """Clear the batch info from the deque."""
        self.batch_info_deque.clear()

    @property
    def is_in_forward(self) -> bool:
        """Check if the current context is in forward pass."""
        return not self.is_in_backward

    def set_backward_pass_state(self, *, is_backward: bool) -> None:
        """Set whether we're currently in backward pass or forward pass.

        Args:
            is_backward: True if in backward pass, False if in forward pass
        """
        self.is_in_backward = is_backward

    def mark_backward_pass_started(self) -> bool:
        """Mark that backward pass has started.

        Returns:
            True to indicate successful state change
        """
        return self.set_backward_pass_state(is_backward=True)

    def mark_forward_pass_started(self) -> bool:
        """Mark that forward pass has started.

        Returns:
            True to indicate successful state change
        """
        return self.set_backward_pass_state(is_backward=False)


@dataclass
class MultiLoRABatchInfo:
    """Batch info for Multi-LoRA.

    Args:
        seq_len_list: The sequence length belongs to each adapter.
        lora_idx_list: The index of the adapter.
        lora_rank_list: The LoRA rank belongs to each adapter.
        dropout_p_list: The dropout probability belongs to each adapter.
        alpha_list: The scaling factor belongs to each adapter.
        block_size_m: The block size for the multi-linear LoRA.
        padded_seq_len_list: The padded sequence length belongs to each adapter.
        block_to_lookup_table: The lookup table for the adapter index.
            Each row represents a block of input batch, where each row is a list
            of (lora_rank, valid_size_in_block, start_m), which means the LoRA
            rank, the actual number of active tokens in this block, and the offset
            of pid_m for sequence s.
        block_to_dropout_p: The dropout probability belongs to each adapter in each
            block.
        block_to_alpha: The scaling factor belongs to each adapter in each block.
        enable_dropout: Whether to enable dropout.
        max_r: The maximum LoRA rank.
        micro_batch_info: Reference to the original MicroBatchInfo object that
            contains detailed adapter sample information.
    """

    # Input batch info
    seq_len_list: list[int]
    lora_idx_list: list[int]
    lora_rank_list: list[int]
    dropout_p_list: list[float]
    alpha_list: list[float]
    block_size_m: int
    # Processed batch info
    padded_seq_len_list: list[int]
    block_to_lookup_table: torch.Tensor
    block_to_dropout_p: torch.Tensor
    block_to_alpha: torch.Tensor
    enable_dropout: bool
    same_dropout_p_value: float | None
    max_r: int
    # Index
    microbatch_idx: int | None = None
    # Original batch info
    micro_batch_info: MicroBatchInfo | None = None
    allow_empty_micro_batch_info: bool = False

    @property
    def num_active_adapters(self) -> int:
        """The number of active adapters."""
        return len(set(self.lora_idx_list))


def prepare_multi_lora_batch_info(
    seq_len_list: list[int],
    lora_idx_list: list[int],
    lora_rank_list: list[int],
    dropout_p_list: list[float],
    alpha_list: list[float],
    block_size_m: int,
    output_dtype: torch.dtype = torch.bfloat16,
    micro_batch_info: MicroBatchInfo | None = None,
    *,
    allow_empty_micro_batch_info: bool = False,
) -> MultiLoRABatchInfo:
    """Prepare the input tensors for the fused Multi-LoRA xw + sb kernel."""
    # Return:
    # 1. padded_seq_len_list: total sequence length
    # 2. block_to_lookup_table: (contains the r, valid_size, start_m for each block)
    # 3. block_to_dropout_p: (contains the dropout_p for each block)
    # 4. block_to_alpha: (contains the alpha for each block)
    # 5. max_r: The maximum LoRA rank.
    # 6. enable_dropout: Whether to enable dropout.

    # First, pad the seq_len_list to the multiple of block_size_m, also
    # calculate the strides for each m-th block.
    padded_seq_len_list = [
        ((seq_len - 1) // block_size_m + 1) * block_size_m for seq_len in seq_len_list
    ]
    # Because it's already padded, we can directly calculate the num_pid_m.
    num_blocks = sum(padded_seq_len_list) // block_size_m
    # Masks for each m-th block.
    # For example, if seq_len_list = [129, 130, 128],
    # then padded_seq_len_list = [256, 256, 128],
    # and the valid_size_list = [128, 1, 128, 2, 128]

    # Calculate the total number of blocks needed
    total_blocks = sum(padded_seq_len_list) // block_size_m

    # Initialize lookup table, and alpha list
    # For each block, we need to store:
    # > [R, valid_size, s_offset_m, a_stride_k, a_stride_r,
    #                               s_stride_m, s_stride_k, b_stride_r, b_stride_n]
    # >>[R, valid_size, s_offset_m, 1, K, R, 1, 1, R]
    # Therefore, we actually only need to store R and valid_size
    block_to_lookup_table = torch.zeros(
        (total_blocks, 3), dtype=torch.int32, device="cpu"
    )
    block_to_dropout_p = torch.zeros(total_blocks, dtype=output_dtype, device="cpu")
    block_to_alpha = torch.zeros(total_blocks, dtype=output_dtype, device="cpu")

    # Fill in lookup table and alpha list
    block_idx = 0
    max_r = max(lora_rank_list)
    enable_dropout = any(dropout_p > 0 for dropout_p in dropout_p_list)
    same_dropout_p = all(dropout_p == dropout_p_list[0] for dropout_p in dropout_p_list)
    same_dropout_p_value = dropout_p_list[0] if same_dropout_p else None
    for seq_idx, (seq_len, lora_rank, dropout_p, alpha) in enumerate(
        zip(seq_len_list, lora_rank_list, dropout_p_list, alpha_list, strict=True)
    ):
        # Calculate number of blocks for this sequence
        num_blocks = padded_seq_len_list[seq_idx] // block_size_m
        start_m = block_idx

        # For each block in this sequence
        for i in range(num_blocks):
            # Calculate the actual mask size for this block (handle padding)
            valid_size = min(block_size_m, seq_len - i * block_size_m)

            # Store information in lookup table
            block_to_lookup_table[block_idx, 0] = lora_rank  # LoRA rank
            block_to_lookup_table[block_idx, 1] = (
                valid_size  # Actual number of active rows
            )
            block_to_lookup_table[block_idx, 2] = start_m  # Start m

            # Store alpha value
            block_to_dropout_p[block_idx] = dropout_p
            block_to_alpha[block_idx] = alpha

            block_idx += 1

    block_to_lookup_table = block_to_lookup_table.to(device="cuda")
    block_to_dropout_p = block_to_dropout_p.to(device="cuda")
    block_to_alpha = block_to_alpha.to(device="cuda")

    # Verify we've filled all blocks
    if block_idx != total_blocks:
        msg = f"Expected {total_blocks} blocks, but filled {block_idx}"
        logger.warning(msg)
        raise ValueError(msg)

    return MultiLoRABatchInfo(
        seq_len_list=seq_len_list,
        lora_idx_list=lora_idx_list,
        lora_rank_list=lora_rank_list,
        dropout_p_list=dropout_p_list,
        alpha_list=alpha_list,
        block_size_m=block_size_m,
        padded_seq_len_list=padded_seq_len_list,
        block_to_lookup_table=block_to_lookup_table,
        block_to_dropout_p=block_to_dropout_p,
        block_to_alpha=block_to_alpha,
        enable_dropout=enable_dropout,
        same_dropout_p_value=same_dropout_p_value,
        max_r=max_r,
        micro_batch_info=micro_batch_info,
        allow_empty_micro_batch_info=allow_empty_micro_batch_info,
    )


def _fused_linear_multi_lora_forward(
    padded_x: torch.Tensor,
    linear_w: torch.Tensor,
    *,
    lora_a_list: list[torch.Tensor | None],
    lora_b_list: list[torch.Tensor | None],
    seq_len_list: list[int],
    padded_seq_len_list: list[int],
    block_to_lookup_table: torch.Tensor,
    block_to_dropout_p: torch.Tensor,
    block_to_alpha: torch.Tensor,
    enable_dropout: bool,
    same_dropout_p_value: float | None,
    max_r: int,
    seed: int,
    linear_bias: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    list[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    """Forward pass."""
    if enable_dropout:
        if same_dropout_p_value is not None:
            masked_scaled_x, dropout_mask = seeded_dropout(
                x=padded_x,
                p=same_dropout_p_value,
                seed=seed,
                store_mask=True,
            )
        else:
            masked_scaled_x, dropout_mask = blocked_seeded_dropout(
                x=padded_x,
                block_to_dropout_p=block_to_dropout_p,
                block_size=MULTI_LORA_BLOCK_SIZE,
                seed=seed,
                store_mask=True,
            )
    else:
        masked_scaled_x = padded_x
        dropout_mask = None

    # Calculate the s list for each block
    full_s = padded_x @ (torch.cat(lora_a_list, dim=0).T)
    curr_m, curr_r = 0, 0
    s_list = []
    for padded_seq_len, lora_a_tensor in zip(
        padded_seq_len_list, lora_a_list, strict=True
    ):
        lora_rank = lora_a_tensor.shape[0]
        s_list.append(
            full_s[curr_m : curr_m + padded_seq_len, curr_r : curr_r + lora_rank]
        )
        curr_m += padded_seq_len
        curr_r += lora_rank

    # s stride / total r dim
    total_r = full_s.shape[1]

    # Construct the s and b ptrs list
    s_ptrs_list, b_ptrs_list, _, _ = construct_s_and_b_ptrs_list(
        raw_s_list=s_list,
        raw_b_list=lora_b_list,
    )

    y = fused_multi_lora_xw_sb(
        x=masked_scaled_x,
        w=linear_w,
        s_ptrs_list=s_ptrs_list,
        b_ptrs_list=b_ptrs_list,
        block_to_lookup_table=block_to_lookup_table,
        block_to_alpha=block_to_alpha,
        max_r=max_r,
        total_r=total_r,
        bias=linear_bias,
    )
    return y, dropout_mask, masked_scaled_x, s_list, s_ptrs_list, b_ptrs_list


def _fused_linear_multi_lora_backward(
    dy: torch.Tensor,
    linear_w: torch.Tensor,
    *,
    s_ptrs_list: torch.Tensor,
    b_ptrs_list: torch.Tensor,
    lora_a_list: list[torch.Tensor | None],
    lora_b_list: list[torch.Tensor | None],
    s_list: list[torch.Tensor],
    seq_len_list: list[int],
    padded_seq_len_list: list[int],
    masked_scaled_x: torch.Tensor,
    block_to_lookup_table: torch.Tensor,
    block_to_dropout_p: torch.Tensor,
    block_to_alpha: torch.Tensor,
    enable_dropout: bool,
    dropout_mask: torch.Tensor | None,
    max_r: int,
    requires_dx: bool = True,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Backward pass."""
    # Calculate db_list and ds_list using fused operation
    total_r = sum(lora_b.shape[1] for lora_b in lora_b_list if lora_b is not None)
    db_list, ds_list = fused_multi_lora_dys_dyb(
        dy=dy,
        s_ptrs_list=s_ptrs_list,
        b_ptrs_list=b_ptrs_list,
        raw_s_list=s_list,
        raw_b_list=lora_b_list,
        block_to_lookup_table=block_to_lookup_table,
        block_to_alpha=block_to_alpha,
        max_r=max_r,
        total_r=total_r,
    )

    # Calculate da_list from ds and masked_scaled_x
    da_list = []
    curr_m = 0
    for padded_seq_len, ds in zip(padded_seq_len_list, ds_list, strict=True):
        da = ds.T @ masked_scaled_x[curr_m : curr_m + padded_seq_len]
        da_list.append(da)
        curr_m += padded_seq_len

    # Calculate dx using fused operation
    if requires_dx:
        dx = fused_multi_lora_dyw_dsa(
            dy=dy,
            w=linear_w,
            raw_ds_list=ds_list,
            raw_a_list=lora_a_list,
            block_to_lookup_table=block_to_lookup_table,
            block_to_dropout_p=block_to_dropout_p,
            enable_dropout=enable_dropout,
            dropout_mask=dropout_mask,
            max_r=max_r,
        )
    else:
        dx = None

    return dx, da_list, db_list, ds_list


class FusedLinearMultiLoRA(torch.autograd.Function):
    """Fused linear and multi-linear LoRA."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionContext,
        padded_x: torch.Tensor,
        linear_w: torch.Tensor,
        lora_a_0: torch.Tensor | None,
        lora_a_1: torch.Tensor | None,
        lora_a_2: torch.Tensor | None,
        lora_a_3: torch.Tensor | None,
        lora_b_0: torch.Tensor | None,
        lora_b_1: torch.Tensor | None,
        lora_b_2: torch.Tensor | None,
        lora_b_3: torch.Tensor | None,
        seq_len_list: list[int],
        padded_seq_len_list: list[int],
        block_to_lookup_table: torch.Tensor,
        block_to_dropout_p: torch.Tensor,
        block_to_alpha: torch.Tensor,
        enable_dropout: bool,  # noqa: FBT001
        same_dropout_p_value: float | None,
        max_r: int,
        seed: int | None,
        linear_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Forward pass."""
        if seed is None:
            seed = random.randrange(int(1e6))  # noqa: S311

        if padded_x.ndim != THREE_DIM:
            msg = f"padded_x must be a 3D tensor, but got {padded_x.ndim}"
            raise ValueError(msg)

        bsz, seq_len, n = padded_x.shape
        if bsz != 1:
            msg = "currently, we only support batch size 1 for multi-linear LoRA"
            raise ValueError(msg)

        padded_x = padded_x.reshape(seq_len, n)

        lora_a_list = [lora_a_0, lora_a_1, lora_a_2, lora_a_3]
        lora_b_list = [lora_b_0, lora_b_1, lora_b_2, lora_b_3]
        lora_a_list = [lora_a for lora_a in lora_a_list if lora_a is not None]
        lora_b_list = [lora_b for lora_b in lora_b_list if lora_b is not None]

        y, dropout_mask, masked_scaled_x, s_list, s_ptrs_list, b_ptrs_list = (
            _fused_linear_multi_lora_forward(
                padded_x=padded_x,
                linear_w=linear_w,
                lora_a_list=lora_a_list,
                lora_b_list=lora_b_list,
                seq_len_list=seq_len_list,
                padded_seq_len_list=padded_seq_len_list,
                block_to_lookup_table=block_to_lookup_table,
                block_to_dropout_p=block_to_dropout_p,
                block_to_alpha=block_to_alpha,
                enable_dropout=enable_dropout,
                same_dropout_p_value=same_dropout_p_value,
                max_r=max_r,
                seed=seed,
                linear_bias=linear_bias,
            )
        )
        ctx.save_for_backward(
            masked_scaled_x,
            linear_w,
            dropout_mask,
            block_to_lookup_table,
            block_to_dropout_p,
            block_to_alpha,
            s_ptrs_list,
            b_ptrs_list,
        )
        ctx.lora_a_list = lora_a_list
        ctx.lora_b_list = lora_b_list
        ctx.s_list = s_list
        ctx.seq_len_list = seq_len_list
        ctx.padded_seq_len_list = padded_seq_len_list
        ctx.enable_dropout = enable_dropout
        ctx.same_dropout_p_value = same_dropout_p_value
        ctx.max_r = max_r
        ctx.seed = seed
        return y.reshape(1, seq_len, -1)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionContext,
        dy: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Backward pass."""
        (
            masked_scaled_x,
            linear_w,
            dropout_mask,
            block_to_lookup_table,
            block_to_dropout_p,
            block_to_alpha,
            s_ptrs_list,
            b_ptrs_list,
        ) = ctx.saved_tensors
        lora_a_list = ctx.lora_a_list
        lora_b_list = ctx.lora_b_list
        s_list = ctx.s_list

        if dy.ndim != THREE_DIM:
            msg = f"dy must be a 3D tensor, but got {dy.ndim}"
            raise ValueError(msg)

        bsz, seq_len, n = dy.shape
        if bsz != 1:
            msg = "currently, we only support batch size 1 for multi-linear LoRA"
            raise ValueError(msg)

        dy = dy.reshape(seq_len, n)

        dx, da_list, db_list, ds_list = _fused_linear_multi_lora_backward(
            dy=dy,
            linear_w=linear_w,
            lora_a_list=lora_a_list,
            lora_b_list=lora_b_list,
            s_ptrs_list=s_ptrs_list,
            b_ptrs_list=b_ptrs_list,
            s_list=s_list,
            seq_len_list=ctx.seq_len_list,
            padded_seq_len_list=ctx.padded_seq_len_list,
            masked_scaled_x=masked_scaled_x,
            block_to_lookup_table=block_to_lookup_table,
            block_to_dropout_p=block_to_dropout_p,
            block_to_alpha=block_to_alpha,
            enable_dropout=ctx.enable_dropout,
            dropout_mask=dropout_mask,
            max_r=ctx.max_r,
            requires_dx=ctx.needs_input_grad[0],
        )
        if dx is not None:
            dx = dx.reshape(1, seq_len, -1)

        da_list.extend([None] * (4 - len(da_list)))
        db_list.extend([None] * (4 - len(db_list)))

        return (
            dx,
            None,
            *da_list,
            *db_list,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def fused_linear_multi_lora(
    padded_x: torch.Tensor,
    linear_w: torch.Tensor,
    *,
    lora_a_list: list[torch.Tensor],
    lora_b_list: list[torch.Tensor],
    seq_len_list: list[int],
    padded_seq_len_list: list[int],
    block_to_lookup_table: torch.Tensor,
    block_to_dropout_p: torch.Tensor,
    block_to_alpha: torch.Tensor,
    enable_dropout: bool,
    same_dropout_p_value: float | None,
    max_r: int,
    linear_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused multi-linear LoRA.

    Args:
        padded_x: (bsz, seq_len, n)
        linear_w: (n, k)
        lora_a_list: list of (r, n)
        lora_b_list: list of (r, k)
        seq_len_list: list of int
        padded_seq_len_list: list of int
        block_to_lookup_table: (total_blocks, 3)
        block_to_dropout_p: (total_blocks,)
        block_to_alpha: (total_blocks,)
        enable_dropout: bool
        same_dropout_p_value: float | None
        max_r: int
        linear_bias: The bias tensor of the linear layer.
    """
    # Because Autograd.Function does not support gradients for list arguments,
    # so we manually implement that.
    if len(lora_a_list) != len(lora_b_list):
        msg = "lora_a_list and lora_b_list must have the same length"
        raise ValueError(msg)
    if len(lora_a_list) == 0:
        msg = "lora_a_list and lora_b_list must not be empty"
        raise ValueError(msg)
    if len(lora_a_list) > 4:  # noqa: PLR2004
        msg = "currently, we only support up to 4 simultaneous LoRA adapters"
        raise ValueError(msg)
    for _ in range(4 - len(lora_a_list)):
        lora_a_list.append(None)
        lora_b_list.append(None)
    return FusedLinearMultiLoRA.apply(
        padded_x,
        linear_w,
        lora_a_list[0],
        lora_a_list[1],
        lora_a_list[2],
        lora_a_list[3],
        lora_b_list[0],
        lora_b_list[1],
        lora_b_list[2],
        lora_b_list[3],
        seq_len_list,
        padded_seq_len_list,
        block_to_lookup_table,
        block_to_dropout_p,
        block_to_alpha,
        enable_dropout,
        same_dropout_p_value,
        max_r,
        None,  # seed
        linear_bias,
    )


if __name__ == "__main__":
    import torch

    hidden_size = 8192
    seed = 42

    seq_len_list = [3072, 1024]
    lora_idx_list = [0, 1]
    lora_rank_list = [16, 16]
    dropout_p_list = [0.1, 0.1]
    alpha_list = [16.0, 16.0]

    multi_lora_batch_info = prepare_multi_lora_batch_info(
        seq_len_list=seq_len_list,
        lora_idx_list=lora_idx_list,
        lora_rank_list=lora_rank_list,
        dropout_p_list=dropout_p_list,
        alpha_list=alpha_list,
        block_size_m=MULTI_LORA_BLOCK_SIZE,
    )
    padded_seq_len_list = multi_lora_batch_info.padded_seq_len_list
    block_to_lookup_table = multi_lora_batch_info.block_to_lookup_table
    block_to_dropout_p = multi_lora_batch_info.block_to_dropout_p
    block_to_alpha = multi_lora_batch_info.block_to_alpha
    enable_dropout = multi_lora_batch_info.enable_dropout
    same_dropout_p_value = multi_lora_batch_info.same_dropout_p_value
    max_r = multi_lora_batch_info.max_r

    padded_x = torch.randn(
        (1, sum(padded_seq_len_list), hidden_size),
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )
    linear_w = torch.randn(
        hidden_size, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    lora_a_list = [
        torch.randn(
            lora_rank,
            hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )
        for lora_rank in multi_lora_batch_info.lora_rank_list
    ]
    lora_b_list = [
        torch.randn(
            hidden_size,
            lora_rank,
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )
        for lora_rank in multi_lora_batch_info.lora_rank_list
    ]

    test_step_by_step = False

    if test_step_by_step:
        with torch.no_grad():
            # Forward pass - blocked_seeded_dropout
            test_same_dropout_p = False
            if test_same_dropout_p:
                masked_scaled_x, dropout_mask = seeded_dropout(
                    x=padded_x,
                    p=same_dropout_p_value,
                    seed=seed,
                    store_mask=True,
                )
            else:
                masked_scaled_x, dropout_mask = blocked_seeded_dropout(
                    x=padded_x,
                    block_to_dropout_p=block_to_dropout_p,
                    block_size_m=MULTI_LORA_BLOCK_SIZE,
                    seed=seed,
                    store_mask=True,
                )

            # Forward pass
            full_s = padded_x @ (torch.cat(lora_a_list, dim=0).T)
            curr_m, curr_r = 0, 0
            s_list = []
            for padded_seq_len, lora_a_tensor in zip(
                padded_seq_len_list, lora_a_list, strict=True
            ):
                lora_rank = lora_a_tensor.shape[0]
                s_list.append(
                    full_s[
                        curr_m : curr_m + padded_seq_len, curr_r : curr_r + lora_rank
                    ]
                )
                curr_m += padded_seq_len
                curr_r += lora_rank

            # Forward pass - fused_multi_lora_xw_sb

            total_r = full_s.shape[1]

            result = fused_multi_lora_xw_sb(
                x=padded_x,
                w=linear_w,
                raw_s_list=s_list,
                raw_b_list=lora_b_list,
                block_to_lookup_table=block_to_lookup_table,
                block_to_alpha=block_to_alpha,
                max_r=max_r,
                total_r=total_r,
            )

            dy = torch.ones_like(result)
            db_list, ds_list = fused_multi_lora_dys_dyb(
                dy=dy,
                raw_s_list=s_list,
                raw_b_list=lora_b_list,
                block_to_lookup_table=block_to_lookup_table,
                block_to_alpha=block_to_alpha,
                max_r=max_r,
                total_r=total_r,
            )

            curr_m = 0
            da_list = []
            for padded_seq_len, ds in zip(padded_seq_len_list, ds_list, strict=True):
                da = ds.T @ masked_scaled_x[curr_m : curr_m + padded_seq_len]
                da_list.append(da)
                curr_m += padded_seq_len

            dx = fused_multi_lora_dyw_dsa(
                dy=dy,
                w=linear_w,
                raw_ds_list=ds_list,
                raw_a_list=lora_a_list,
                block_to_lookup_table=block_to_lookup_table,
                block_to_dropout_p=block_to_dropout_p,
                enable_dropout=enable_dropout,
                dropout_mask=dropout_mask,
                max_r=max_r,
            )

    test_fused_func = True
    if test_fused_func:
        y = fused_linear_multi_lora(
            padded_x=padded_x,
            linear_w=linear_w,
            lora_a_list=lora_a_list,
            lora_b_list=lora_b_list,
            seq_len_list=seq_len_list,
            padded_seq_len_list=padded_seq_len_list,
            block_to_lookup_table=block_to_lookup_table,
            block_to_dropout_p=block_to_dropout_p,
            block_to_alpha=block_to_alpha,
            enable_dropout=True,
            same_dropout_p_value=0.1,
            max_r=max_r,
            linear_bias=None,
        )
        y.sum().backward()
