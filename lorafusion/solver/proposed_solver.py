"""Proposed solver 5: micro-MILP with multi-processing."""

from __future__ import annotations

import math
import multiprocessing
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from itertools import product

import numpy as np
import pulp
from loguru import logger
from tqdm import tqdm

from lorafusion.ops.triton_ops.config import get_lora_kernel_config

ZERO_POINT_FIVE = 0.5
M = 1000000

ADAPTER_PADDING_MULTIPLE = get_lora_kernel_config("fused_multi_lora_block_size_m")

def check_adapter_global_batch_idx_consistency(
    list_of_micro_batch_infos: list[MicroBatchInfo],
) -> bool:
    """Check if the adapter global batch index is consistent across micro-batches."""
    flattened = [
        (adapter_idx, global_batch_idx, sample_idx)
        for micro_batch in list_of_micro_batch_infos
        for (adapter_idx, global_batch_idx, sample_idx) in micro_batch.data_samples
    ]
    adapter_indices = {indices[0] for indices in flattened}
    for targeted_adapter_idx in adapter_indices:
        global_batch_indices = {
            global_batch_idx
            for adapter_idx, global_batch_idx, _ in flattened
            if adapter_idx == targeted_adapter_idx
        }
        if len(global_batch_indices) > 1:
            logger.warning(
                f"Inconsistent adapter global batch index: {global_batch_indices} "
                f"for adapter {targeted_adapter_idx}."
            )
            return False
    return True


@dataclass(slots=True)
class MicroBatchInfo:
    """Information about a micro-batch.

    Args:
        data_indices: List of (adapter_idx, global_batch_idx, sample_idx)
        micro_batch_idx: The index of the micro-batch in the schedule
    """

    # All following fields are sorted by the padded_total_tokens
    # > [adapter_idx, global_batch_idx, sample_idx, num_tokens]
    data_samples: dict[tuple[int, int, int], int]
    max_microbatch_size: int
    adapter_padding_multiple: int
    adapter_group_info: set[int] | None = None
    adapter_global_batch_idx_pairs: set[tuple[int, int]] | None = None
    adapter_token_lengths_pairs: dict[int, list[int]] | None = None
    adapter_num_samples_pairs: dict[int, int] | None = None
    adapter_num_tokens_pairs: dict[int, int] | None = None
    total_tokens: int | None = None
    padded_adapter_num_tokens_pairs: dict[int, int] | None = None
    padded_total_tokens: int | None = None
    # Fixed fields
    micro_batch_idx: int | None = None
    is_empty_marker: bool = False

    def __post_init__(self) -> None:
        """Post-initialization hook."""
        self.self_update()

    def self_update(self) -> None:
        """Update the MicroBatchInfo object from the data_samples."""
        self.update_from_data_samples(self.data_samples)

    def update_from_data_samples(
        self, data_samples: dict[tuple[int, int, int], int]
    ) -> None:
        """Update the MicroBatchInfo object from the data_samples."""
        self.data_samples = data_samples

        # Update the adapter group info.
        self.adapter_group_info = {a_idx for a_idx, _, _ in data_samples}

        # Update the adapter global batch idx pairs.
        self.adapter_global_batch_idx_pairs = {
            (a_idx, global_batch_idx) for a_idx, global_batch_idx, _ in data_samples
        }

        # Update the adapter token lengths pairs.
        self.adapter_token_lengths_pairs = {
            target_adapter_idx: [
                tokens
                for (a_idx, _, _), tokens in data_samples.items()
                if a_idx == target_adapter_idx
            ]
            for target_adapter_idx in self.adapter_group_info
        }

        # Update the adapter num samples pairs.
        self.adapter_num_samples_pairs = {
            a_idx: len(samples)
            for a_idx, samples in self.adapter_token_lengths_pairs.items()
        }

        # Update the adapter num tokens pairs.
        self.adapter_num_tokens_pairs = {
            a_idx: sum(tokens)
            for a_idx, tokens in self.adapter_token_lengths_pairs.items()
        }

        # Total tokens.
        self.total_tokens = sum(self.adapter_num_tokens_pairs.values())

        # Padded adapter num tokens pairs.
        self.padded_adapter_num_tokens_pairs = {
            a_idx: _round_up(num_tokens, self.adapter_padding_multiple)
            for a_idx, num_tokens in self.adapter_num_tokens_pairs.items()
        }

        # Padded total tokens.
        self.padded_total_tokens = sum(self.padded_adapter_num_tokens_pairs.values())

    @classmethod
    def from_raw_data_indices(
        cls,
        raw_data_indices: list[tuple[int, int]],
        local_batch_data: list[list[int]],
        max_microbatch_size: int,
        adapter_padding_multiple: int = ADAPTER_PADDING_MULTIPLE,
        global_batch_idx: int | None = None,
        adapter_mapping: list[int] | None = None,
    ) -> MicroBatchInfo:
        """Create a MicroBatchInfo object from raw data indices."""
        num_adapters = len(local_batch_data)
        if adapter_mapping is None:
            adapter_mapping = list(range(num_adapters))
        if global_batch_idx is None:
            global_batch_idx = 0

        # data_samples collects all the samples in the micro-batch
        # contains list of (adapter_idx, global_batch_idx, sample_idx, num_tokens).
        data_samples = {
            (
                adapter_mapping[adapter_idx],
                global_batch_idx,
                sample_idx,
            ): local_batch_data[adapter_idx][sample_idx]
            for adapter_idx, sample_idx in raw_data_indices
        }

        return cls(
            data_samples=data_samples,
            max_microbatch_size=max_microbatch_size,
            adapter_padding_multiple=adapter_padding_multiple,
        )

    def full_merge(self, other: MicroBatchInfo) -> MicroBatchInfo:
        """Merge two micro-batches."""
        if not check_adapter_global_batch_idx_consistency([self, other]):
            msg = f"Overlapping adapters! {self.data_samples} & {other.data_samples}"
            raise ValueError(msg)

        self.data_samples.update(other.data_samples)
        self.self_update()
        return self

    def partial_merge(
        self, other: MicroBatchInfo
    ) -> tuple[MicroBatchInfo, MicroBatchInfo]:
        """Partially merge samples from other micro-batch into this one.

        Try to transfer as many samples as possible from the other micro-batch
        to this one without exceeding the max_microbatch_size.

        Args:
            other: The micro-batch to partially merge from

        Returns:
            tuple: (self, remaining_samples)
                - self: The updated current micro-batch
                - remaining_samples: A new micro-batch containing samples that couldn't
                  be merged
        """
        # There should be no overlap between the adapters
        # Check if the same adapter with different global batch indices are being merged
        self_adapters = set(self.adapter_group_info)
        other_adapters = set(other.adapter_group_info)
        for adapter_id in self_adapters & other_adapters:
            self_global_batches = {
                adapter_id[1]
                for adapter_id in self.data_samples
                if adapter_id[0] == adapter_id
            }
            other_global_batches = {
                adapter_id[1]
                for adapter_id in other.data_samples
                if adapter_id[0] == adapter_id
            }
            all_global_batches = self_global_batches | other_global_batches
            # There should be only one global batch index
            if len(all_global_batches) > 1:
                msg = (
                    f"Cannot merge samples from the same adapter {adapter_id} with "
                    f"overlapping global batch indices: {all_global_batches}"
                )
                raise ValueError(msg)

        # Calculate available space in the current micro-batch
        available_space = self.max_microbatch_size - self.padded_total_tokens

        # Get adapter padding multiple from any existing padded adapter tokens
        # This assumes all adapters use the same padding multiple
        adapter_padding_multiple = ADAPTER_PADDING_MULTIPLE  # Default value

        # Sort samples by size (smallest first to maximize how many we can merge)
        sorted_samples = sorted(
            other.data_samples.items(),
            key=lambda x: x[1],  # Sort by token length
        )

        # Initialize dictionaries for transferable and remaining samples
        transferable_samples = {}
        remaining_samples = {}

        # Keep track of adapters and their token counts
        transferable_adapter_tokens = defaultdict(int)
        remaining_adapter_tokens = defaultdict(int)

        # Current padded size of transferable samples
        current_padded_size = 0

        # Process each sample to determine if it can be transferred
        for (adapter_idx, global_batch_idx, sample_idx), token_length in sorted_samples:
            # Calculate how much this sample would add to the padded size
            # We need to consider padding at the adapter level
            prev_adapter_tokens = transferable_adapter_tokens[adapter_idx]
            new_adapter_tokens = prev_adapter_tokens + token_length

            prev_padded = (
                _round_up(prev_adapter_tokens, adapter_padding_multiple)
                if prev_adapter_tokens > 0
                else 0
            )
            new_padded = _round_up(new_adapter_tokens, adapter_padding_multiple)

            padding_increase = new_padded - prev_padded

            # Check if this sample can fit
            if current_padded_size + padding_increase <= available_space:
                # This sample can be transferred
                transferable_samples[(adapter_idx, global_batch_idx, sample_idx)] = (
                    token_length
                )
                transferable_adapter_tokens[adapter_idx] = new_adapter_tokens
                current_padded_size += padding_increase
            else:
                # This sample must remain
                remaining_samples[(adapter_idx, global_batch_idx, sample_idx)] = (
                    token_length
                )
                remaining_adapter_tokens[adapter_idx] += token_length

        # If no samples can be transferred, return original micro-batches
        if not transferable_samples:
            return self, other

        # Create a new MicroBatchInfo for remaining samples if any
        if remaining_samples:
            # Calculate padded tokens for remaining samples
            remaining_micro_batch = MicroBatchInfo(
                data_samples=remaining_samples,
                max_microbatch_size=self.max_microbatch_size,
                adapter_padding_multiple=self.adapter_padding_multiple,
                micro_batch_idx=other.micro_batch_idx,
            )
        else:
            # All samples could be transferred
            remaining_micro_batch = None

        # Update this micro-batch with transferred samples
        for (
            adapter_idx,
            global_batch_idx,
            sample_idx,
        ), token_length in transferable_samples.items():
            self.data_samples[(adapter_idx, global_batch_idx, sample_idx)] = (
                token_length
            )

        # Update adapter token counts
        for adapter_idx, tokens in transferable_adapter_tokens.items():
            if adapter_idx in self.adapter_num_tokens_pairs:
                self.adapter_num_tokens_pairs[adapter_idx] += tokens
            else:
                self.adapter_num_tokens_pairs[adapter_idx] = tokens

            # Update padded tokens
            self.padded_adapter_num_tokens_pairs[adapter_idx] = _round_up(
                self.adapter_num_tokens_pairs[adapter_idx], adapter_padding_multiple
            )

        # Update total tokens
        self.total_tokens += sum(transferable_adapter_tokens.values())

        # Recalculate padded total tokens
        self.padded_total_tokens = sum(self.padded_adapter_num_tokens_pairs.values())

        # Update adapter group info
        self.adapter_group_info = self.adapter_group_info | set(
            transferable_adapter_tokens.keys()
        )

        return self, remaining_micro_batch


@dataclass(slots=True)
class AdapterGroupStepInfo:
    """Information about an adapter group step."""

    adapter_group_info: set[int]
    internal_adapter_start_end_indices: dict[int, tuple[int, int]]
    micro_batch_infos: list[MicroBatchInfo]

    def self_update(self) -> None:
        """Update the AdapterGroupStepInfo object from the micro_batch_infos."""
        self.update_from_self_micro_batch_infos(self.micro_batch_infos)

    def update_from_self_micro_batch_infos(
        self, micro_batch_infos_list: list[MicroBatchInfo]
    ) -> None:
        """Create the object from a list of MicroBatchInfo objects."""
        # Handle the case where micro_batch_infos_list is empty
        if not micro_batch_infos_list:
            self.adapter_group_info = set()
            self.internal_adapter_start_end_indices = {}
            self.micro_batch_infos = []
            return

        adapter_group_info: set[int] = micro_batch_infos_list[0].adapter_group_info
        # Sort the micro_batch_infos_list by the padded_total_tokens
        # preserve the order if padded_total_tokens is the same, using descending order
        micro_batch_infos_list.sort(
            key=lambda x: x.padded_total_tokens,
            reverse=True,
        )
        internal_adapter_start_end_indices: dict[int, tuple[int, int]] = {
            adapter_idx: (-1, -1) for adapter_idx in adapter_group_info
        }
        used_adapters_list = [
            list(micro_batch_info.adapter_num_tokens_pairs.keys())
            for micro_batch_info in micro_batch_infos_list
        ]

        # Track the start and end indices for each adapter
        for adapter_idx in adapter_group_info:
            # Find the first microbatch that uses this adapter
            for i, used_adapters in enumerate(used_adapters_list):
                if adapter_idx in used_adapters:
                    internal_adapter_start_end_indices[adapter_idx] = (i, -1)
                    break

            # Find the last microbatch that uses this adapter
            for i in range(len(used_adapters_list) - 1, -1, -1):
                if adapter_idx in used_adapters_list[i]:
                    # Update the end index in the tuple
                    start_idx = internal_adapter_start_end_indices[adapter_idx][0]
                    internal_adapter_start_end_indices[adapter_idx] = (start_idx, i)
                    break
        self.adapter_group_info = adapter_group_info
        self.internal_adapter_start_end_indices = internal_adapter_start_end_indices
        self.micro_batch_infos = micro_batch_infos_list

    @classmethod
    def from_micro_batch_infos_list(
        cls,
        micro_batch_infos_list: list[MicroBatchInfo],
    ) -> AdapterGroupStepInfo:
        """Create the object from a list of MicroBatchInfo objects."""
        adapter_group_step_info = cls(
            adapter_group_info=None,
            internal_adapter_start_end_indices=None,
            micro_batch_infos=None,
        )
        adapter_group_step_info.update_from_self_micro_batch_infos(
            micro_batch_infos_list
        )
        return adapter_group_step_info


def apply_merge_pass_for_adapter_group_step_infos(  # noqa: C901, PLR0912
    adapter_group_step_infos: list[AdapterGroupStepInfo],
    num_pipeline_stages: int,
) -> list[AdapterGroupStepInfo]:
    """Apply the merge pass for the adapter group step infos.

    This function attempts to merge micro-batches across consecutive adapter groups
    to reduce the total number of micro-batches while respecting capacity constraints
    and the bubble lemma.

    It supports two types of merging:
    1. Complete merging: When two micro-batches can be fully merged while respecting
       capacity constraints
    2. Partial merging: When only some samples from one micro-batch can be transferred
       to another while respecting capacity constraints

    The bubble lemma ensures that between two executions of the same adapter group,
    there are at least (num_pipeline_stages - 1) micro-batches from other adapter
    groups, which is necessary to prevent pipeline bubbles.

    Args:
        adapter_group_step_infos: List of AdapterGroupStepInfo objects
        num_pipeline_stages: Number of pipeline stages

    Returns:
        Updated list of AdapterGroupStepInfo objects after merging
    """
    # Skip processing if any empty adapter_group_step_infos
    if not adapter_group_step_infos:
        return adapter_group_step_infos

    # First, filter out any adapter groups with empty micro_batch_infos
    filtered_adapter_group_step_infos = []
    for adapter_group_step_info in adapter_group_step_infos:
        if not adapter_group_step_info.micro_batch_infos:
            logger.info("Removing adapter group with no micro batches")
            continue
        filtered_adapter_group_step_infos.append(adapter_group_step_info)

    # If all groups were filtered out, return empty list
    if not filtered_adapter_group_step_infos:
        return []

    # Use the filtered list for further processing
    adapter_group_step_infos = filtered_adapter_group_step_infos

    # Calculate remaining padded total tokens for logging
    previous_remaining_padded_total_tokens = [
        adapter_group_step_info.micro_batch_infos[-1].padded_total_tokens
        for adapter_group_step_info in adapter_group_step_infos
    ]

    logger.info(
        f"Previous remaining padded total tokens: "
        f"{previous_remaining_padded_total_tokens}"
    )

    # Try to merge micro-batches
    for i in range(len(adapter_group_step_infos) - 1):
        curr_adapter_group_step_info = adapter_group_step_infos[i]
        next_adapter_group_step_info = adapter_group_step_infos[i + 1]

        if (
            not curr_adapter_group_step_info.micro_batch_infos
            or not next_adapter_group_step_info.micro_batch_infos
        ):
            logger.info(
                f"Skipping merge for adapter group "
                f"{curr_adapter_group_step_info.adapter_group_info} "
                f"and {next_adapter_group_step_info.adapter_group_info} "
                f"due to empty micro-batch infos"
            )
            continue

        # See whether we can merge two remaining micro-batches
        curr_remaining_micro_batch = curr_adapter_group_step_info.micro_batch_infos[-1]
        next_remaining_micro_batch = next_adapter_group_step_info.micro_batch_infos[-1]

        # Check if there's at least some space available for partial merging
        if (
            curr_remaining_micro_batch.padded_total_tokens
            < curr_remaining_micro_batch.max_microbatch_size
        ):
            # Check the bubble lemma
            for j in range(i + 1, len(adapter_group_step_infos)):
                if (
                    adapter_group_step_infos[j].adapter_group_info
                    == curr_adapter_group_step_info.adapter_group_info
                ):
                    # Calculate in-between micro-batches
                    in_between_micro_batches = sum(
                        len(adapter_group_step_infos[k].micro_batch_infos)
                        for k in range(i + 1, j)
                    )

                    # Check if bubble lemma is satisfied
                    if in_between_micro_batches >= num_pipeline_stages - 1:
                        # Check if we can do full merge
                        if (
                            curr_remaining_micro_batch.padded_total_tokens
                            + next_remaining_micro_batch.padded_total_tokens
                            <= curr_remaining_micro_batch.max_microbatch_size
                        ):
                            logger.info(
                                f"Merging the remaining micro-batches completely for "
                                f"adapter group "
                                f"{curr_adapter_group_step_info.adapter_group_info} "
                                f"and {next_adapter_group_step_info.adapter_group_info}"
                                f"with {in_between_micro_batches} micro-batches in "
                                f"between."
                            )
                            # Merge the remaining micro-batches
                            curr_remaining_micro_batch.full_merge(
                                next_remaining_micro_batch
                            )
                            next_adapter_group_step_info.micro_batch_infos.pop(-1)

                            # Log if group became empty
                            if not next_adapter_group_step_info.micro_batch_infos:
                                logger.info(
                                    "Next adapter group now has zero micro-batches "
                                    "after merge"
                                )
                        else:
                            # Try partial merging
                            logger.info(
                                f"Attempting partial merge of remaining micro-batches "
                                f"for adapter group "
                                f"{curr_adapter_group_step_info.adapter_group_info} "
                                f"and {next_adapter_group_step_info.adapter_group_info}"
                            )

                            # Apply partial merging
                            _, remaining = curr_remaining_micro_batch.partial_merge(
                                next_remaining_micro_batch
                            )

                            # If partial merging was successful
                            if (
                                remaining is not None
                                and remaining != next_remaining_micro_batch
                            ):
                                # Replace the next micro batch with the reduced version
                                next_adapter_group_step_info.micro_batch_infos[-1] = (
                                    remaining
                                )
                                logger.info(
                                    f"Partial merge successful: reduced next "
                                    f"micro-batch from "
                                    f"{next_remaining_micro_batch.padded_total_tokens} "
                                    f"to "
                                    f"{remaining.padded_total_tokens} tokens."
                                )
                            elif remaining is None:
                                # Complete merge happened through partial_merge
                                next_adapter_group_step_info.micro_batch_infos.pop(-1)
                                logger.info(
                                    "All samples successfully transferred in partial "
                                    "merge."
                                )

                                # Log if group became empty
                                if not next_adapter_group_step_info.micro_batch_infos:
                                    logger.info(
                                        "Next adapter group now has zero micro-batches "
                                        "after partial merge"
                                    )

                        # Update the internal_adapter_start_end_indices
                        curr_adapter_group_step_info.self_update()
                        next_adapter_group_step_info.self_update()

                    break

    # Filter out groups that became empty during merging
    final_adapter_group_step_infos = [
        group for group in adapter_group_step_infos if group.micro_batch_infos
    ]

    if len(final_adapter_group_step_infos) < len(adapter_group_step_infos):
        logger.info(
            f"Removed {
                len(adapter_group_step_infos) - len(final_adapter_group_step_infos)
            } "
            f"empty adapter groups after merging"
        )

    # Calculate after-merge statistics
    after_merge_remaining_padded_total_tokens = [
        group.micro_batch_infos[-1].padded_total_tokens
        for group in final_adapter_group_step_infos
    ]

    logger.info(
        f"After merge remaining padded total tokens: "
        f"{after_merge_remaining_padded_total_tokens}"
    )

    return final_adapter_group_step_infos


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _can_merge_full(mb_a: MicroBatchInfo, mb_b: MicroBatchInfo) -> bool:
    """Capacity + adapter-overlap check before we try anything expensive."""
    if mb_a.padded_total_tokens + mb_b.padded_total_tokens > mb_a.max_microbatch_size:
        return False

    # same adapter appears with *different* global-batch - forbidden
    for a_idx in mb_a.adapter_group_info & mb_b.adapter_group_info:
        g_a = {g for (a, g, _) in mb_a.data_samples if a == a_idx}
        g_b = {g for (a, g, _) in mb_b.data_samples if a == a_idx}
        if len(g_a | g_b) > 1:  # two different global-batch ids
            return False
    return True


def _bubble_ok_after_move(flat: list[MicroBatchInfo], num_pipeline_stages: int) -> bool:
    """Check bubble lemma after a tentative merge / move."""
    last_pos: dict[tuple[int, int], int] = {}  # (adapter, global_batch) → mb_idx
    for pos, mb in enumerate(flat):
        seen_in_mb: dict[int, int] = (
            {}
        )  # adapter → that adapter's global_batch in this mb
        for a, g, _ in mb.data_samples:
            if a in seen_in_mb and seen_in_mb[a] != g:
                return False  # two different global batches of a in the *same* mb
            seen_in_mb[a] = g

        for a, g in seen_in_mb.items():
            prev = last_pos.get((a, g - 1))
            if prev is not None and pos - prev < num_pipeline_stages:
                return False
        for key in seen_in_mb.items():
            last_pos[key] = pos
    return True


def apply_merge_pass_for_adapter_group_step_infos_impl_v2(  # noqa: C901, PLR0912
    adapter_group_step_infos: list[AdapterGroupStepInfo],
    num_pipeline_stages: int,
    *,
    delta: int = 1024,  # how much slack (=unused tokens) triggers a search
    look_ahead: int = 8,  # how far we search for a partner micro-batch
) -> list[AdapterGroupStepInfo]:
    """Greedy left-to-right merge pass (single sweep) on the **flattened** list.

    1.  Flatten the schedule but remember the parent AdapterGroupStepInfo.
    2.  For every micro-batch that still has ≥ delta free tokens
        search the *next `look_ahead`* micro-batches for the **smallest**
        merge-candidate that passes simple capacity / overlap checks.
    3.  Try a *full merge* first.  If it fits and `_bubble_ok_after_move`
        still holds ⇒ commit.
    4.  Otherwise fall back to `partial_merge`.  Keep the move only when
        bubble lemma is preserved.
    5.  Re-assemble the per-group lists and call `.self_update()` so that all
        cached fields (padded_total_tokens, etc.) are consistent again.
    """
    if not adapter_group_step_infos:
        return adapter_group_step_infos

    # ------------------------------------------------------------------
    # 1.  flatten
    # ------------------------------------------------------------------
    flat: list[tuple[int, MicroBatchInfo]] = []  # (group_idx, mb)
    for g_idx, ag in enumerate(adapter_group_step_infos):
        flat.extend((g_idx, mb) for mb in ag.micro_batch_infos)

    i = 0
    while i < len(flat):
        g_idx_i, mb_i = flat[i]

        # only bother if there's *meaningful* free capacity
        if mb_i.padded_total_tokens <= mb_i.max_microbatch_size - delta:
            # ----------------------------------------------------------
            # 3.  search window for the "best" partner (smallest mb)
            # ----------------------------------------------------------
            best_j, best_tokens = None, None
            for offset in range(1, min(look_ahead + 1, len(flat) - i)):
                j = i + offset
                g_idx_j, mb_j = flat[j]

                # no adapter overlap conflict & fits capacity
                if not _can_merge_full(mb_i, mb_j):
                    continue

                if best_j is None or mb_j.padded_total_tokens < best_tokens:
                    best_j = j
                    best_tokens = mb_j.padded_total_tokens

            # ----------------------------------------------------------
            # 4.  merge attempt (full → partial → none)
            # ----------------------------------------------------------
            if best_j is not None:
                g_idx_j, mb_j = flat[best_j]

                # try *full* merge first
                snapshot_i = mb_i.data_samples.copy()
                snapshot_j = mb_j.data_samples.copy()

                mb_i.full_merge(mb_j)

                if _bubble_ok_after_move([x[1] for x in flat], num_pipeline_stages):
                    # success → delete mb_j
                    del adapter_group_step_infos[g_idx_j].micro_batch_infos[
                        adapter_group_step_infos[g_idx_j].micro_batch_infos.index(mb_j)
                    ]
                    del flat[best_j]
                    # don't advance i (mb_i grew, may still have room)
                    continue
                # rollback full-merge
                mb_i.update_from_data_samples(snapshot_i)
                mb_j.update_from_data_samples(snapshot_j)
                # ---------- partial merge ---------------------------------------------
                _, remaining_mb_j = mb_i.partial_merge(mb_j)

                if remaining_mb_j is not None:
                    flat[best_j] = (g_idx_j, remaining_mb_j)
                    adapter_group_step_infos[g_idx_j].micro_batch_infos[
                        adapter_group_step_infos[g_idx_j].micro_batch_infos.index(mb_j)
                    ] = remaining_mb_j
                    candidate_flat = [x[1] for x in flat]
                    if _bubble_ok_after_move(candidate_flat, num_pipeline_stages):
                        mb_j = remaining_mb_j  # commit
                        if not mb_j.data_samples:
                            # emptied - delete
                            del adapter_group_step_infos[g_idx_j].micro_batch_infos[
                                adapter_group_step_infos[
                                    g_idx_j
                                ].micro_batch_infos.index(mb_j)
                            ]
                            del flat[best_j]
                        else:
                            # keep updated mb_j in place
                            pass
                    else:
                        # rollback partial merge
                        mb_i.update_from_data_samples(snapshot_i)
                        flat[best_j] = (g_idx_j, mb_j)  # original
                        adapter_group_step_infos[g_idx_j].micro_batch_infos[
                            adapter_group_step_infos[g_idx_j].micro_batch_infos.index(
                                remaining_mb_j
                            )
                        ] = mb_j

        i += 1  # advance sweep pointer

    # ------------------------------------------------------------------
    # 5.  re-assemble groups + clean empty ones
    # ------------------------------------------------------------------
    new_groups: list[AdapterGroupStepInfo] = []
    for ag in adapter_group_step_infos:
        ag.micro_batch_infos = [mb for mb in ag.micro_batch_infos if mb.data_samples]
        if ag.micro_batch_infos:
            ag.self_update()
            new_groups.append(ag)

    return new_groups


def verify_correctness_of_schedule(  # noqa: C901, PLR0912, PLR0915
    data: list[list[list[int]]],
    schedule: list[AdapterGroupStepInfo],
    num_pipeline_stages: int,
    *,
    add_bubble: bool = False,
) -> bool:
    """Verify the correctness of the schedule.

    1. Check all data is used exactly once.
    2. Check the bubble lemma: from the finish of a adapter group to the start of the
       next same adapter group, there should be at least num_pipeline_stages - 1
       micro-batches in between.
    """
    error_msgs = []

    # 1. Check the size of tokens in each micro-batch
    for adapter_group_idx, adapter_group_step_info in enumerate(schedule):
        for micro_batch_idx, micro_batch_info in enumerate(
            adapter_group_step_info.micro_batch_infos
        ):
            if (
                micro_batch_info.padded_total_tokens
                > micro_batch_info.max_microbatch_size
            ):
                error_msg = (
                    f"[{adapter_group_idx=}, {micro_batch_idx=}] "
                    f"Micro-batch has "
                    f"padded total tokens {micro_batch_info.padded_total_tokens} "
                    f"which exceeds the max microbatch size "
                    f"{micro_batch_info.max_microbatch_size}."
                )
                logger.warning(error_msg)
                error_msgs.append(error_msg)

    # 2. Check all data is used exactly once
    num_adapters = len(data)

    # Create a set of all data indices (adapter_idx, global_batch_idx, sample_idx)
    all_data_indices = set()
    for adapter_idx in range(num_adapters):
        for global_batch_idx in range(len(data[adapter_idx])):
            for sample_idx in range(len(data[adapter_idx][global_batch_idx])):
                all_data_indices.add((adapter_idx, global_batch_idx, sample_idx))

    # Create a set of all scheduled data indices
    scheduled_data_indices = set()
    for adapter_group_step_info in schedule:
        for micro_batch_info in adapter_group_step_info.micro_batch_infos:
            scheduled_data_indices.update(micro_batch_info.data_samples.keys())

    # Check if the sets are equal
    missing_indices = all_data_indices - scheduled_data_indices
    duplicate_indices = scheduled_data_indices - all_data_indices

    if missing_indices:
        curr_error_msg = f"Missing data indices: {missing_indices}"
        logger.warning(curr_error_msg)
        error_msgs.append(curr_error_msg)

    if duplicate_indices:
        curr_error_msg = f"Duplicate data indices: {duplicate_indices}"
        logger.warning(curr_error_msg)
        error_msgs.append(curr_error_msg)

    # To check bubble lemma, we do the following:
    # 1. Flatten the schedule into a list of micro-batches
    # 2. For each micro-batch, we get the adapter_sets
    # 3. We check all global batch indices are the same for each adapter_set
    # 4. We check the following num_pipeline_stages - 1 micro-batches don't have
    #    the same adapter_set with increased global batch index

    # Build the flat schedule and maintain a mapping back to original positions
    flattened_schedule = []
    # Maps from flattened position to (adapter_group_idx, micro_batch_idx)
    flat_to_schedule_map = []

    for adapter_group_idx, adapter_group_step_info in enumerate(schedule):
        for micro_batch_idx, micro_batch_info in enumerate(
            adapter_group_step_info.micro_batch_infos
        ):
            flattened_schedule.append(list(micro_batch_info.data_samples.keys()))
            flat_to_schedule_map.append((adapter_group_idx, micro_batch_idx))

    # For each micro-batch, we get the adapter_sets
    adapter_sets = []
    for micro_batch in flattened_schedule:
        adapter_to_global_batch_idx = {}
        for adapter_idx, global_batch_idx, _ in micro_batch:
            if adapter_idx not in adapter_to_global_batch_idx:
                adapter_to_global_batch_idx[adapter_idx] = global_batch_idx
            elif adapter_to_global_batch_idx[adapter_idx] != global_batch_idx:
                curr_error_msg = (
                    f"Micro-batch {micro_batch} has different global batch indices "
                    f"for the same adapter {adapter_idx}."
                )
                logger.warning(curr_error_msg)
                error_msgs.append(curr_error_msg)
        adapter_sets.append(adapter_to_global_batch_idx)

    bubble_insertion_info = []  # Store (adapter_group_idx, after_micro_batch_idx) pairs

    # For each micro-batch, we check the bubble lemma
    for i, _ in enumerate(flattened_schedule):
        for j in range(i + 1, min(i + num_pipeline_stages, len(flattened_schedule))):
            # Get the adapter sets from both micro-batches
            curr_adapter_set = adapter_sets[i]
            next_adapter_set = adapter_sets[j]

            # Check for overlapping adapters with different global batch indices
            for adapter_idx in curr_adapter_set:
                if (
                    adapter_idx in next_adapter_set
                    and curr_adapter_set[adapter_idx] != next_adapter_set[adapter_idx]
                ):
                    curr_error_msg = (
                        f"Bubble lemma violation: Micro-batch at position {i} and {j} "
                        f"have the same adapter {adapter_idx} but different global "
                        f"batch indices ({curr_adapter_set[adapter_idx]} vs "
                        f"{next_adapter_set[adapter_idx]}). These micro-batches are "
                        f"closer than the required {num_pipeline_stages} stages."
                    )
                    logger.warning(curr_error_msg)
                    error_msgs.append(curr_error_msg)

                    if add_bubble:
                        # Need to insert bubble at position j in the flat schedule
                        # Determine where this corresponds in the schedule structure
                        adapter_group_idx, micro_batch_idx = flat_to_schedule_map[j]
                        bubble_insertion_info.append(
                            (adapter_group_idx, micro_batch_idx)
                        )

    # Process bubble insertions
    if add_bubble and bubble_insertion_info:
        # Sort by adapter_group_idx (descending) and then by micro_batch_idx
        # (descending) This ensures we insert from back to front to avoid position
        # shifts
        bubble_insertion_info.sort(reverse=True)

        # Remove duplicates while preserving order
        seen = set()
        unique_insertion_info = []
        for info in bubble_insertion_info:
            if info not in seen:
                seen.add(info)
                unique_insertion_info.append(info)

        for adapter_group_idx, micro_batch_idx in unique_insertion_info:
            logger.info(
                f"Inserting a no-op before group {adapter_group_idx}, "
                f"micro-batch {micro_batch_idx}"
            )
            # Insert a new empty AdapterGroupStepInfo before the problem group
            schedule.insert(
                adapter_group_idx,
                AdapterGroupStepInfo(
                    micro_batch_infos=[
                        MicroBatchInfo(
                            data_samples={},
                            max_microbatch_size=0,
                            adapter_padding_multiple=1,
                            is_empty_marker=True,
                        )
                    ],
                    adapter_group_info=set(),
                    internal_adapter_start_end_indices={},
                ),
            )

    if error_msgs:
        logger.warning(
            "Schedule verification failed! The schedule may be invalid."
            "If only bubble lemma is violated, you can either reduce the microbatch "
            "size or simply let the pipeline stall to satisfy the bubble lemma."
        )
        for error_msg in error_msgs:
            logger.warning(error_msg)
        return False

    logger.info(
        "Schedule verified successfully: "
        "All data used exactly once and bubble lemma satisfied."
    )
    return True


def group_adapters(
    data: list[list[list[int]]],
    capacity: int = 4096,
    *,
    k_sigma: float = 0.25,
    num_stages: int = 4,
) -> list[list[int]]:
    """Greedy "head-tail" merge that uses a *probabilistic* length bound.

    Args:
        data: Nested list:  data[adapter_idx][global_batch_idx][sample] = token-length.
        capacity: Maximum number of tokens that fit into one pipeline micro-batch.
        k_sigma: Safety multiplier for the standard deviation.  k=1.5 ⇒ ≈ 86.6 %
            quantile for a light-tailed distribution.  Increase k to reduce
            overflow events.
        num_stages: Pipeline depth S.  Needed for the bubble-free check
            Σ N̂ - max N̂  ≥  S - 1.

    Returns:
        A list whose elements are lists of adapter indices.
        Each inner list represents one *group* that may share micro-batches.
        Groups are of size 1 or 2; no adapter appears in two groups.
    """
    # ------------------------------------------------------------------
    # 1.  Per-adapter summary statistics and conservative MB estimates
    # ------------------------------------------------------------------
    stats = []
    for a_idx, batches in enumerate(data):
        # Flatten all batches for this adapter to calculate statistics
        all_samples = np.array(batches)
        mu = np.mean(all_samples)
        sigma = np.std(all_samples, ddof=1) if all_samples.size > 1 else 0.0

        # Use the first batch size for estimating tokens per global batch
        gbs = len(batches[0])
        t_mu = gbs * mu
        n_mu = math.ceil(t_mu / capacity)
        t_k_sigma_upper_estimate = gbs * (mu + k_sigma * sigma)
        n_k_sigma_upper_estimate = math.ceil(t_k_sigma_upper_estimate / capacity)
        t_k_sigma_lower_estimate = gbs * (mu - k_sigma * sigma)
        n_k_sigma_lower_estimate = math.ceil(t_k_sigma_lower_estimate / capacity)
        stats.append(
            {
                "idx": a_idx,
                "mu": mu,  # mean of the token length
                "sigma": sigma,  # standard deviation of the token length
                "t_mu": t_mu,  # mean of the global batch tokens
                "n_mu": n_mu,  # mean of the global batch micro-batches
                "t_k_sigma_upper_estimate": t_k_sigma_upper_estimate,
                "n_k_sigma_upper_estimate": n_k_sigma_upper_estimate,
                "t_k_sigma_lower_estimate": t_k_sigma_lower_estimate,
                "n_k_sigma_lower_estimate": n_k_sigma_lower_estimate,
            }
        )

    # ------------------------------------------------------------------
    # 2.  Bubble lemma must already hold with the conservative bound
    # ------------------------------------------------------------------
    def check_bubble_lemma(
        stats: list[dict], removed_indices: list[int] | None = None
    ) -> bool:
        if removed_indices is None:
            # Pick the idx with the largest n_mu
            n_mu_array = [s["n_mu"] for s in stats]
            max_n_mu_idx = np.argmax(n_mu_array)
            removed_indices = [stats[max_n_mu_idx]["idx"]]
        elif not isinstance(removed_indices, list):
            removed_indices = [removed_indices]

        other_n_k_sigma_lower_estimates = [
            s["n_k_sigma_lower_estimate"]
            for s in stats
            if s["idx"] not in removed_indices
        ]
        return sum(other_n_k_sigma_lower_estimates) >= (num_stages - 1)

    if not check_bubble_lemma(stats):
        # No grouping can rescue the schedule - return singletons.
        return [[s["idx"]] for s in stats]

    # ------------------------------------------------------------------
    # 3.  Sort by mean sample length (descending) and run head-tail sweep
    # ------------------------------------------------------------------
    stats.sort(key=lambda s: s["mu"], reverse=True)
    left, right = 0, len(stats) - 1
    groups: list[list[int]] = []

    while left < right:
        stats_left, stats_right = stats[left], stats[right]

        # Re-evaluate bubble lemma
        if check_bubble_lemma(stats, [stats_left["idx"], stats_right["idx"]]):
            # Merge is safe - record group and advance both pointers
            groups.append([stats_left["idx"], stats_right["idx"]])
            left += 1
            right -= 1
        # If the check failed, it means that one block is too large
        # we skip the larger block
        elif stats_left["n_mu"] < stats_right["n_mu"]:
            right -= 1
            groups.append([stats_right["idx"]])
        else:
            left += 1
            groups.append([stats_left["idx"]])

    # Handle the case when there's one adapter left
    if left == right:
        groups.append([stats[left]["idx"]])

    return groups


def packing_data_to_microbatches_micro_milp(  # noqa: C901
    data: list[list[list[int]]],
    groups: list[list[int]],
    num_pipeline_stages: int,
    num_global_batches_per_adapter: int | None = None,
    capacity: int = 4096,
    adapter_padding_multiple: int = ADAPTER_PADDING_MULTIPLE,
    *,
    time_limit: int = 600,
    verbose: bool = False,
    disable_tqdm: bool = True,
) -> list[list[tuple[int, int, int]]]:
    """Pack the data into micro-batches using a micro-MILP.

    Args:
        data: Nested list:  data[adapter_idx][global_batch_idx][sample] = token-length.
        groups: List of groups of adapter indices.
        num_pipeline_stages: Number of pipeline stages.
        num_global_batches_per_adapter: Number of global batches per adapter.
        capacity: Maximum number of tokens that fit into one pipeline micro-batch.
        adapter_padding_multiple: Padding multiple for the adapter indices.
        time_limit: Time limit for the solver.
        verbose: Whether to print verbose output.
        disable_tqdm: Whether to disable the tqdm progress bar.

    Returns:
        A list of length L, each micro-batch is list[(i,j,k)].
    """
    num_adapters = len(data)

    # Check the groups contain all the adapters and no duplicates
    flattened_groups = [item for sublist in groups for item in sublist]
    if len(flattened_groups) != len(set(flattened_groups)):
        msg = "Groups contain duplicates!"
        raise ValueError(msg)
    if set(flattened_groups) != set(range(num_adapters)):
        msg = "Groups do not contain all the adapters!"
        raise ValueError(msg)

    # Check the number of global batches per adapter
    min_global_batches_per_adapter = min(
        len(adapter_to_batch_data) for adapter_to_batch_data in data
    )
    if num_global_batches_per_adapter is None:
        # Use the smallest number of global batches per adapter
        num_global_batches_per_adapter = min_global_batches_per_adapter
    elif num_global_batches_per_adapter > min_global_batches_per_adapter:
        msg = (
            f"num_global_batches_per_adapter ({num_global_batches_per_adapter}) "
            f"is greater than the minimum number of global batches per adapter "
            f"({min_global_batches_per_adapter})!"
        )
        raise ValueError(msg)

    # The input data is structured as (adapter_idx, global_batch_idx, sample)
    # we should get (global_batch_idx, adapter_idx, sample)
    batch_first_data = [
        [data[adapter_idx][batch_idx] for adapter_idx in range(num_adapters)]
        for batch_idx in range(num_global_batches_per_adapter)
    ]

    # Group the data
    # schedule is defined as (micro_batch_idx, list of sample index tuples)
    adapter_group_step_infos = []
    for batch_idx in tqdm(range(num_global_batches_per_adapter), disable=disable_tqdm):
        adapter_to_samples = batch_first_data[batch_idx]
        for group in groups:
            grouped_data = [adapter_to_samples[adapter_idx] for adapter_idx in group]
            curr_schedule, curr_schedule_size, curr_padded_schedule_size = (
                solve_micro_milp_lexicographic(
                    data=grouped_data,
                    target_microbatch_size=capacity,
                    adapter_padding_multiple=adapter_padding_multiple,
                    time_limit=time_limit,
                    verbose=verbose,
                )
            )
            micro_batch_infos = [
                MicroBatchInfo.from_raw_data_indices(
                    micro_batch,
                    local_batch_data=grouped_data,
                    max_microbatch_size=capacity,
                    adapter_mapping=group,
                    global_batch_idx=batch_idx,
                )
                for micro_batch in curr_schedule
            ]
            adapter_group_step_info = AdapterGroupStepInfo.from_micro_batch_infos_list(
                micro_batch_infos
            )
            adapter_group_step_infos.append(adapter_group_step_info)

    # Apply the merge pass
    merged_adapter_group_step_infos = (
        apply_merge_pass_for_adapter_group_step_infos_impl_v2(
            adapter_group_step_infos, num_pipeline_stages=num_pipeline_stages
        )
    )

    # Self-update all micro-batch infos
    for adapter_group_step_info in merged_adapter_group_step_infos:
        for micro_batch_info in adapter_group_step_info.micro_batch_infos:
            micro_batch_info.self_update()

    # Verify the correctness of the schedule
    is_valid = verify_correctness_of_schedule(
        data,
        merged_adapter_group_step_infos,
        num_pipeline_stages,
        add_bubble=True,
    )

    if not is_valid:
        logger.warning(
            "Schedule verification failed! The schedule may be invalid. "
            "If the error is due to the bubble lemma, you can just try "
            "a smaller microbatch token size, or simply let the pipeline "
            "run no-op for the mentioned groups for certain iterations."
        )

    # Verify again
    is_valid = verify_correctness_of_schedule(
        data,
        merged_adapter_group_step_infos,
        num_pipeline_stages,
        add_bubble=True,
    )

    if not is_valid:
        logger.warning(
            "Schedule verification failed! The schedule may be invalid. "
            "If the error is due to the bubble lemma, you can just try "
            "a smaller microbatch token size, or simply let the pipeline "
            "run no-op for the mentioned groups for certain iterations."
        )

    return merged_adapter_group_step_infos


def _pack_one_group(
    task: tuple[list[list[int]], int, int],
    *,
    capacity: int,
    adapter_padding_multiple: int,
    time_limit: int,
    verbose: bool,
) -> tuple[int, int, list[list[tuple[int, int]]]]:
    """Run MILP (or greedy) on ONE (batch_idx, group)."""
    batch_idx, group_idx, grouped_data = task
    schedule, _, _ = solve_micro_milp_lexicographic(
        data=grouped_data,
        target_microbatch_size=capacity,
        adapter_padding_multiple=adapter_padding_multiple,
        time_limit=time_limit,
        verbose=verbose,
    )
    return (batch_idx, group_idx, grouped_data), schedule


def packing_data_to_microbatches_micro_milp_with_multiprocessing(  # noqa: C901
    data: list[list[list[int]]],
    groups: list[list[int]],
    num_global_batches_per_adapter: int,
    num_pipeline_stages: int,
    capacity: int = 4096,
    adapter_padding_multiple: int = ADAPTER_PADDING_MULTIPLE,
    *,
    time_limit: int = 600,
    verbose: bool = False,
    disable_tqdm: bool = True,
    num_workers: int | None = None,
) -> list[list[tuple[int, int, int]]]:
    """Pack the data into micro-batches using a micro-MILP with multi-processing.

    Args:
        data: Nested list:  data[adapter_idx][global_batch_idx][sample] = token-length.
        groups: List of groups of adapter indices.
        num_global_batches_per_adapter: Number of global batches per adapter.
        num_pipeline_stages: Number of pipeline stages.
        capacity: Maximum number of tokens that fit into one pipeline micro-batch.
        adapter_padding_multiple: Padding multiple for the adapter indices.
        time_limit: Time limit for the solver.
        verbose: Whether to print verbose output.
        disable_tqdm: Whether to disable the tqdm progress bar.
        num_workers: Number of workers for the multi-processing pool.

    Returns:
        A list of length L, each micro-batch is list[(i,j,k)].
    """
    num_adapters = len(data)

    # Check the groups contain all the adapters and no duplicates
    flattened_groups = [item for sublist in groups for item in sublist]
    if len(flattened_groups) != len(set(flattened_groups)):
        msg = "Groups contain duplicates!"
        raise ValueError(msg)
    if set(flattened_groups) != set(range(num_adapters)):
        msg = "Groups do not contain all the adapters!"
        raise ValueError(msg)

    # The input data is structured as (adapter_idx, global_batch_idx, sample)
    # we should get (global_batch_idx, adapter_idx, sample)
    batch_first_data = [
        [data[adapter_idx][batch_idx] for adapter_idx in range(num_adapters)]
        for batch_idx in range(num_global_batches_per_adapter)
    ]

    # --------------- assemble tasks -----------------------------------
    # each task = (grouped_data) for ONE (batch, group) pair
    tasks: list[tuple[list[list[int]], int, int]] = []
    for batch_idx, (group_idx, group) in product(
        range(num_global_batches_per_adapter), enumerate(groups)
    ):
        grouped_data = [batch_first_data[batch_idx][a_idx] for a_idx in group]
        tasks.append((batch_idx, group_idx, grouped_data))

    # --------------- multiprocessing pool -----------------------------
    worker = partial(
        _pack_one_group,
        capacity=capacity,
        adapter_padding_multiple=adapter_padding_multiple,
        time_limit=time_limit,
        verbose=verbose,
    )

    group_size = len(groups)
    schedules: list[AdapterGroupStepInfo | None] = [
        None for _ in range(group_size * num_global_batches_per_adapter)
    ]
    with multiprocessing.Pool(processes=num_workers) as pool:
        for (batch_idx, group_idx, grouped_data), raw_sched in tqdm(
            pool.imap_unordered(worker, tasks),
            total=len(tasks),
            disable=disable_tqdm,
        ):
            # Convert the raw schedule to MicroBatchInfo objects
            micro_batch_infos = [
                MicroBatchInfo.from_raw_data_indices(
                    micro_batch,
                    local_batch_data=grouped_data,
                    max_microbatch_size=capacity,
                    adapter_mapping=groups[group_idx],
                    global_batch_idx=batch_idx,
                )
                for micro_batch in raw_sched
            ]
            adapter_group_step_info = AdapterGroupStepInfo.from_micro_batch_infos_list(
                micro_batch_infos
            )
            schedules[batch_idx * group_size + group_idx] = adapter_group_step_info

    # Make sure all the schedules are filled
    for i, schedule in enumerate(schedules):
        if schedule is None:
            msg = (
                f"Schedule for batch {i // group_size} and group {i % group_size} "
                f"is not filled!"
            )
            raise ValueError(msg)

    # Apply the merge pass
    schedules = apply_merge_pass_for_adapter_group_step_infos(
        schedules, num_pipeline_stages=num_pipeline_stages
    )

    # Self-update all micro-batch infos
    print_list = []
    for adapter_group_step_info in schedules:
        for micro_batch_info in adapter_group_step_info.micro_batch_infos:
            micro_batch_info.self_update()
            print_list.append(
                f"{micro_batch_info.padded_total_tokens} "
                f"{micro_batch_info.adapter_group_info}"
            )

    logger.info(f"Print list: {', '.join(print_list)}")

    logger.info(f"Length of schedules: {len(schedules)} before verification")

    # Verify the correctness of the schedule
    is_valid = verify_correctness_of_schedule(
        data, schedules, num_pipeline_stages, add_bubble=True
    )

    logger.info(f"Length of schedules: {len(schedules)} adding bubbles")

    if not is_valid:
        logger.warning("Schedule verification failed! The schedule may be invalid.")

    # Verify again
    is_valid = verify_correctness_of_schedule(
        data, schedules, num_pipeline_stages, add_bubble=True
    )

    if not is_valid:
        logger.warning("Schedule verification failed! The schedule may be invalid.")

    return schedules


def _round_up(v: int, multiple: int) -> int:
    return ((v + multiple - 1) // multiple) * multiple


def greedy_pack_decide_bins(
    data: list[list[int]],
    target_microbatch_size: int,
    samples: list[tuple[int, int, int]] | None = None,
    adapter_padding_multiple: int = ADAPTER_PADDING_MULTIPLE,
) -> tuple[
    int, list[list[tuple[int, int]]], list[list[int]], list[list[tuple[int, int]]], int
]:
    """Greedy pack the data into micro-batches and decide the number of bins."""
    # Return: number of bins, schedule, true_len, padded, min_remaining_size
    if samples is None:
        samples = [
            (i, j, d) for i, adapter in enumerate(data) for j, d in enumerate(adapter)
        ]

    if not samples:
        return [], [], [], [], target_microbatch_size + 1

    # sort indices by descending token length
    idx = sorted(range(len(samples)), key=lambda i: samples[i][2], reverse=True)

    bins: list[list[int]] = []  # store sample indices
    bin_pad_sums: list[int] = []  # store padded total per bin
    bin_adapter_tokens: list[defaultdict[int, int]] = []

    # -------- first-fit-decreasing loop -------------------------------
    for s in idx:
        a, _, t = samples[s]
        placed = False
        for b, _ in enumerate(bin_pad_sums):
            # try to fit sample into existing bin b
            new_tok = bin_adapter_tokens[b].copy()
            new_tok[a] += t
            new_pad = sum(
                _round_up(v, adapter_padding_multiple) for v in new_tok.values()
            )
            if new_pad <= target_microbatch_size:
                bins[b].append(s)
                bin_pad_sums[b] = new_pad
                bin_adapter_tokens[b] = new_tok
                placed = True
                break
        if not placed:
            # open new bin
            new_tok = defaultdict(int)
            new_tok[a] = t
            bins.append([s])
            bin_adapter_tokens.append(new_tok)
            bin_pad_sums.append(_round_up(t, adapter_padding_multiple))

    # -------- generate return objects ---------------------------------
    schedule, true_len, padded = [], [], []
    for b, s_idx_list in enumerate(bins):
        tok_sum_dict = bin_adapter_tokens[b]
        schedule.append([(samples[i][0], samples[i][1]) for i in s_idx_list])
        true_len.append([samples[i][2] for i in s_idx_list])
        padded.append(
            [
                (a_id, _round_up(toks, adapter_padding_multiple))
                for a_id, toks in tok_sum_dict.items()
            ]
        )

    # sort bins by padded size (largest → smallest) to keep convention
    order = sorted(range(len(bins)), key=lambda i: bin_pad_sums[i], reverse=True)
    remaining_size = min(bin_pad_sums[i] for i in order)
    return (
        len(bins),
        [schedule[i] for i in order],
        [true_len[i] for i in order],
        [padded[i] for i in order],
        remaining_size,
    )


def solve_micro_milp_lexicographic(  # noqa: PLR0915
    data: list[list[int]],
    target_microbatch_size: int,
    adapter_padding_multiple: int = ADAPTER_PADDING_MULTIPLE,
    *,
    verbose: bool = False,
    time_limit: int = 600,
) -> list[list[tuple[int, int, int]]]:
    """Pack the data into micro-batches using a micro-MILP.

    Args:
        data: Nested list:  data[adapter_idx][global_batch_idx][sample] = token-length.
        target_microbatch_size: Maximum number of tokens that fit into one pipeline
            micro-batch.
        adapter_padding_multiple: Padding multiple for the adapter indices.
        verbose: Whether to print verbose output.
        time_limit: Time limit for the solver.

    Returns:
        A list of length L, each micro-batch is list[(i,j,k)].
    """
    logger.info("-" * 100)
    samples = [
        (i, j, d) for i, adapter in enumerate(data) for j, d in enumerate(adapter)
    ]

    start_time_greedy = time.perf_counter()
    (
        num_bins_greedy,
        schedule_greedy,
        schedule_size_greedy,
        padded_schedule_size_greedy,
        remaining_size_greedy,
    ) = greedy_pack_decide_bins(
        data=data,
        target_microbatch_size=target_microbatch_size,
        samples=samples,
        adapter_padding_multiple=adapter_padding_multiple,
    )
    end_time_greedy = time.perf_counter()
    logger.info(
        f"greedy_pack_decide_bins time: {end_time_greedy - start_time_greedy:.2f}s. "
        f"num_bins_greedy: {num_bins_greedy}, "
        f"remaining_size_greedy: {remaining_size_greedy}."
    )

    start_time = time.perf_counter()
    num_bins_milp = solve_micro_milp_min_bins(
        data=data,
        target_microbatch_size=target_microbatch_size,
        samples=samples,
        adapter_padding_multiple=adapter_padding_multiple,
        time_limit=time_limit,
        verbose=verbose,
    )
    end_time = time.perf_counter()
    logger.info(
        f"solve_micro_milp_min_bins time: {end_time - start_time:.2f}s. "
        f"num_bins_milp: {num_bins_milp}."
    )

    if num_bins_greedy <= num_bins_milp:
        min_bins = num_bins_greedy
        logger.warning(
            f"greedy_pack_decide_bins found {num_bins_greedy} bins, "
            f"but solve_micro_milp_min_bins found {num_bins_milp} bins. "
            f"Using greedy_pack_decide_bins result."
        )
    else:
        logger.success(
            f"solve_micro_milp_min_bins found {num_bins_milp} bins, "
            f"but greedy_pack_decide_bins found {num_bins_greedy} bins. "
            f"Using solve_micro_milp_min_bins result."
        )
        min_bins = num_bins_milp

    start_time = time.perf_counter()
    schedule, schedule_size, padded_schedule_size, remaining_size_milp = (
        solve_micro_milp_min_remaining_impl(
            data=data,
            target_microbatch_size=target_microbatch_size,
            num_bins=min_bins,
            samples=samples,
            adapter_padding_multiple=adapter_padding_multiple,
            time_limit=time_limit,
            verbose=verbose,
        )
    )
    end_time = time.perf_counter()
    logger.info(
        f"solve_micro_milp_min_remaining time: {end_time - start_time:.2f}s. "
        f"remaining_size_milp: {remaining_size_milp}."
    )

    if remaining_size_milp > target_microbatch_size or (
        num_bins_greedy == min_bins and remaining_size_greedy < remaining_size_milp
    ):
        logger.warning(
            f"greedy algorithm found remaining size {remaining_size_greedy}, "
            f"but solve_micro_milp_min_remaining found {remaining_size_milp}. "
            f"Using greedy algorithm result."
        )
        schedule = schedule_greedy
        schedule_size = schedule_size_greedy
        padded_schedule_size = padded_schedule_size_greedy
    else:
        logger.success("MILP works better than greedy algorithm. ")

    # ==== Check all indices are used ====
    # Check all indices are used
    all_samples = {(i, j) for i, j, _ in samples}
    used_samples = {(i, j) for microbatch in schedule for i, j in microbatch}
    unused_samples = all_samples - used_samples
    if unused_samples:
        logger.error(f"Unused samples: {unused_samples}. ")
        logger.error(f"All samples: {all_samples}. ")
        logger.error(f"Used samples: {used_samples}. ")
        logger.error(f"Raw samples: {samples}. ")
        logger.error(f"Schedule: {schedule}. ")
        logger.error(f"Num bins greedy: {num_bins_greedy}. ")
        logger.error(f"Num bins milp: {num_bins_milp}. ")
        logger.error(f"Remaining size greedy: {remaining_size_greedy}. ")
        logger.error(f"Remaining size milp: {remaining_size_milp}. ")
        logger.error(f"Greedy schedule: {schedule_greedy}. ")
        logger.error(f"Greedy padded schedule: {padded_schedule_size_greedy}. ")
        logger.error(f"Milp schedule: {schedule}. ")
        logger.error(f"Milp padded schedule: {padded_schedule_size}. ")
        msg = "Unused samples found"
        raise ValueError(msg)

    padded_total_tokens = [
        sum(s[1] for s in microbatch) for microbatch in padded_schedule_size
    ]
    logger.info(f"schedule: {schedule}")
    logger.info(f"schedule_size: {schedule_size}")
    logger.info(f"padded_schedule_size: {padded_schedule_size}")
    logger.info(f"padded_total_tokens: {padded_total_tokens}")
    return schedule, schedule_size, padded_schedule_size


def solve_micro_milp_min_bins(  # noqa: C901, PLR0912
    data: list[list[int]],
    target_microbatch_size: int,
    samples: list[tuple[int, int, int]] | None = None,
    adapter_padding_multiple: int = ADAPTER_PADDING_MULTIPLE,
    time_limit: int = 600,
    *,
    verbose: bool = False,
) -> int:
    """Pack the data into micro-batches using a micro-MILP.

    Args:
        data: Nested list:  data[adapter_idx][global_batch_idx][sample] = token-length.
        target_microbatch_size:
            Maximum number of tokens that fit into one pipeline micro-batch.
        samples: List of samples to pack.
        adapter_padding_multiple: Padding multiple for the adapter indices.
        time_limit: Time limit for the solver.
        verbose: Whether to print verbose output.

    Returns:
        Number of micro-batches.
    """
    # ------------------------------------------------------------------
    # 1.  Flatten samples
    # ------------------------------------------------------------------
    # samples: list (adapter, sample, len).
    if samples is None:
        samples = [
            (i, j, d) for i, adapter in enumerate(data) for j, d in enumerate(adapter)
        ]

    if not samples:
        return 0

    n_samples = len(samples)

    # Maximum number of bins needed is the number of samples in the worst case
    # where each sample goes into its own bin
    max_bins = n_samples

    # index helpers
    adapters = {a for a, _, _ in samples}

    # ------------------------------------------------------------------
    # 2.  MILP variables
    # ------------------------------------------------------------------
    # x[s, b] ∈ {0,1}  — sample s assigned to bin b
    prob = pulp.LpProblem("MicroBatchPacking", pulp.LpMinimize)
    x = {
        (s, b): pulp.LpVariable(f"x_{s}_{b}", cat="Binary")
        for s in range(n_samples)
        for b in range(max_bins)
    }

    # k[a, b] ∈ N  - padded multiples contributed by adapter a in bin b
    k = {
        (a, b): pulp.LpVariable(f"k_{a}_{b}", lowBound=0, cat="Integer")
        for a in adapters
        for b in range(max_bins)
    }

    # bin_used[b] ∈ {0,1}  — bin b has samples
    bin_used = {
        b: pulp.LpVariable(f"bin_used_{b}", cat="Binary") for b in range(max_bins)
    }

    # ------------------------------------------------------------------
    # 3.  Constraints
    # ------------------------------------------------------------------

    # 3.1 each sample exactly in one bin
    for s in range(n_samples):
        prob += pulp.lpSum(x[s, b] for b in range(max_bins)) == 1

    # 3.2 adapter-local token subtotal ≤ k[a,b] \times adapter_padding_multiple
    for a in adapters:
        for b in range(max_bins):
            subtotal = pulp.lpSum(
                samples[s][2] * x[s, b]  # true length
                for s in range(n_samples)
                if samples[s][0] == a
            )
            prob += subtotal <= k[a, b] * adapter_padding_multiple

    # 3.3 capacity of each bin and linking with bin_used
    for b in range(max_bins):
        # Constraint 1: if the bin is used, then bin_used[b] == 1
        prob += (
            pulp.lpSum(k[a, b] * adapter_padding_multiple for a in adapters)
            <= target_microbatch_size * bin_used[b]
        )
        # Constraint 2: if the bin is not used, then the bin_used[b] must be 0
        prob += (
            pulp.lpSum(k[a, b] * adapter_padding_multiple for a in adapters)
            >= bin_used[b]
        )

    # 3.4 if bin_used[b] == 0, then bin_used[b+1:] must be 0
    for b in range(max_bins - 1):
        prob += bin_used[b + 1] <= bin_used[b]

    # ------------------------------------------------------------------
    # 4.  Objective  — minimise number of bins
    # ------------------------------------------------------------------
    prob += pulp.lpSum(bin_used[b] for b in range(max_bins))

    # Solver setup with retries
    max_retries = 3
    current_time_limit = time_limit

    for attempt in range(max_retries):
        solver = pulp.PULP_CBC_CMD(
            mip=True,
            msg=verbose,
            timeLimit=current_time_limit,
            threads=multiprocessing.cpu_count(),
            options=["cuts on", "presolve on", "heuristics on"],
        )

        try:
            prob.solve(solver)

            if prob.status == pulp.LpStatusOptimal:
                # We found an optimal solution, no need to retry
                break

            if attempt < max_retries - 1:
                logger.warning(
                    f"Non-optimal solution (status: {prob.status}) on attempt "
                    f"{attempt + 1}. Retrying with doubled time limit "
                    f"({current_time_limit} → {current_time_limit * 2})."
                )
                current_time_limit *= 2
            else:
                logger.warning(
                    f"Non-optimal solution (status: {prob.status}) after "
                    f"{max_retries} attempts. Proceeding with best solution found."
                )
        except pulp.PulpSolverError as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"Solver error on attempt {attempt + 1}: {e}. "
                    f"Retrying with doubled time limit "
                    f"({current_time_limit} → {current_time_limit * 2})."
                )
                current_time_limit *= 2
            else:
                logger.error(f"Solver error after {max_retries} attempts: {e}")
                msg = f"Solver error after {max_retries} attempts: {e}"
                return max_bins

    # ------------------------------------------------------------------
    # 5.  Extract schedule
    # ------------------------------------------------------------------
    num_bins = pulp.value(pulp.lpSum(bin_used[b] for b in range(max_bins)))

    if not num_bins.is_integer():
        msg = f"Number of bins {num_bins} is not an integer!"
        logger.warning(msg)
        num_bins = math.ceil(num_bins)

    return int(num_bins)


def solve_micro_milp_min_remaining_impl(  # noqa: C901, PLR0912, PLR0915
    data: list[list[int]],
    target_microbatch_size: int,
    num_bins: int,
    samples: list[tuple[int, int, int]] | None = None,
    adapter_padding_multiple: int = ADAPTER_PADDING_MULTIPLE,
    time_limit: int = 600,
    *,
    verbose: bool = False,
) -> tuple[
    list[list[tuple[int, int]]], list[list[int]], list[list[tuple[int, int]]], int
]:
    """Pack the data into micro-batches using a micro-MILP.

    Args:
        data: Nested list:  data[adapter_idx][global_batch_idx][sample] = token-length.
        target_microbatch_size:
            Maximum number of tokens that fit into one pipeline micro-batch.
        num_bins: Number of micro-batches.
        samples: List of samples to pack.
        adapter_padding_multiple: Padding multiple for the adapter indices.
        time_limit: Time limit for the solver.
        verbose: Whether to print verbose output.

    Returns:
        Schedule of micro-batches.
    """
    # ------------------------------------------------------------------
    # 1.  Flatten samples
    # ------------------------------------------------------------------
    # samples: list (adapter, sample, len).
    if samples is None:
        samples = [
            (i, j, d) for i, adapter in enumerate(data) for j, d in enumerate(adapter)
        ]

    if not samples:
        return [], [], [], target_microbatch_size + 1

    n_samples = len(samples)
    adapters = {a for a, _, _ in samples}

    # ------------------------------------------------------------------
    # 2.  MILP variables
    # ------------------------------------------------------------------
    # x[s, b] ∈ {0,1}  — sample s assigned to bin b
    x = {
        (s, b): pulp.LpVariable(f"x_{s}_{b}", cat="Binary")
        for s in range(n_samples)
        for b in range(num_bins)
    }

    # k[a, b] ∈ N  - padded multiples contributed by adapter a in bin b
    k = {
        (a, b): pulp.LpVariable(f"k_{a}_{b}", lowBound=0, cat="Integer")
        for a in adapters
        for b in range(num_bins)
    }

    # bin_size[b] ∈ N  - size of bin b
    is_t_min = {
        b: pulp.LpVariable(f"is_t_min_{b}", cat="Binary") for b in range(num_bins)
    }
    t_min = pulp.LpVariable("t_min", lowBound=0, cat="Integer")

    # ------------------------------------------------------------------
    # 3.  Constraints
    # ------------------------------------------------------------------
    prob = pulp.LpProblem("MicroBatchPacking", pulp.LpMinimize)

    # 3.1 each sample exactly in one bin
    for s in range(n_samples):
        prob += pulp.lpSum(x[s, b] for b in range(num_bins)) == 1

    # 3.2 adapter-local token subtotal ≤ k[a,b] \times M
    for a in adapters:
        for b in range(num_bins):
            subtotal = pulp.lpSum(
                samples[s][2] * x[s, b]  # true length
                for s in range(n_samples)
                if samples[s][0] == a
            )
            prob += subtotal <= k[a, b] * adapter_padding_multiple

    # 3.3 capacity of each bin and linking with bin_size
    for b in range(num_bins):
        # Constraint 2: bin_size[b] must be less than or equal to the target microbatch
        # size
        bin_size = pulp.lpSum(k[a, b] * adapter_padding_multiple for a in adapters)
        prob += bin_size <= target_microbatch_size

        # Constraint 3: t_min is the minimum of the bin sizes
        prob += t_min >= bin_size - M * (1 - is_t_min[b])
        prob += t_min <= bin_size + M * (1 - is_t_min[b])

    # 3.4 only one t_min
    prob += pulp.lpSum(is_t_min[b] for b in range(num_bins)) == 1

    # ------------------------------------------------------------------
    # 4.  Objective  — minimise number of bins
    # ------------------------------------------------------------------
    prob += t_min

    # Solver setup with retries
    max_retries = 3
    current_time_limit = time_limit

    for attempt in range(max_retries):
        solver = pulp.PULP_CBC_CMD(
            mip=True,
            msg=verbose,
            timeLimit=current_time_limit,
            threads=multiprocessing.cpu_count(),
            options=["ratio 0.15", "heur on", "cuts on", "presolve on"],
        )

        try:
            prob.solve(solver)

            if prob.status == pulp.LpStatusOptimal:
                # We found an optimal solution, no need to retry
                break

            if attempt < max_retries - 1:
                logger.warning(
                    f"Non-optimal solution (status: {prob.status}) on attempt "
                    f"{attempt + 1}. "
                    f"Retrying with doubled time limit "
                    f"({current_time_limit} → {current_time_limit * 2})."
                )
                current_time_limit *= 2
            else:
                logger.warning(
                    f"Non-optimal solution (status: {prob.status}) after "
                    f"{max_retries} attempts. "
                    f"Proceeding with best solution found."
                )
        except pulp.LpSolverError as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"Solver error on attempt {attempt + 1}: {e}. "
                    f"Retrying with doubled time limit "
                    f"({current_time_limit} → {current_time_limit * 2})."
                )
                current_time_limit *= 2
            else:
                logger.error(f"Solver error after {max_retries} attempts: {e}")
                return [], [], [], target_microbatch_size + 1

    # ------------------------------------------------------------------
    # 5.  Extract schedule
    # ------------------------------------------------------------------
    x_values = {
        (s, b): pulp.value(x[s, b]) for s in range(n_samples) for b in range(num_bins)
    }
    schedule = [[] for _ in range(num_bins)]
    schedule_size = [[] for _ in range(num_bins)]
    padded_schedule_size = [[] for _ in range(num_bins)]
    for s in range(n_samples):
        scheduled = False
        for bin_idx in range(num_bins):
            if x_values[(s, bin_idx)] > ZERO_POINT_FIVE:
                adapter_idx, sample_idx, sample_len = samples[s]
                schedule[bin_idx].append((adapter_idx, sample_idx))
                schedule_size[bin_idx].append(sample_len)
                scheduled = True
                break

        if not scheduled:
            logger.error(f"Sample {samples[s]} is not scheduled!")
            return [], [], [], target_microbatch_size + 1

    # Calculate the padded schedule size
    for bin_idx in range(num_bins):
        adapter_to_padded_size = {}
        for (adapter_idx, _), sample_len in zip(
            schedule[bin_idx], schedule_size[bin_idx], strict=True
        ):
            if adapter_idx not in adapter_to_padded_size:
                adapter_to_padded_size[adapter_idx] = 0
            adapter_to_padded_size[adapter_idx] += sample_len
        adapter_to_padded_size = {
            k: _round_up(v, adapter_padding_multiple)
            for k, v in adapter_to_padded_size.items()
        }
        padded_schedule_size[bin_idx] = list(adapter_to_padded_size.items())

    # Smallest remaining size
    total_padded_schedule_size = [
        sum([p[1] for p in microbatch]) for microbatch in padded_schedule_size
    ]
    for i, size in enumerate(total_padded_schedule_size):
        if size > target_microbatch_size:
            msg = (
                f"Total padded schedule size {size} for bin {i} "
                f"exceeds target microbatch size {target_microbatch_size}!"
            )
            logger.error(msg)
            return [], [], [], target_microbatch_size + 1

    smallest_remaining_size = (
        min(total_padded_schedule_size)
        if total_padded_schedule_size
        else target_microbatch_size + 1
    )

    logger.info(
        f"padded_schedule_size: {padded_schedule_size}, "
        f"smallest_remaining_size: {smallest_remaining_size}"
    )

    return (
        schedule,
        schedule_size,
        padded_schedule_size,
        smallest_remaining_size,
    )


if __name__ == "__main__":
    debug_1 = False
    if debug_1:
        data = [[3000, 3000, 2000], [900, 1000, 1000, 1000, 1000]]
        target_microbatch_size = 4096
        verbose = True

        num_bins = solve_micro_milp_min_bins(
            data, target_microbatch_size, verbose=verbose
        )
        logger.info(f"num_bins: {num_bins}")
        schedule, schedule_size, padded_schedule_size = (
            solve_micro_milp_min_remaining_impl(
                data, target_microbatch_size, num_bins, verbose=verbose
            )
        )
        logger.info(f"schedule: {schedule}")
        logger.info(f"schedule_size: {schedule_size}")
        logger.info(f"padded_schedule_size: {padded_schedule_size}")

    debug_2 = False
    if debug_2:
        data = [
            [[664, 299, 908, 716, 431, 1323, 258, 165]],
            [[709, 1160, 796, 1386, 661, 917, 484, 651]],
            [[2637, 1260, 2043, 1847, 1219, 1850, 1439, 1060]],
            [[917, 287, 593, 1293, 996, 1160, 1875, 836]],
        ]
        target_microbatch_size = 4096
        verbose = False
        groups = group_adapters(data)
        schedule = packing_data_to_microbatches_micro_milp(
            data,
            groups,
            num_global_batches_per_adapter=1,
            capacity=4096,
            adapter_padding_multiple=get_lora_kernel_config("fused_multi_lora_block_size_m"),
            verbose=verbose,
            time_limit=0.5,
        )
        logger.info(f"schedule: {schedule}")
