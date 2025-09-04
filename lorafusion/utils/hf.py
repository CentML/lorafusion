"""Hugging Face utilities."""

import torch


def create_packed_dummy_inputs(
    hidden_size: int,
    seq_len_list: list[int],
    multiple_of: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str | None = None,
    *,
    return_input_ids: bool = True,
    return_inputs_embeds: bool = False,
    return_labels: bool = False,
) -> dict[str, torch.Tensor | int]:
    """Create dummy inputs for packed sequence model benchmarking.

    Args:
        hidden_size: The hidden size of the model.
        seq_len_list: The list of sequence lengths to use for the benchmark.
        multiple_of: The multiple of the padding length to use for the benchmark.
        dtype: The dtype of the hidden state inputs.
        device: The device to use for the inputs.
        return_input_ids: Whether to include input_ids in the returned dictionary.
        return_inputs_embeds: Whether to include inputs_embeds in the returned
            dictionary.
        return_labels: Whether to include labels in the returned dictionary.

    Returns:
        A dictionary containing dummy inputs for the model.

    Raises:
        ValueError: If seq_len_list is not a valid list.
    """
    # See:
    # https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py#L212-L239
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/modeling_flash_attention_utils.py#L136-L183
    # The inputs should have
    # position_ids, cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k
    if device is None:
        device = torch.cuda.current_device()

    # Seq len list must be provided
    if seq_len_list is None or not isinstance(seq_len_list, list):
        msg = f"seq_len_list must be a list, got {seq_len_list}"
        raise ValueError(msg)

    # Calculate the padded total sequence length
    total_seq_len = sum(seq_len_list)
    if multiple_of is not None:
        padded_total_seq_len = ((total_seq_len - 1) // multiple_of + 1) * multiple_of
    else:
        padded_total_seq_len = total_seq_len

    # Create the packed inputs
    packed_input_ids = torch.randint(
        low=0,
        high=10000,
        size=(1, padded_total_seq_len),
        device=device,
        dtype=torch.long,
    )
    packed_hidden_states = torch.randn(
        (1, padded_total_seq_len, hidden_size),
        dtype=dtype,
        requires_grad=True,
        device=device,
    )

    # Create position_ids for each sequence in seq_len_list
    # > e.g. Tensor[0, 1, 2, ..., N_1, 0, 1, 2, ..., N_2, ...]
    position_ids_list = [
        torch.arange(seq_len, device=device, dtype=torch.int32)
        for seq_len in seq_len_list
    ]
    position_ids = torch.cat(position_ids_list).unsqueeze(0)
    flattened_position_ids = position_ids.flatten()
    indices_q = torch.arange(
        flattened_position_ids.shape[0],
        device=device,
        dtype=torch.int32,
    )
    cu_seq_lens_q = torch.cat(
        (
            indices_q[flattened_position_ids == 0],
            torch.tensor(
                flattened_position_ids.size(),
                device=flattened_position_ids.device,
                dtype=torch.int32,
            ),
        )
    )
    max_length_q = flattened_position_ids.max().item() + 1

    # Build the base inputs dict
    inputs: dict[str, torch.Tensor | int] = {
        "position_ids": position_ids,
        "cu_seq_lens_q": cu_seq_lens_q,
        "cu_seq_lens_k": cu_seq_lens_q,
        "max_length_q": max_length_q,
        "max_length_k": max_length_q,
    }

    # Add optional inputs based on flags
    if return_input_ids:
        inputs["input_ids"] = packed_input_ids

    if return_inputs_embeds:
        inputs["inputs_embeds"] = packed_hidden_states

    if return_labels:
        inputs["labels"] = torch.randint(
            low=0,
            high=10000,
            size=(1, padded_total_seq_len),
            device=device,
            dtype=torch.long,
        )

    return inputs
