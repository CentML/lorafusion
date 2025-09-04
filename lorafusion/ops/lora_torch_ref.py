"""PyTorch reference implementations of LoRA and multi-LoRA."""

import torch


def torch_ref_linear_lora(
    x: torch.Tensor,
    linear_w: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    alpha: float,
    dropout_p: float = 0.0,
    linear_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """PyTorch reference implementation of linear+LoRA.

    This uses standard PyTorch operations instead of fused kernels.

    Args:
        x: Input tensor [B, M, D]
        linear_w: Weight tensor for linear layer [N, D]
        lora_a: Weight A for LoRA [r, D]
        lora_b: Weight B for LoRA [N, r]
        alpha: Scaling factor for LoRA
        dropout_p: Dropout probability
        linear_bias: Bias for linear layer

    Returns:
        Output tensor [B, M, N]
    """
    # Standard linear operation
    result = x @ linear_w.T

    # Add bias if provided
    if linear_bias is not None:
        result = result + linear_bias

    # Apply dropout to LoRA input if needed
    if dropout_p > 0.0 and torch.is_grad_enabled():
        x_for_lora = torch.nn.functional.dropout(x, p=dropout_p)
    else:
        x_for_lora = x

    # Apply LoRA: x @ A.T @ B.T * (alpha/r)
    lora_output = (x_for_lora @ lora_a.T @ lora_b.T) * alpha

    # Add LoRA contribution to the result
    return result + lora_output


def torch_ref_linear_multi_lora(
    padded_x: torch.Tensor,
    linear_w: torch.Tensor,
    *,
    lora_a_list: list[torch.Tensor],
    lora_b_list: list[torch.Tensor],
    seq_len_list: list[int],
    padded_seq_len_list: list[int],
    alpha_list: list[float],
    dropout_p_list: list[float],
    linear_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """PyTorch reference implementation of multi-LoRA.

    This function handles multiple LoRA adapters, each applied to a different segment
    of the input sequence. Uses standard PyTorch operations instead of fused kernels.

    Args:
        padded_x: Input tensor [B, sum(padded_seq_len_list), D]
        linear_w: Weight tensor for linear layer [N, D]
        lora_a_list: List of A weights for each LoRA adapter
        lora_b_list: List of B weights for each LoRA adapter
        seq_len_list: List of sequence lengths for each adapter
        padded_seq_len_list: List of padded sequence lengths for each adapter
        alpha_list: List of scaling factors for each adapter
        dropout_p_list: List of dropout probabilities for each adapter
        linear_bias: Bias for linear layer

    Returns:
        Output tensor [B, sum(padded_seq_len_list), N]
    """
    # Apply the base linear layer to the entire input
    result = padded_x @ linear_w.T

    # Add bias if provided
    if linear_bias is not None:
        result = result + linear_bias

    # Keep track of current position in the sequence
    curr_pos = 0

    # Process each adapter segment
    for seq_len, padded_seq_len, lora_a, lora_b, alpha, dropout_p in zip(
        seq_len_list,
        padded_seq_len_list,
        lora_a_list,
        lora_b_list,
        alpha_list,
        dropout_p_list,
        strict=True,
    ):
        # Extract the portion of x for this adapter
        x_segment = padded_x[curr_pos : curr_pos + seq_len]

        # Apply dropout if needed
        if dropout_p > 0.0 and torch.is_grad_enabled():
            x_segment_dropped = torch.nn.functional.dropout(x_segment, p=dropout_p)
        else:
            x_segment_dropped = x_segment

        # Apply LoRA for this segment
        lora_output = (x_segment_dropped @ lora_a.T @ lora_b.T) * alpha

        # Add to the corresponding part of the result
        result[curr_pos : curr_pos + seq_len] += lora_output

        # Update position counter - use padded sequence length to move to next segment
        curr_pos += padded_seq_len

    return result
