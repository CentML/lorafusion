"""LoRA operations."""

from __future__ import annotations

import os
import random
import warnings

import torch

from lorafusion.ops.triton_ops.dropout import seeded_dropout
from lorafusion.ops.triton_ops.fused_dropout_matmul import fused_dropout_matmul
from lorafusion.ops.triton_ops.fused_lora_dys_dyb import fused_lora_dys_dyb
from lorafusion.ops.triton_ops.fused_lora_dyw_dsa import fused_lora_dyw_dsa
from lorafusion.ops.triton_ops.fused_lora_dyw_dsa_tma import fused_lora_dyw_dsa_tma
from lorafusion.ops.triton_ops.fused_lora_xw_sb import fused_lora_xw_sb
from lorafusion.ops.triton_ops.fused_lora_xw_sb_tma import fused_lora_xw_sb_tma
from lorafusion.ops.triton_ops.tma_utils import HAS_TMA_DESC


def _get_bool_env_var(env_var: str, *, default: bool = False) -> bool:
    """Get a boolean value from an environment variable.

    Args:
        env_var: The environment variable name.
        default: The default value if the environment variable is not set.

    Returns:
        The boolean value from the environment variable or default.
    """
    value = os.environ.get(env_var, "").lower()
    if value in ("true", "1", "yes", "on"):
        return True
    if value in ("false", "0", "no", "off"):
        return False
    return default


def _use_tma_kernel() -> bool:
    """Check whether to use TMA kernel. It also needs SM >= 90 since it uses cp.async.

    Returns:
        True if TMA is supported and SM >= 90, False otherwise.
    """
    return HAS_TMA_DESC and torch.cuda.get_device_properties().major >= 9 # noqa: PLR2004


# Configuration from environment variables with hardware checks
FORWARD_USE_TMA_ENV = _get_bool_env_var("LORAFUSION_FORWARD_USE_TMA", default=True)
BACKWARD_USE_TMA_ENV = _get_bool_env_var("LORAFUSION_BACKWARD_USE_TMA", default=False)
USE_FUSED_DROPOUT_MATMUL_ENV = _get_bool_env_var(
    "LORAFUSION_USE_FUSED_DROPOUT_MATMUL", default=False
)

# Check hardware support and apply environment variable settings
HARDWARE_USE_TMA = _use_tma_kernel()

# Apply configuration with hardware checks
FORWARD_USE_TMA = FORWARD_USE_TMA_ENV and HARDWARE_USE_TMA
BACKWARD_USE_TMA = BACKWARD_USE_TMA_ENV and HARDWARE_USE_TMA
USE_FUSED_DROPOUT_MATMUL = USE_FUSED_DROPOUT_MATMUL_ENV

# Warn users if TMA was requested but not supported
if FORWARD_USE_TMA_ENV and not HARDWARE_USE_TMA:
    warnings.warn(
        "LORAFUSION_FORWARD_USE_TMA is enabled but it is not supported by the hardware."
        " Falling back to non-TMA implementation.",
        UserWarning,
        stacklevel=2,
    )

if BACKWARD_USE_TMA_ENV and not HARDWARE_USE_TMA:
    warnings.warn(
        "LORAFUSION_BACKWARD_USE_TMA is enabled but it is not supported by the hardware."
        " Falling back to non-TMA implementation.",
        UserWarning,
        stacklevel=2,
    )

# Select appropriate functions based on configuration
fused_lora_xw_sb_func = fused_lora_xw_sb_tma if FORWARD_USE_TMA else fused_lora_xw_sb
fused_lora_dyw_dsa_func = (
    fused_lora_dyw_dsa_tma if BACKWARD_USE_TMA else fused_lora_dyw_dsa
)


def _fused_linear_lora_forward(
    x: torch.Tensor,
    linear_w: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    alpha: float,
    dropout_p: float,
    seed: int,
    linear_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Fused linear and LoRA forward."""
    dropout_mask: torch.Tensor | None = None
    masked_scaled_x: torch.Tensor = x
    if dropout_p > 0:
        if USE_FUSED_DROPOUT_MATMUL:
            s, dropout_mask, masked_scaled_x = fused_dropout_matmul(
                x=x,
                a=lora_a,
                dropout_p=dropout_p,
                seed=seed,
                store_mask=True,
                store_masked_scaled_x=True,
            )
        else:
            masked_scaled_x, dropout_mask = seeded_dropout(
                x=x,
                p=dropout_p,
                seed=seed,
                store_mask=True,
            )
            s = masked_scaled_x @ lora_a.T
    else:
        s = x @ lora_a.T

    y = fused_lora_xw_sb_func(
        x=x,
        w=linear_w,
        s=s,
        b=lora_b,
        alpha=alpha,
        bias=linear_bias,
    )
    return y, dropout_mask, masked_scaled_x, s


def _fused_linear_lora_backward(
    dy: torch.Tensor,
    linear_w: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    s: torch.Tensor,
    masked_scaled_x: torch.Tensor,
    alpha: float,
    dropout_p: float,
    dropout_mask: torch.Tensor | None,
    seed: int,
    *,
    requires_dx: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused linear and LoRA backward."""
    db, ds = fused_lora_dys_dyb(
        dy=dy,
        b=lora_b,
        s=s,
        alpha=alpha,
    )
    da = ds.T @ masked_scaled_x
    if requires_dx:
        dx = fused_lora_dyw_dsa_func(
            dy=dy,
            w=linear_w,
            ds=ds,
            a=lora_a,
            dropout_p=dropout_p,
            dropout_mask=dropout_mask,
        )
    else:
        dx = None

    return dx, da, db, ds


class FusedLinearLoRA(torch.autograd.Function):
    """Fused linear and LoRA."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionContext,
        x: torch.Tensor,
        linear_w: torch.Tensor,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        alpha: float,
        dropout_p: float,
        seed: int | None = None,
        linear_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Computes:
            output = (x @ linear_w.T) + (dropout(x) @ lora_a.T @ lora_b.T) * alpha
                        + linear_bias

        Args:
            ctx: The context object.
            x: The input tensor.
            linear_w: The weight tensor of the linear layer.
            lora_a: The weight tensor of the LoRA layer.
            lora_b: The weight tensor of the LoRA layer.
            alpha: The scaling factor for the LoRA layer.
            dropout_p: The dropout probability.
            seed: The seed for the dropout.
            linear_bias: The bias tensor of the linear layer.
        """
        if seed is None:
            seed = random.randrange(int(1e6))  # noqa: S311

        # Reshape x to 2D
        ndim_3 = 3
        ndim_2 = 2
        if x.ndim == ndim_3:
            bsz, seq, _ = x.shape
            x = x.reshape(bsz * seq, -1)
        elif x.ndim != ndim_2:
            msg = f"Expected x to have {ndim_2} dimensions, got {x.ndim}."
            raise ValueError(msg)

        y, dropout_mask, masked_scaled_x, s = _fused_linear_lora_forward(
            x=x,
            linear_w=linear_w,
            lora_a=lora_a,
            lora_b=lora_b,
            alpha=alpha,
            dropout_p=dropout_p,
            seed=seed,
            linear_bias=linear_bias,
        )
        ctx.save_for_backward(
            masked_scaled_x, linear_w, lora_a, lora_b, dropout_mask, s
        )
        ctx.alpha = alpha
        ctx.dropout_p = dropout_p
        ctx.seed = seed

        # Convert output to x_shapes
        return y.reshape(bsz, seq, -1)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionContext,
        dy: torch.Tensor,
    ) -> tuple[torch.Tensor, None, torch.Tensor, torch.Tensor, None, None, None, None]:
        """Backward pass.

        Computes:
            db = (dy.T @ s) * alpha
            ds = dy @ b * alpha
            da = ds.T @ x
            dx_1 = dy @ w
            dx_2 = ds @ a * dropout_mask / (1 - dropout_p)
            dx = dx_1 + dx_2

        Args:
            ctx: The context object.
            dy: The gradient of the output.
        """
        masked_scaled_x, linear_w, lora_a, lora_b, dropout_mask, s = ctx.saved_tensors

        # Change the shape of dy to 2D
        ndim_3 = 3
        ndim_2 = 2
        if dy.ndim == ndim_3:
            bsz, seq, _ = dy.shape
            dy = dy.reshape(bsz * seq, -1)
        elif dy.ndim != ndim_2:
            msg = f"Expected dy to have {ndim_2} dimensions, got {dy.ndim}."
            raise ValueError(msg)

        dx, da, db, ds = _fused_linear_lora_backward(
            dy=dy,
            linear_w=linear_w,
            lora_a=lora_a,
            lora_b=lora_b,
            s=s,
            masked_scaled_x=masked_scaled_x,
            alpha=ctx.alpha,
            dropout_p=ctx.dropout_p,
            dropout_mask=dropout_mask,
            seed=ctx.seed,
            requires_dx=ctx.needs_input_grad[0],
        )
        if dx is not None:
            dx = dx.reshape(bsz, seq, -1)

        return dx, None, da, db, None, None, None, None


def fused_linear_lora(
    x: torch.Tensor,
    linear_w: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    alpha: float,
    seed: int | None = None,
    dropout_p: float = 0.0,
    linear_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused linear and LoRA."""
    return FusedLinearLoRA.apply(
        x, linear_w, lora_a, lora_b, alpha, dropout_p, seed, linear_bias
    )


dropout = torch.nn.Dropout(p=0.1)
x = torch.randn(10, 10)
a = torch.randn(10, 10)

hat_x = dropout(x)
s = hat_x @ a.T
