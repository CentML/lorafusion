"""Testing utilities."""

import torch
from loguru import logger

_DTYPE_PRECISIONS = {
    # Meaning: dtype: (rtol, atol)
    torch.float16: (0.001, 1e-5),
    torch.bfloat16: (0.016, 1e-5),
    torch.float32: (1.3e-6, 1e-5),
    torch.float64: (1e-7, 1e-7),
    torch.complex32: (0.001, 1e-5),
    torch.complex64: (1.3e-6, 1e-5),
    torch.complex128: (1e-7, 1e-7),
}


def get_default_rtol_atol(dtype: torch.dtype) -> tuple[float, float]:
    """Get the default absolute and relative tolerance for a given dtype."""
    return _DTYPE_PRECISIONS[dtype]


def verbose_allclose(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    print_error: bool = True,
    max_print: int | None = 10,
    msg: str | None = None,
) -> int:
    """Compare two tensors and print out the mismatched elements.

    Args:
        actual: The actual tensor.
        expected: The expected tensor.
        atol: The absolute tolerance.
        rtol: The relative tolerance.
        print_error: Whether to print the error.
        max_print: The maximum number of elements to print. If None, all elements
            will be printed.
        msg: The message to print.
    """
    rtol = rtol or get_default_rtol_atol(actual.dtype)[0]
    atol = atol or get_default_rtol_atol(actual.dtype)[1]

    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as e:
        close_mask = torch.isclose(
            actual, expected, rtol=rtol, atol=atol, equal_nan=True
        )
        mismatch_mask = ~close_mask

        # Gather all mismatched indices
        mismatched_indices = mismatch_mask.nonzero(as_tuple=True)

        # 3) Retrieve actual and expected values at these indices
        mismatch_actual = actual[mismatched_indices]
        mismatch_expected = expected[mismatched_indices]

        # Print out a summary
        total_mismatches = mismatch_mask.sum().item()

        if print_error:
            logger.error(f"[{msg}] ================================================")
            logger.error(f"[{msg}] {atol=}, {rtol=}")
            logger.error(e)

            # 4) Print each mismatched index and the corresponding values
            for idx in range(min(total_mismatches, max_print)):
                # Convert the nth mismatch to a multi-dimensional index if needed
                index_tuple = tuple(
                    dim_idx[idx].item() for dim_idx in mismatched_indices
                )
                actual_val = mismatch_actual[idx].item()
                expected_val = mismatch_expected[idx].item()
                logger.error(
                    f"[{msg}] Index {index_tuple}:\n"
                    f"  actual  = {actual_val}\n"
                    f"  expected= {expected_val}\n"
                    f"  diff    = {abs(actual_val - expected_val)}\n"
                    f"  threshold = {atol + rtol * abs(expected_val)}\n"
                )

            logger.error(f"[{msg}] {atol=}, {rtol=}")
            logger.error(e)
            logger.error(f"[{msg}] Total mismatches: {total_mismatches}")
            logger.error(f"[{msg}] ================================================")

        return total_mismatches

    else:
        return 0


def assert_verbose_allclose_two_rounds(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float | None = None,
    rtol: float | None = None,
    max_print: int | None = 10,
    arithmetic_error_ratio: float = 2.5,
    mismatch_ratio_threshold: float = 1e-3,
    msg: str | None = None,
) -> None:
    """Assert that two tensors are close to each other.

    This function will check the following:
    1. If the original torch.testing.assert_close passes, then the function will pass.
    2. If the original torch.testing.assert_close fails, then the function will check:
        - If the matching ratio is greater than the threshold, AND
        - If torch.testing.assert_close passes with
            atol=arithmetic_error_ratio * tested_atol, then the function will pass.
        - Only when the above two conditions are met, the function will pass.

    Args:
        actual: The actual tensor.
        expected: The expected tensor.
        atol: The absolute tolerance.
        rtol: The relative tolerance.
        max_print: The maximum number of elements to print. If None, all elements
            will be printed.
        arithmetic_error_ratio: The ratio of allowed arithmetic error.
        mismatch_ratio_threshold: The threshold of mismatch ratio.
        msg: The message to print.
    """
    # First check if the tensors are accurately close
    # this is just to check if the tensors are close to each other
    # so we don't need to print out the mismatched elements
    # details are printed in the second check
    first_check_mismatches = verbose_allclose(
        actual, expected, atol=atol, rtol=rtol, max_print=0, print_error=False
    )
    if first_check_mismatches == 0:
        return

    # 1. Check if the matching ratio is greater than the threshold
    if first_check_mismatches / actual.numel() >= mismatch_ratio_threshold:
        mismatches = verbose_allclose(
            actual,
            expected,
            atol=atol,
            rtol=rtol,
            max_print=max_print,
            print_error=True,
            msg=msg,
        )
        if mismatches != 0:
            log_msg = (
                f"[{msg}] Failed with atol={atol} and rtol={rtol}, mismatch ratio: "
                f"{first_check_mismatches / actual.numel() * 100:.4f}% "
                f"({first_check_mismatches}/{actual.numel()}), "
            )
            raise AssertionError(log_msg)

    # 2. Check if the tensors are close with the allowed arithmetic error
    # since each arithmetic opeator introduces some error
    tested_atol = ((expected + 0.3 - 0.3) - expected).abs().max().item()
    scaled_tested_atol = arithmetic_error_ratio * tested_atol
    mismatches = verbose_allclose(
        actual,
        expected,
        atol=scaled_tested_atol,
        rtol=rtol,
        max_print=max_print,
        print_error=True,
        msg=msg,
    )
    if mismatches != 0:
        log_msg = (
            f"[{msg}] Failed with atol={atol} and rtol={rtol}, "
            f"mismatch ratio: {first_check_mismatches / actual.numel() * 100:.4f}% "
            f"({first_check_mismatches}/{actual.numel()}), "
            f"and also failed with atol={scaled_tested_atol} and rtol={rtol}, "
            f"mismatch ratio: {mismatches / actual.numel() * 100:.4f}% "
            f"({mismatches}/{actual.numel()}). "
        )
        raise AssertionError(log_msg)

    # If both check passes, then the function will pass
    logger.info(
        f"[{msg}] Failed with atol={atol} and rtol={rtol}, "
        f"mismatch ratio: {first_check_mismatches / actual.numel() * 100:.4f}% "
        f"({first_check_mismatches}/{actual.numel()}), "
        f"but passed with atol={scaled_tested_atol} and rtol={rtol}. "
    )
