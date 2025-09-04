# ruff: noqa
"""Triton TMA utils."""

import torch
import triton
import triton.language as tl

HAS_TMA_DESC = "nv_tma_desc_type" in dir(tl)


class TmaAutoTuneHelper:
    """TMA Auto-Tune Helper.

    TmaAutoTuneHelper used in htyu's PR #5622
    """

    class KernelParamWrapper:
        """Kernel Param Wrapper.

        Duck typing wrapper to implement the same interface as TmaDescKernelParam in
        Triton PR #4498
        """

        def __init__(self, desc) -> None:
            """Initialize the KernelParamWrapper.

            Args:
                desc: The TMA descriptor.
            """
            self.desc = desc

        def tma_desc_cpu_ptr(self) -> int:
            """TMA Descriptor CPU Pointer.

            Returns the data pointer of the TMA descriptor.
            """
            return self.desc.data_ptr()

    TMA_SIZE = 128

    def __init__(self) -> None:
        """Initialize the TmaAutoTuneHelper.

        Args:
            None
        """
        self.fill_1d_tma_descriptor_inner = (
            triton.runtime.driver.active.utils.fill_1d_tma_descriptor
        )
        self.fill_2d_tma_descriptor_inner = (
            triton.runtime.driver.active.utils.fill_2d_tma_descriptor
        )
        if HAS_TMA_DESC:
            self.descriptors = {}
        else:
            self.cuda_descriptors = {}

    def init_tma_descriptor(self, name) -> None:
        """Initialize the TMA descriptor.

        Args:
            name: The name of the descriptor.
        """
        if HAS_TMA_DESC:
            self.descriptors[name] = torch.empty(
                TmaAutoTuneHelper.TMA_SIZE, device="cpu", dtype=torch.int8
            )
        else:
            self.cuda_descriptors[name] = torch.empty(
                TmaAutoTuneHelper.TMA_SIZE, device="cuda", dtype=torch.int8
            )

    def fill_1d_tma_descriptor(self, name, ptr, dim, block_dim, element_size) -> None:
        """Fill the 1D TMA descriptor.

        Args:
            name: The name of the descriptor.
            ptr: The pointer to the descriptor.
            dim: The dimension of the descriptor.
            block_dim: The block dimension of the descriptor.
            element_size: The element size of the descriptor.
        """
        if HAS_TMA_DESC:
            desc_x = self.descriptors[name]
            if desc_x.data_ptr() % 64 != 0:
                msg = "TMA descriptor is not 64-byte aligned. This may cause performance issues."
                raise ValueError(msg)
            self.fill_1d_tma_descriptor_inner(
                ptr, dim, block_dim, element_size, desc_x.data_ptr()
            )
        else:
            desc_x = self.cuda_descriptors[name]
            buf_x = torch.empty_like(desc_x, device="cpu", pin_memory=True)
            self.fill_1d_tma_descriptor_inner(
                ptr, dim, block_dim, element_size, buf_x.data_ptr()
            )
            desc_x.copy_(buf_x, non_blocking=True)

    # Call this method inside the lambda function for grid size
    def fill_2d_tma_descriptor(
        self, name, ptr, dim1, dim0, block_dim1, block_dim0, element_size
    ) -> None:
        """Fill the 2D TMA descriptor.

        Args:
            name: The name of the descriptor.
            ptr: The pointer to the descriptor.
            dim1: The dimension of the descriptor.
            dim0: The dimension of the descriptor.
            block_dim1: The block dimension of the descriptor.
            block_dim0: The block dimension of the descriptor.
            element_size: The element size of the descriptor.
        """
        if HAS_TMA_DESC:
            desc_x = self.descriptors[name]
            if desc_x.data_ptr() % 64 != 0:
                msg = (
                    "TMA descriptor is not 64-byte aligned. "
                    "This may cause performance issues."
                )
                raise ValueError(msg)
            self.fill_2d_tma_descriptor_inner(
                ptr, dim1, dim0, block_dim1, block_dim0, element_size, desc_x.data_ptr()
            )
        else:
            desc_x = self.cuda_descriptors[name]
            buf_x = torch.empty_like(desc_x, device="cpu", pin_memory=True)
            self.fill_2d_tma_descriptor_inner(
                ptr, dim1, dim0, block_dim1, block_dim0, element_size, buf_x.data_ptr()
            )
            desc_x.copy_(buf_x, non_blocking=True)

    def get_tma_descriptor_kernel_param(self, name) -> KernelParamWrapper:
        """Get the TMA descriptor kernel parameter.

        Args:
            name: The name of the descriptor.
        """
        if HAS_TMA_DESC:
            if self.descriptors[name] is None:
                msg = (
                    "TMA descriptor is not initialized. "
                    "Please call init_tma_descriptor first."
                )
                raise ValueError(msg)
            return self.KernelParamWrapper(self.descriptors[name])
        if self.cuda_descriptors[name] is None:
            msg = (
                "TMA descriptor is not initialized. "
                "Please call init_tma_descriptor first."
            )
            raise ValueError(msg)
        return self.cuda_descriptors[name]


@triton.jit
def _compute_pid(
    tile_id,
    num_pid_in_group,
    num_pid_m,
    GROUP_SIZE_M,
    NUM_SMS,
):
    """Compute the pid for the persistent matmul kernel."""
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


tl_experimental_descriptor_load = tl._experimental_descriptor_load
tl_experimental_descriptor_store = tl._experimental_descriptor_store
