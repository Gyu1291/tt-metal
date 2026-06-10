# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Sub-device overlap scenario harness.

This file is intentionally separate from the plain matmul throughput sweep.  It
keeps a small set of model-inspired overlap experiments that are useful when
checking whether disjoint sub-devices and hardware command queues can hide work:

  * decode prefetcher-like work with decode compute
  * MoE dispatch-like work with a shared-expert-like matmul
  * independent small/medium ops on disjoint sub-devices
  * equal-size routed expert FFNs on sub-devices vs full-grid serial execution
  * CCL + compute is reported as a multi-device TODO row by default

The first three scenarios are single-card microbenches.  They are not a full
model replacement; they use the same sub-device/CQ scheduling style as the
model paths while keeping setup cheap and editable.

Useful overrides:
    SUBDEVICE_OVERLAP_SCENARIOS=decode_prefetch_compute,moe_dispatch_shared,independent_small_medium,expert_ffn_cpp_fused,ccl_compute
    SUBDEVICE_OVERLAP_DTYPE=bf8
    SUBDEVICE_OVERLAP_FIDELITY=HiFi2
    SUBDEVICE_OVERLAP_FP32_ACC=0
    SUBDEVICE_OVERLAP_PACKER_L1_ACC=0
    SUBDEVICE_OVERLAP_WARMUP=2
    SUBDEVICE_OVERLAP_ITERS=5
    SUBDEVICE_OVERLAP_GROUPS_PER_ITER=1
    SUBDEVICE_OVERLAP_NUM_CQS=2
    SUBDEVICE_OVERLAP_DECODE_PREFETCH_SHAPE=32x2048x2048
    SUBDEVICE_OVERLAP_DECODE_COMPUTE_SHAPE=32x2048x2048
    SUBDEVICE_OVERLAP_MOE_DISPATCH_SHAPE=32x2048x768
    SUBDEVICE_OVERLAP_MOE_SHARED_SHAPE=64x8192x8192
    SUBDEVICE_OVERLAP_INDEPENDENT_SHAPES=64x2048x768,64x768x2048
    SUBDEVICE_OVERLAP_INDEPENDENT_SUBDEVICES=2
    SUBDEVICE_OVERLAP_EXPERT_FFN_SHAPE=64x2048x768  # rows x emb_dim x hidden_dim
    SUBDEVICE_OVERLAP_EXPERT_FFN_EXPERTS=2
    SUBDEVICE_OVERLAP_EXPERT_FFN_ACTIVATION_DTYPE=bf16
    SUBDEVICE_OVERLAP_EXPERT_FFN_WEIGHT_DTYPE=bf8
    SUBDEVICE_OVERLAP_CSV=/home/tenstorrent/tt-metal/subdevice_opt/subdevice_overlap_scenarios.csv
"""

from __future__ import annotations

import csv
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import pytest
import torch

import ttnn


_TILE = 32
_MAX_COMMAND_QUEUES = 2
_DEFAULT_CSV = Path(__file__).resolve().with_name("subdevice_overlap_scenarios.csv")
_DEFAULT_SCENARIOS = [
    "decode_prefetch_compute",
    "moe_dispatch_shared",
    "independent_small_medium",
    "expert_ffn_cpp_fused",
    "ccl_compute",
]


@dataclass(frozen=True)
class OverlapConfig:
    dtype: ttnn.DataType
    dtype_name: str
    math_fidelity: object
    math_fidelity_name: str
    fp32_dest_acc_en: bool
    packer_l1_acc: bool
    math_approx_mode: bool
    warmup_iterations: int
    measured_iterations: int
    groups_per_iteration: int
    num_command_queues: int
    decode_prefetch_shape: tuple[int, int, int]
    decode_compute_shape: tuple[int, int, int]
    moe_dispatch_shape: tuple[int, int, int]
    moe_shared_shape: tuple[int, int, int]
    independent_shapes: tuple[tuple[int, int, int], ...]
    independent_subdevices: int
    expert_ffn_shape: tuple[int, int, int]
    expert_ffn_num_experts: int
    expert_ffn_activation_dtype: ttnn.DataType
    expert_ffn_activation_dtype_name: str
    expert_ffn_weight_dtype: ttnn.DataType
    expert_ffn_weight_dtype_name: str
    dispatch_rows: int
    prefetch_rows: int
    in0_block_w_max: int
    l1_tile_budget: int
    out_block_tile_budget: int
    csv_path: Path
    timeout_s: int
    scenarios: tuple[str, ...]


@dataclass
class MatmulWork:
    name: str
    role: str
    shape: tuple[int, int, int]
    tt_a: ttnn.Tensor
    tt_b: ttnn.Tensor
    full_program_config: ttnn.MatmulMultiCoreReuseMultiCastProgramConfig
    sub_program_config: ttnn.MatmulMultiCoreReuseMultiCastProgramConfig
    sub_device_id: ttnn.SubDeviceId
    queue_id: int

    @property
    def flops(self) -> float:
        m, n, k = self.shape
        return 2.0 * m * n * k


@dataclass
class BinaryWork:
    name: str
    role: str
    shape: tuple[int, int]
    op_name: str
    op_fn: Callable
    tt_a: ttnn.Tensor
    tt_b: ttnn.Tensor
    sub_device_id: ttnn.SubDeviceId
    queue_id: int

    @property
    def flops(self) -> float:
        m, n = self.shape
        return float(m * n)


Work = MatmulWork | BinaryWork


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def _split_env_list(name: str, default: Iterable[str] = ()) -> list[str]:
    value = os.getenv(name)
    if value is None:
        return list(default)
    return [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]


def _parse_shape(value: str) -> tuple[int, int, int]:
    normalized = value.lower().replace(" ", "")
    for separator in ("x", ":", "/"):
        if separator in normalized:
            parts = normalized.split(separator)
            break
    else:
        parts = normalized.split("_")
    if len(parts) != 3:
        raise ValueError(f"Unsupported shape={value!r}; use MxNxK, for example 64x2048x768")
    shape = tuple(int(part) for part in parts)
    for dim_name, dim in zip(("M", "N", "K"), shape):
        if dim % _TILE != 0:
            raise ValueError(f"{dim_name} in shape={value!r} must be divisible by {_TILE}")
    return shape


def _parse_dtype(value: str) -> tuple[ttnn.DataType, str]:
    normalized = value.lower()
    if normalized in {"bf16", "bfloat16"}:
        return ttnn.bfloat16, "BFLOAT16"
    if normalized in {"bf8", "bfloat8", "bfloat8_b"}:
        return ttnn.bfloat8_b, "BFLOAT8_B"
    if normalized in {"bf4", "bfloat4", "bfloat4_b"}:
        return ttnn.bfloat4_b, "BFLOAT4_B"
    raise ValueError(f"Unsupported SUBDEVICE_OVERLAP_DTYPE={value!r}; use bf16, bf8, or bf4")


def _parse_math_fidelity(value: str):
    normalized = value.lower().replace("_", "")
    mapping = {
        "lofi": (ttnn.MathFidelity.LoFi, "LoFi"),
        "hifi2": (ttnn.MathFidelity.HiFi2, "HiFi2"),
        "hifi3": (getattr(ttnn.MathFidelity, "HiFi3", ttnn.MathFidelity.HiFi2), "HiFi3"),
        "hifi4": (ttnn.MathFidelity.HiFi4, "HiFi4"),
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported SUBDEVICE_OVERLAP_FIDELITY={value!r}; use LoFi, HiFi2, HiFi3, or HiFi4")
    return mapping[normalized]


def _load_config() -> OverlapConfig:
    dtype, dtype_name = _parse_dtype(os.getenv("SUBDEVICE_OVERLAP_DTYPE", "bf8"))
    math_fidelity, math_fidelity_name = _parse_math_fidelity(os.getenv("SUBDEVICE_OVERLAP_FIDELITY", "HiFi2"))
    expert_ffn_activation_dtype, expert_ffn_activation_dtype_name = _parse_dtype(
        os.getenv("SUBDEVICE_OVERLAP_EXPERT_FFN_ACTIVATION_DTYPE", "bf16")
    )
    expert_ffn_weight_dtype, expert_ffn_weight_dtype_name = _parse_dtype(
        os.getenv("SUBDEVICE_OVERLAP_EXPERT_FFN_WEIGHT_DTYPE", "bf8")
    )
    scenarios = tuple(_split_env_list("SUBDEVICE_OVERLAP_SCENARIOS", _DEFAULT_SCENARIOS))
    config = OverlapConfig(
        dtype=dtype,
        dtype_name=dtype_name,
        math_fidelity=math_fidelity,
        math_fidelity_name=math_fidelity_name,
        fp32_dest_acc_en=_env_flag("SUBDEVICE_OVERLAP_FP32_ACC", False),
        packer_l1_acc=_env_flag("SUBDEVICE_OVERLAP_PACKER_L1_ACC", False),
        math_approx_mode=_env_flag("SUBDEVICE_OVERLAP_MATH_APPROX", False),
        warmup_iterations=_env_int("SUBDEVICE_OVERLAP_WARMUP", 2),
        measured_iterations=_env_int("SUBDEVICE_OVERLAP_ITERS", 5),
        groups_per_iteration=_env_int("SUBDEVICE_OVERLAP_GROUPS_PER_ITER", 1),
        num_command_queues=_env_int("SUBDEVICE_OVERLAP_NUM_CQS", 2),
        decode_prefetch_shape=_parse_shape(os.getenv("SUBDEVICE_OVERLAP_DECODE_PREFETCH_SHAPE", "32x2048x2048")),
        decode_compute_shape=_parse_shape(os.getenv("SUBDEVICE_OVERLAP_DECODE_COMPUTE_SHAPE", "32x2048x2048")),
        moe_dispatch_shape=_parse_shape(os.getenv("SUBDEVICE_OVERLAP_MOE_DISPATCH_SHAPE", "32x2048x768")),
        moe_shared_shape=_parse_shape(os.getenv("SUBDEVICE_OVERLAP_MOE_SHARED_SHAPE", "64x8192x8192")),
        independent_shapes=tuple(
            _parse_shape(value)
            for value in _split_env_list(
                "SUBDEVICE_OVERLAP_INDEPENDENT_SHAPES",
                ["64x2048x768", "64x768x2048"],
            )
        ),
        independent_subdevices=_env_int("SUBDEVICE_OVERLAP_INDEPENDENT_SUBDEVICES", 2),
        expert_ffn_shape=_parse_shape(os.getenv("SUBDEVICE_OVERLAP_EXPERT_FFN_SHAPE", "64x2048x768")),
        expert_ffn_num_experts=_env_int("SUBDEVICE_OVERLAP_EXPERT_FFN_EXPERTS", 2),
        expert_ffn_activation_dtype=expert_ffn_activation_dtype,
        expert_ffn_activation_dtype_name=expert_ffn_activation_dtype_name,
        expert_ffn_weight_dtype=expert_ffn_weight_dtype,
        expert_ffn_weight_dtype_name=expert_ffn_weight_dtype_name,
        dispatch_rows=_env_int("SUBDEVICE_OVERLAP_DISPATCH_ROWS", 1),
        prefetch_rows=_env_int("SUBDEVICE_OVERLAP_PREFETCH_ROWS", 1),
        in0_block_w_max=_env_int("SUBDEVICE_OVERLAP_IN0_BLOCK_W_MAX", 16),
        l1_tile_budget=_env_int("SUBDEVICE_OVERLAP_L1_TILE_BUDGET", 1280),
        out_block_tile_budget=_env_int("SUBDEVICE_OVERLAP_OUT_BLOCK_TILE_BUDGET", 256),
        csv_path=Path(os.getenv("SUBDEVICE_OVERLAP_CSV", str(_DEFAULT_CSV))),
        timeout_s=_env_int("SUBDEVICE_OVERLAP_TIMEOUT", 300),
        scenarios=scenarios,
    )
    if config.num_command_queues < 1 or config.num_command_queues > _MAX_COMMAND_QUEUES:
        raise ValueError(f"SUBDEVICE_OVERLAP_NUM_CQS must be in [1, {_MAX_COMMAND_QUEUES}]")
    if config.measured_iterations < 1:
        raise ValueError("SUBDEVICE_OVERLAP_ITERS must be >= 1")
    if config.groups_per_iteration < 1:
        raise ValueError("SUBDEVICE_OVERLAP_GROUPS_PER_ITER must be >= 1")
    if config.expert_ffn_num_experts < 1:
        raise ValueError("SUBDEVICE_OVERLAP_EXPERT_FFN_EXPERTS must be >= 1")
    return config


CONFIG = _load_config()


def _compute_kernel_config(config: OverlapConfig):
    config_type = getattr(ttnn, "BlackholeComputeKernelConfig", None)
    if config_type is None:
        config_type = getattr(getattr(ttnn, "types", object()), "BlackholeComputeKernelConfig", None)
    if config_type is None:
        config_type = ttnn.WormholeComputeKernelConfig
    return config_type(
        math_fidelity=config.math_fidelity,
        math_approx_mode=config.math_approx_mode,
        fp32_dest_acc_en=config.fp32_dest_acc_en,
        packer_l1_acc=config.packer_l1_acc,
    )


def _core_range(x0: int, y0: int, x1: int, y1: int) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))})


def _row_ranges_from_heights(cols: int, rows: int, heights: list[int]) -> list[ttnn.CoreRangeSet]:
    if sum(heights) != rows:
        raise ValueError(f"Sub-device heights={heights} must sum to device rows={rows}")
    ranges = []
    y = 0
    for height in heights:
        if height <= 0:
            raise ValueError(f"Sub-device height must be > 0, got {height}")
        ranges.append(_core_range(0, y, cols - 1, y + height - 1))
        y += height
    return ranges


def _even_row_ranges(cols: int, rows: int, num_subdevices: int) -> list[ttnn.CoreRangeSet]:
    if rows % num_subdevices != 0:
        raise ValueError(f"num_subdevices={num_subdevices} must evenly divide device rows={rows}")
    return _row_ranges_from_heights(cols, rows, [rows // num_subdevices] * num_subdevices)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _divisors_desc(value: int) -> list[int]:
    return [candidate for candidate in range(value, 0, -1) if value % candidate == 0]


def _effective_l1_tile_budget(k_tiles: int, l1_tile_budget: int) -> int:
    if k_tiles >= 128:
        return min(l1_tile_budget, 1024)
    return l1_tile_budget


def _choose_in0_block_w(k_tiles: int, max_block_w: int, per_core_m: int, per_core_n: int, l1_tile_budget: int) -> int:
    if k_tiles >= 128:
        max_block_w = min(max_block_w, 4)
    max_block_w = max(1, min(max_block_w, k_tiles))
    effective_budget = _effective_l1_tile_budget(k_tiles, l1_tile_budget)
    for candidate in range(max_block_w, 0, -1):
        if k_tiles % candidate == 0 and per_core_m * per_core_n * candidate <= effective_budget:
            return candidate
    return 1


def _choose_out_block(per_core_m: int, per_core_n: int, max_block_tiles: int) -> tuple[int, int]:
    max_block_tiles = max(1, max_block_tiles)
    best_h = 1
    best_w = 1
    best_area = 1
    best_ratio = float("inf")
    for candidate_h in _divisors_desc(per_core_m):
        for candidate_w in _divisors_desc(per_core_n):
            area = candidate_h * candidate_w
            if area > max_block_tiles:
                continue
            ratio = max(candidate_h, candidate_w) / max(1, min(candidate_h, candidate_w))
            if area > best_area or (area == best_area and ratio < best_ratio):
                best_h = candidate_h
                best_w = candidate_w
                best_area = area
                best_ratio = ratio
    return best_h, best_w


def _choose_out_subblock(out_block_h: int, out_block_w: int, fp32_dest_acc_en: bool) -> tuple[int, int]:
    subblock_choices = (
        (4, 2),
        (2, 4),
        (8, 1),
        (1, 8),
        (7, 1),
        (1, 7),
        (3, 2),
        (2, 3),
        (6, 1),
        (1, 6),
        (5, 1),
        (1, 5),
        (2, 2),
        (4, 1),
        (1, 4),
        (3, 1),
        (1, 3),
        (2, 1),
        (1, 2),
        (1, 1),
    )
    for candidate_h, candidate_w in subblock_choices:
        if fp32_dest_acc_en and candidate_h * candidate_w > 4:
            continue
        if out_block_h % candidate_h == 0 and out_block_w % candidate_w == 0:
            return candidate_h, candidate_w
    return 1, 1


def _mcast2d_config(
    config: OverlapConfig,
    shape: tuple[int, int, int],
    grid_x: int,
    grid_y: int,
    allowed_worker_cores: ttnn.CoreRangeSet,
) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    m, n, k = shape
    m_tiles = m // _TILE
    n_tiles = n // _TILE
    k_tiles = k // _TILE
    per_core_m = _ceil_div(m_tiles, grid_y)
    per_core_n = _ceil_div(n_tiles, grid_x)
    in0_block_w = _choose_in0_block_w(
        k_tiles, config.in0_block_w_max, per_core_m, per_core_n, config.l1_tile_budget
    )
    out_block_h, out_block_w = _choose_out_block(per_core_m, per_core_n, config.out_block_tile_budget)
    out_subblock_h, out_subblock_w = _choose_out_subblock(out_block_h, out_block_w, config.fp32_dest_acc_en)
    kwargs = {
        "compute_with_storage_grid_size": ttnn.CoreCoord(grid_x, grid_y),
        "in0_block_w": in0_block_w,
        "out_subblock_h": out_subblock_h,
        "out_subblock_w": out_subblock_w,
        "out_block_h": out_block_h,
        "out_block_w": out_block_w,
        "per_core_M": per_core_m,
        "per_core_N": per_core_n,
        "transpose_mcast": False,
        "fused_activation": None,
        "fuse_batch": False,
    }
    try:
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            **kwargs, allowed_worker_cores=allowed_worker_cores
        )
    except TypeError as exc:
        if "allowed_worker_cores" not in str(exc):
            raise
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(**kwargs)


def _program_summary(config: OverlapConfig, shape: tuple[int, int, int], grid_x: int, grid_y: int) -> str:
    m, n, k = shape
    m_tiles = m // _TILE
    n_tiles = n // _TILE
    k_tiles = k // _TILE
    per_core_m = _ceil_div(m_tiles, grid_y)
    per_core_n = _ceil_div(n_tiles, grid_x)
    out_block_h, out_block_w = _choose_out_block(per_core_m, per_core_n, config.out_block_tile_budget)
    out_subblock_h, out_subblock_w = _choose_out_subblock(out_block_h, out_block_w, config.fp32_dest_acc_en)
    in0_block_w = _choose_in0_block_w(
        k_tiles, config.in0_block_w_max, per_core_m, per_core_n, config.l1_tile_budget
    )
    return (
        f"grid={grid_x}x{grid_y} cores={grid_x * grid_y} "
        f"tiles=M{m_tiles}/N{n_tiles}/K{k_tiles} "
        f"per_core={per_core_n}x{per_core_m} "
        f"out_block={out_block_w}x{out_block_h} "
        f"out_subblock={out_subblock_w}x{out_subblock_h} "
        f"in0_block_w={in0_block_w}"
    )


def _make_matmul_inputs(device, config: OverlapConfig, shape: tuple[int, int, int], seed: int):
    m, n, k = shape
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn((1, 1, m, k), dtype=torch.bfloat16, generator=generator)
    b = torch.randn((1, 1, k, n), dtype=torch.bfloat16, generator=generator)
    tt_a = ttnn.from_torch(
        a,
        dtype=config.dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_b = ttnn.from_torch(
        b,
        dtype=config.dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return tt_a, tt_b


def _make_binary_inputs(device, config: OverlapConfig, shape: tuple[int, int], seed: int):
    m, n = shape
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn((1, 1, m, n), dtype=torch.bfloat16, generator=generator)
    b = torch.randn((1, 1, m, n), dtype=torch.bfloat16, generator=generator)
    tt_a = ttnn.from_torch(
        a,
        dtype=config.dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_b = ttnn.from_torch(
        b,
        dtype=config.dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return tt_a, tt_b


def _make_matmul_work(
    device,
    config: OverlapConfig,
    name: str,
    role: str,
    shape: tuple[int, int, int],
    cols: int,
    rows: int,
    full_core_range: ttnn.CoreRangeSet,
    sub_core_range: ttnn.CoreRangeSet,
    sub_rows: int,
    sub_device_id: ttnn.SubDeviceId,
    queue_id: int,
    seed: int,
) -> MatmulWork:
    tt_a, tt_b = _make_matmul_inputs(device, config, shape, seed)
    return MatmulWork(
        name=name,
        role=role,
        shape=shape,
        tt_a=tt_a,
        tt_b=tt_b,
        full_program_config=_mcast2d_config(config, shape, cols, rows, full_core_range),
        sub_program_config=_mcast2d_config(config, shape, cols, sub_rows, sub_core_range),
        sub_device_id=sub_device_id,
        queue_id=queue_id,
    )


def _make_binary_work(
    device,
    config: OverlapConfig,
    name: str,
    role: str,
    shape: tuple[int, int],
    op_name: str,
    op_fn: Callable,
    sub_device_id: ttnn.SubDeviceId,
    queue_id: int,
    seed: int,
) -> BinaryWork:
    tt_a, tt_b = _make_binary_inputs(device, config, shape, seed)
    return BinaryWork(
        name=name,
        role=role,
        shape=shape,
        op_name=op_name,
        op_fn=op_fn,
        tt_a=tt_a,
        tt_b=tt_b,
        sub_device_id=sub_device_id,
        queue_id=queue_id,
    )


def _deallocate_all(tensors: Iterable[ttnn.Tensor]) -> None:
    for tensor in tensors:
        try:
            ttnn.deallocate(tensor)
        except Exception:
            try:
                tensor.deallocate()
            except Exception:
                pass


def _deallocate_work(work_items: Iterable[Work]) -> None:
    for work in work_items:
        _deallocate_all([work.tt_a, work.tt_b])


def _run_work(work: Work, config: OverlapConfig, compute_kernel_config, use_subdevice: bool):
    if isinstance(work, MatmulWork):
        program_config = work.sub_program_config if use_subdevice else work.full_program_config
        kwargs = {
            "queue_id": work.queue_id if use_subdevice else 0,
            "program_config": program_config,
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "dtype": config.dtype,
            "compute_kernel_config": compute_kernel_config,
        }
        if use_subdevice:
            kwargs["sub_device_id"] = work.sub_device_id
        return ttnn.matmul(work.tt_a, work.tt_b, **kwargs)

    kwargs = {
        "queue_id": work.queue_id if use_subdevice else 0,
        "memory_config": ttnn.DRAM_MEMORY_CONFIG,
    }
    if use_subdevice:
        kwargs["sub_device_id"] = work.sub_device_id
    return work.op_fn(work.tt_a, work.tt_b, **kwargs)


def _run_full_seq_once(device, work_items: list[Work], config: OverlapConfig, compute_kernel_config) -> float:
    ttnn.synchronize_device(device)
    outputs = []
    start_s = time.perf_counter()
    for _ in range(config.groups_per_iteration):
        for work in work_items:
            outputs.append(_run_work(work, config, compute_kernel_config, use_subdevice=False))
    ttnn.synchronize_device(device)
    elapsed_s = time.perf_counter() - start_s
    _deallocate_all(outputs)
    return elapsed_s


def _run_sub_seq_once(device, work_items: list[Work], config: OverlapConfig, compute_kernel_config) -> float:
    ttnn.synchronize_device(device)
    outputs = []
    start_s = time.perf_counter()
    for _ in range(config.groups_per_iteration):
        for work in work_items:
            device.set_sub_device_stall_group([work.sub_device_id])
            output = _run_work(work, config, compute_kernel_config, use_subdevice=True)
            ttnn.synchronize_device(device)
            outputs.append(output)
    device.reset_sub_device_stall_group()
    elapsed_s = time.perf_counter() - start_s
    _deallocate_all(outputs)
    return elapsed_s


def _run_overlap_once(device, work_items: list[Work], config: OverlapConfig, compute_kernel_config) -> float:
    ttnn.synchronize_device(device)
    outputs = []
    start_s = time.perf_counter()
    for _ in range(config.groups_per_iteration):
        for work in work_items:
            device.set_sub_device_stall_group([work.sub_device_id])
            outputs.append(_run_work(work, config, compute_kernel_config, use_subdevice=True))
    device.reset_sub_device_stall_group()
    ttnn.synchronize_device(device)
    elapsed_s = time.perf_counter() - start_s
    _deallocate_all(outputs)
    return elapsed_s


def _measure_scenario(
    device,
    config: OverlapConfig,
    scenario_name: str,
    work_items: list[Work],
    sub_core_ranges: list[ttnn.CoreRangeSet],
    notes: str,
) -> dict:
    compute_kernel_config = _compute_kernel_config(config)
    sub_devices = [ttnn.SubDevice([core_range]) for core_range in sub_core_ranges]
    manager = device.create_sub_device_manager(sub_devices, 0)
    full_seq_s = []
    sub_seq_s = []
    overlap_s = []

    total_flops = float(config.groups_per_iteration) * sum(work.flops for work in work_items)
    op_shapes = ",".join(
        f"{work.name}:{'x'.join(str(dim) for dim in (work.shape if isinstance(work, MatmulWork) else work.shape))}"
        for work in work_items
    )
    roles = ",".join(f"{work.name}={work.role}" for work in work_items)
    queues = ",".join(f"{work.name}:cq{work.queue_id}" for work in work_items)

    try:
        for _ in range(config.warmup_iterations):
            _run_full_seq_once(device, work_items, config, compute_kernel_config)

        for _ in range(config.measured_iterations):
            full_seq_s.append(_run_full_seq_once(device, work_items, config, compute_kernel_config))

        device.load_sub_device_manager(manager)
        try:
            for _ in range(config.warmup_iterations):
                _run_sub_seq_once(device, work_items, config, compute_kernel_config)
                _run_overlap_once(device, work_items, config, compute_kernel_config)

            for _ in range(config.measured_iterations):
                sub_seq_s.append(_run_sub_seq_once(device, work_items, config, compute_kernel_config))
                overlap_s.append(_run_overlap_once(device, work_items, config, compute_kernel_config))
        finally:
            device.reset_sub_device_stall_group()
            device.clear_loaded_sub_device_manager()
    finally:
        device.remove_sub_device_manager(manager)
        _deallocate_work(work_items)

    full_avg_s = statistics.mean(full_seq_s)
    sub_avg_s = statistics.mean(sub_seq_s)
    overlap_avg_s = statistics.mean(overlap_s)
    return {
        "scenario": scenario_name,
        "status": "ran",
        "dtype": config.dtype_name,
        "fidelity": config.math_fidelity_name,
        "fp32_acc": int(config.fp32_dest_acc_en),
        "packer_l1_acc": int(config.packer_l1_acc),
        "groups_per_iter": config.groups_per_iteration,
        "ops": op_shapes,
        "roles": roles,
        "queues": queues,
        "full_seq_ms": full_avg_s * 1000.0,
        "sub_seq_ms": sub_avg_s * 1000.0,
        "overlap_ms": overlap_avg_s * 1000.0,
        "speedup_vs_sub_seq": sub_avg_s / max(overlap_avg_s, 1e-12),
        "speedup_vs_full_seq": full_avg_s / max(overlap_avg_s, 1e-12),
        "full_seq_tflops": total_flops / max(full_avg_s, 1e-12) / 1e12,
        "sub_seq_tflops": total_flops / max(sub_avg_s, 1e-12) / 1e12,
        "overlap_tflops": total_flops / max(overlap_avg_s, 1e-12) / 1e12,
        "total_flops_t": total_flops / 1e12,
        "notes": notes,
    }


def _skipped_row(config: OverlapConfig, scenario_name: str, notes: str) -> dict:
    return {
        "scenario": scenario_name,
        "status": "skipped",
        "dtype": config.dtype_name,
        "fidelity": config.math_fidelity_name,
        "fp32_acc": int(config.fp32_dest_acc_en),
        "packer_l1_acc": int(config.packer_l1_acc),
        "groups_per_iter": config.groups_per_iteration,
        "ops": "",
        "roles": "",
        "queues": "",
        "full_seq_ms": 0.0,
        "sub_seq_ms": 0.0,
        "overlap_ms": 0.0,
        "speedup_vs_sub_seq": 0.0,
        "speedup_vs_full_seq": 0.0,
        "full_seq_tflops": 0.0,
        "sub_seq_tflops": 0.0,
        "overlap_tflops": 0.0,
        "total_flops_t": 0.0,
        "notes": notes,
    }


def _build_decode_prefetch_compute(device, config: OverlapConfig, cols: int, rows: int) -> tuple[list[Work], list[ttnn.CoreRangeSet], str]:
    if not (0 < config.prefetch_rows < rows):
        raise ValueError(f"SUBDEVICE_OVERLAP_PREFETCH_ROWS must be in (0, {rows}), got {config.prefetch_rows}")
    sub_core_ranges = _row_ranges_from_heights(cols, rows, [config.prefetch_rows, rows - config.prefetch_rows])
    full_core_range = _core_range(0, 0, cols - 1, rows - 1)
    work_items = [
        _make_matmul_work(
            device,
            config,
            "prefetch_like",
            "decode_prefetcher_like_dram_to_l1_work",
            config.decode_prefetch_shape,
            cols,
            rows,
            full_core_range,
            sub_core_ranges[0],
            config.prefetch_rows,
            ttnn.SubDeviceId(0),
            0,
            0,
        ),
        _make_matmul_work(
            device,
            config,
            "decode_compute",
            "decode_compute",
            config.decode_compute_shape,
            cols,
            rows,
            full_core_range,
            sub_core_ranges[1],
            rows - config.prefetch_rows,
            ttnn.SubDeviceId(1),
            min(1, config.num_command_queues - 1),
            1,
        ),
    ]
    notes = (
        "single-card approximation of decode prefetcher+compute; real model path uses "
        "ttnn.dram_prefetcher/global_cb and worker sub_device_id"
    )
    return work_items, sub_core_ranges, notes


def _build_moe_dispatch_shared(device, config: OverlapConfig, cols: int, rows: int) -> tuple[list[Work], list[ttnn.CoreRangeSet], str]:
    if not (0 < config.dispatch_rows < rows):
        raise ValueError(f"SUBDEVICE_OVERLAP_DISPATCH_ROWS must be in (0, {rows}), got {config.dispatch_rows}")
    sub_core_ranges = _row_ranges_from_heights(cols, rows, [config.dispatch_rows, rows - config.dispatch_rows])
    full_core_range = _core_range(0, 0, cols - 1, rows - 1)
    work_items = [
        _make_matmul_work(
            device,
            config,
            "dispatch_like",
            "moe_dispatch_like_routing_metadata_or_pack",
            config.moe_dispatch_shape,
            cols,
            rows,
            full_core_range,
            sub_core_ranges[0],
            config.dispatch_rows,
            ttnn.SubDeviceId(0),
            0,
            2,
        ),
        _make_matmul_work(
            device,
            config,
            "shared_expert_like",
            "shared_expert_ffn_like_compute",
            config.moe_shared_shape,
            cols,
            rows,
            full_core_range,
            sub_core_ranges[1],
            rows - config.dispatch_rows,
            ttnn.SubDeviceId(1),
            min(1, config.num_command_queues - 1),
            3,
        ),
    ]
    notes = (
        "mirrors DeepSeek DP prefill subdevice split: dispatch strip plus shared-expert strip; "
        "dispatch op is a tunable matmul proxy, not deepseek_prefill.dispatch"
    )
    return work_items, sub_core_ranges, notes


def _build_independent_small_medium(device, config: OverlapConfig, cols: int, rows: int) -> tuple[list[Work], list[ttnn.CoreRangeSet], str]:
    num_subdevices = config.independent_subdevices
    if num_subdevices < 2:
        raise ValueError("SUBDEVICE_OVERLAP_INDEPENDENT_SUBDEVICES must be >= 2")
    sub_core_ranges = _even_row_ranges(cols, rows, num_subdevices)
    full_core_range = _core_range(0, 0, cols - 1, rows - 1)
    work_items: list[Work] = []
    shapes = list(config.independent_shapes)
    for index in range(num_subdevices):
        shape = shapes[index % len(shapes)]
        work_items.append(
            _make_matmul_work(
                device,
                config,
                f"mm{index}",
                "independent_small_medium_matmul",
                shape,
                cols,
                rows,
                full_core_range,
                sub_core_ranges[index],
                rows // num_subdevices,
                ttnn.SubDeviceId(index),
                index % config.num_command_queues,
                100 + index,
            )
        )

    # Add one binary op when there are enough sub-devices.  This keeps the
    # scenario from being only GEMMs while preserving a simple full/sub compare.
    if num_subdevices >= 3:
        binary_index = num_subdevices - 1
        work_items[binary_index] = _make_binary_work(
            device,
            config,
            f"mul{binary_index}",
            "independent_small_binary_multiply",
            (512, 512),
            "multiply",
            ttnn.multiply,
            ttnn.SubDeviceId(binary_index),
            binary_index % config.num_command_queues,
            200 + binary_index,
        )

    notes = "independent work items queued to disjoint sub-devices; queue IDs are round-robin over available HW CQs"
    return work_items, sub_core_ranges, notes


@dataclass
class ExpertFfnCppWork:
    inputs: list[ttnn.Tensor]
    gate_projs: list[ttnn.Tensor]
    up_projs: list[ttnn.Tensor]
    down_projs: list[ttnn.Tensor]

    @property
    def tensors(self) -> list[ttnn.Tensor]:
        return [*self.inputs, *self.gate_projs, *self.up_projs, *self.down_projs]


def _expert_ffn_cpp_ops_available() -> tuple[bool, str]:
    deepseek = getattr(getattr(ttnn, "experimental", object()), "deepseek_prefill", None)
    if deepseek is None:
        return False, "ttnn.experimental.deepseek_prefill is unavailable"
    required = ("routed_expert_ffn_opt",)
    missing = [name for name in required if getattr(deepseek, name, None) is None]
    if missing:
        return False, f"missing ops={','.join(missing)}"
    return True, ""


def _routed_expert_ffn_opt_supports_subdevice() -> bool:
    fused_op = getattr(ttnn.experimental.deepseek_prefill, "routed_expert_ffn_opt", None)
    return fused_op is not None and "sub_device_id" in str(getattr(fused_op, "__doc__", ""))


def _allocate_device_tensor(device, shape: tuple[int, ...], dtype: ttnn.DataType) -> ttnn.Tensor:
    return ttnn.empty(
        ttnn.Shape(shape),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _make_expert_ffn_cpp_work(device, config: OverlapConfig) -> ExpertFfnCppWork:
    rows, emb_dim, hidden_dim = config.expert_ffn_shape
    inputs = []
    gate_projs = []
    up_projs = []
    down_projs = []
    for _ in range(config.expert_ffn_num_experts):
        inputs.append(_allocate_device_tensor(device, (rows, emb_dim), config.expert_ffn_activation_dtype))
        gate_projs.append(_allocate_device_tensor(device, (emb_dim, hidden_dim), config.expert_ffn_weight_dtype))
        up_projs.append(_allocate_device_tensor(device, (emb_dim, hidden_dim), config.expert_ffn_weight_dtype))
        down_projs.append(_allocate_device_tensor(device, (hidden_dim, emb_dim), config.expert_ffn_weight_dtype))
    return ExpertFfnCppWork(inputs=inputs, gate_projs=gate_projs, up_projs=up_projs, down_projs=down_projs)


def _expert_ffn_subdevice_ranges(cols: int, rows: int, num_experts: int) -> tuple[list[ttnn.CoreRangeSet], int]:
    if num_experts < 1:
        raise ValueError("SUBDEVICE_OVERLAP_EXPERT_FFN_EXPERTS must be >= 1")
    if num_experts > rows:
        raise ValueError(
            f"SUBDEVICE_OVERLAP_EXPERT_FFN_EXPERTS={num_experts} exceeds available compute rows={rows}"
        )
    if rows % num_experts != 0:
        raise ValueError(
            f"SUBDEVICE_OVERLAP_EXPERT_FFN_EXPERTS={num_experts} must evenly divide compute rows={rows} "
            "so every expert gets the same sub-device height"
        )
    core_rows = rows // num_experts
    return _row_ranges_from_heights(cols, rows, [core_rows] * num_experts), core_rows


def _sync_device(device, sub_device_ids: list[ttnn.SubDeviceId] | None = None) -> None:
    if not sub_device_ids:
        ttnn.synchronize_device(device)
        return
    try:
        ttnn.synchronize_device(device, sub_device_ids=sub_device_ids)
    except TypeError:
        ttnn.synchronize_device(device)


def _run_expert_ffn_stage_ops_once(
    device,
    config: OverlapConfig,
    work: ExpertFfnCppWork,
    compute_kernel_config,
) -> float:
    deepseek = ttnn.experimental.deepseek_prefill
    ttnn.synchronize_device(device)
    outputs = []
    start_s = time.perf_counter()
    for _ in range(config.groups_per_iteration):
        for expert_idx in range(config.expert_ffn_num_experts):
            gate_out = deepseek.routed_expert_ffn_opt_gate(
                work.inputs[expert_idx],
                work.gate_projs[expert_idx],
                compute_kernel_config=compute_kernel_config,
            )
            up_out = deepseek.routed_expert_ffn_opt_up(
                work.inputs[expert_idx],
                work.up_projs[expert_idx],
                compute_kernel_config=compute_kernel_config,
            )
            activated = deepseek.routed_expert_ffn_opt_mul_reshard(gate_out, up_out)
            outputs.append(
                deepseek.routed_expert_ffn_opt_down(
                    activated,
                    work.down_projs[expert_idx],
                    compute_kernel_config=compute_kernel_config,
                )
            )
            ttnn.deallocate(activated)
    ttnn.synchronize_device(device)
    elapsed_s = time.perf_counter() - start_s
    _deallocate_all(outputs)
    return elapsed_s


def _run_expert_ffn_full_serial_once(
    device,
    config: OverlapConfig,
    work: ExpertFfnCppWork,
    compute_kernel_config,
) -> float:
    fused_op = ttnn.experimental.deepseek_prefill.routed_expert_ffn_opt
    ttnn.synchronize_device(device)
    outputs = []
    start_s = time.perf_counter()
    for _ in range(config.groups_per_iteration):
        for expert_idx in range(config.expert_ffn_num_experts):
            outputs.append(
                fused_op(
                    work.inputs[expert_idx],
                    work.gate_projs[expert_idx],
                    work.up_projs[expert_idx],
                    work.down_projs[expert_idx],
                    compute_kernel_config=compute_kernel_config,
                )
            )
    ttnn.synchronize_device(device)
    elapsed_s = time.perf_counter() - start_s
    _deallocate_all(outputs)
    return elapsed_s


def _run_expert_ffn_subdevice_serial_once(
    device,
    config: OverlapConfig,
    work: ExpertFfnCppWork,
    compute_kernel_config,
) -> float:
    fused_op = ttnn.experimental.deepseek_prefill.routed_expert_ffn_opt
    _sync_device(device)
    outputs = []
    start_s = time.perf_counter()
    try:
        for _ in range(config.groups_per_iteration):
            for expert_idx in range(config.expert_ffn_num_experts):
                sub_device_id = ttnn.SubDeviceId(expert_idx)
                device.set_sub_device_stall_group([sub_device_id])
                outputs.append(
                    fused_op(
                        work.inputs[expert_idx],
                        work.gate_projs[expert_idx],
                        work.up_projs[expert_idx],
                        work.down_projs[expert_idx],
                        compute_kernel_config=compute_kernel_config,
                        sub_device_id=sub_device_id,
                    )
                )
                device.reset_sub_device_stall_group()
                _sync_device(device, [sub_device_id])
    finally:
        device.reset_sub_device_stall_group()
    elapsed_s = time.perf_counter() - start_s
    _deallocate_all(outputs)
    return elapsed_s


def _run_expert_ffn_subdevice_parallel_once(
    device,
    config: OverlapConfig,
    work: ExpertFfnCppWork,
    compute_kernel_config,
) -> float:
    fused_op = ttnn.experimental.deepseek_prefill.routed_expert_ffn_opt
    sub_device_ids = [ttnn.SubDeviceId(expert_idx) for expert_idx in range(config.expert_ffn_num_experts)]
    _sync_device(device)
    outputs = []
    start_s = time.perf_counter()
    try:
        for _ in range(config.groups_per_iteration):
            device.set_sub_device_stall_group(sub_device_ids)
            for expert_idx, sub_device_id in enumerate(sub_device_ids):
                outputs.append(
                    fused_op(
                        work.inputs[expert_idx],
                        work.gate_projs[expert_idx],
                        work.up_projs[expert_idx],
                        work.down_projs[expert_idx],
                        compute_kernel_config=compute_kernel_config,
                        sub_device_id=sub_device_id,
                    )
                )
            device.reset_sub_device_stall_group()
        _sync_device(device, sub_device_ids)
    finally:
        device.reset_sub_device_stall_group()
    elapsed_s = time.perf_counter() - start_s
    _deallocate_all(outputs)
    return elapsed_s


def _measure_expert_ffn_cpp_fused_scenario(device, config: OverlapConfig) -> dict:
    available, reason = _expert_ffn_cpp_ops_available()
    if not available:
        return _skipped_row(config, "expert_ffn_cpp_fused", reason)
    if not _routed_expert_ffn_opt_supports_subdevice():
        return _skipped_row(
            config,
            "expert_ffn_cpp_fused",
            "routed_expert_ffn_opt binding does not expose sub_device_id",
        )
    if config.expert_ffn_activation_dtype not in {ttnn.bfloat16, ttnn.bfloat8_b}:
        return _skipped_row(
            config,
            "expert_ffn_cpp_fused",
            f"routed_expert_ffn_opt input must be BFLOAT16 or BFLOAT8_B, got {config.expert_ffn_activation_dtype_name}",
        )

    rows, emb_dim, hidden_dim = config.expert_ffn_shape
    grid = device.compute_with_storage_grid_size()
    cols, device_rows = int(grid.x), int(grid.y)
    try:
        sub_core_ranges, sub_core_rows = _expert_ffn_subdevice_ranges(
            cols, device_rows, config.expert_ffn_num_experts
        )
    except ValueError as error:
        return _skipped_row(config, "expert_ffn_cpp_fused", str(error))

    compute_kernel_config = _compute_kernel_config(config)
    work = _make_expert_ffn_cpp_work(device, config)
    sub_devices = [ttnn.SubDevice([core_range]) for core_range in sub_core_ranges]
    manager = device.create_sub_device_manager(sub_devices, 0)
    full_serial_s: list[float] = []
    sub_serial_s: list[float] = []
    sub_parallel_s: list[float] = []
    try:
        for _ in range(config.warmup_iterations):
            _run_expert_ffn_full_serial_once(device, config, work, compute_kernel_config)

        for _ in range(config.measured_iterations):
            full_serial_s.append(_run_expert_ffn_full_serial_once(device, config, work, compute_kernel_config))

        device.load_sub_device_manager(manager)
        try:
            for _ in range(config.warmup_iterations):
                _run_expert_ffn_subdevice_serial_once(device, config, work, compute_kernel_config)
                _run_expert_ffn_subdevice_parallel_once(device, config, work, compute_kernel_config)

            for _ in range(config.measured_iterations):
                sub_serial_s.append(_run_expert_ffn_subdevice_serial_once(device, config, work, compute_kernel_config))
                sub_parallel_s.append(_run_expert_ffn_subdevice_parallel_once(device, config, work, compute_kernel_config))
        finally:
            device.reset_sub_device_stall_group()
            device.clear_loaded_sub_device_manager()
    finally:
        device.remove_sub_device_manager(manager)
        _deallocate_all(work.tensors)

    full_serial_avg_s = statistics.mean(full_serial_s)
    sub_serial_avg_s = statistics.mean(sub_serial_s)
    sub_parallel_avg_s = statistics.mean(sub_parallel_s)
    total_flops = (
        float(config.groups_per_iteration)
        * float(config.expert_ffn_num_experts)
        * 6.0
        * float(rows)
        * float(emb_dim)
        * float(hidden_dim)
    )
    return {
        "scenario": "expert_ffn_cpp_fused",
        "status": "ran",
        "dtype": f"{config.expert_ffn_activation_dtype_name}/{config.expert_ffn_weight_dtype_name}",
        "fidelity": config.math_fidelity_name,
        "fp32_acc": int(config.fp32_dest_acc_en),
        "packer_l1_acc": int(config.packer_l1_acc),
        "groups_per_iter": config.groups_per_iteration,
        "ops": f"experts={config.expert_ffn_num_experts}:each={rows}x{emb_dim}x{hidden_dim}",
        "roles": (
            "full=full-grid serial routed_expert_ffn_opt,"
            f"sub=serial routed_expert_ffn_opt on {cols}x{sub_core_rows} subdevices,"
            "overlap=parallel expert FFNs on disjoint subdevices"
        ),
        "queues": "routed_expert_ffn_opt has no queue_id kwarg; parallelism is via sub_device_id/stall_group",
        "full_seq_ms": full_serial_avg_s * 1000.0,
        "sub_seq_ms": sub_serial_avg_s * 1000.0,
        "overlap_ms": sub_parallel_avg_s * 1000.0,
        "speedup_vs_sub_seq": sub_serial_avg_s / max(sub_parallel_avg_s, 1e-12),
        "speedup_vs_full_seq": full_serial_avg_s / max(sub_parallel_avg_s, 1e-12),
        "full_seq_tflops": total_flops / max(full_serial_avg_s, 1e-12) / 1e12,
        "sub_seq_tflops": total_flops / max(sub_serial_avg_s, 1e-12) / 1e12,
        "overlap_tflops": total_flops / max(sub_parallel_avg_s, 1e-12) / 1e12,
        "total_flops_t": total_flops / 1e12,
        "notes": (
            "Equal-size expert FFN experiment using routed_expert_ffn_opt. "
            f"Each expert gets {rows} rows and a {cols}x{sub_core_rows} sub-device. "
            "This exercises sub-device expert parallelism, but it is still one Python-visible fused C++ op per expert, "
            "not a new C++ op that accepts a list of experts."
        ),
    }


def _write_csv(config: OverlapConfig, rows: list[dict]) -> None:
    config.csv_path.parent.mkdir(parents=True, exist_ok=True)
    with config.csv_path.open("w", newline="") as csv_file:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    header_line = " | ".join(header.ljust(width) for header, width in zip(headers, widths))
    divider = "-+-".join("-" * width for width in widths)
    body = [" | ".join(value.ljust(width) for value, width in zip(row, widths)) for row in rows]
    return [header_line, divider, *body]


def _print_summary(config: OverlapConfig, results: list[dict], device_grid: str) -> None:
    def fmt_ms(result: dict, key: str) -> str:
        value = result.get(key)
        if result["status"] != "ran" or value is None:
            return "-"
        return f"{value:.3f}"

    def fmt_ratio(result: dict, key: str) -> str:
        value = result.get(key)
        if result["status"] != "ran" or value is None:
            return "-"
        return f"{value:.3f}"

    headers = [
        "scenario",
        "status",
        "full_ms",
        "sub_ms",
        "overlap_ms",
        "speedup_sub",
        "speedup_full",
        "overlap_TF/s",
        "ops",
    ]
    table_rows = []
    for result in results:
        table_rows.append(
            [
                result["scenario"],
                result["status"],
                fmt_ms(result, "full_seq_ms"),
                fmt_ms(result, "sub_seq_ms"),
                fmt_ms(result, "overlap_ms"),
                fmt_ratio(result, "speedup_vs_sub_seq"),
                fmt_ratio(result, "speedup_vs_full_seq"),
                fmt_ratio(result, "overlap_tflops"),
                result["ops"][:64],
            ]
        )

    detail_lines = []
    for result in results:
        detail_lines.extend(
            [
                f"{result['scenario']}:",
                f"  roles={result['roles'] or '-'}",
                f"  queues={result['queues'] or '-'}",
                f"  notes={result['notes']}",
            ]
        )

    lines = [
        "",
        "=" * 120,
        "SUBDEVICE OPT OVERLAP SCENARIOS RESULT",
        "-" * 120,
        (
            f"device_grid={device_grid} | dtype={config.dtype_name} | fidelity={config.math_fidelity_name} | "
            f"fp32_acc={int(config.fp32_dest_acc_en)} | packer_l1_acc={int(config.packer_l1_acc)}"
        ),
        (
            f"cqs={config.num_command_queues} | warmup={config.warmup_iterations} | "
            f"iters={config.measured_iterations} | groups_per_iter={config.groups_per_iteration}"
        ),
        "-" * 120,
        *_format_table(headers, table_rows),
        "-" * 120,
        *detail_lines,
        "-" * 120,
        f"csv={config.csv_path}",
        "=" * 120,
    ]
    print("\n".join(lines), flush=True)


@pytest.mark.timeout(CONFIG.timeout_s)
@pytest.mark.parametrize("device_params", [{"num_command_queues": CONFIG.num_command_queues}], indirect=True)
def test_subdevice_overlap_scenarios(device, device_params):
    grid = device.compute_with_storage_grid_size()
    cols, rows = int(grid.x), int(grid.y)
    if rows < 2:
        pytest.skip(f"Need at least 2 compute rows for sub-device overlap experiments, got rows={rows}")

    builders = {
        "decode_prefetch_compute": _build_decode_prefetch_compute,
        "moe_dispatch_shared": _build_moe_dispatch_shared,
        "independent_small_medium": _build_independent_small_medium,
    }
    results = []

    for scenario_name in CONFIG.scenarios:
        if scenario_name == "expert_ffn_cpp_fused":
            try:
                results.append(_measure_expert_ffn_cpp_fused_scenario(device, CONFIG))
            except ValueError as error:
                results.append(_skipped_row(CONFIG, scenario_name, str(error)))
            continue
        if scenario_name == "ccl_compute":
            results.append(
                _skipped_row(
                    CONFIG,
                    scenario_name,
                    "requires multi-device mesh/fabric; use ttnn.experimental.all_gather_async or "
                    "reduce_scatter_minimal_async with a CCL subdevice and compute on another subdevice",
                )
            )
            continue
        if scenario_name not in builders:
            results.append(_skipped_row(CONFIG, scenario_name, f"unknown scenario={scenario_name!r}"))
            continue
        try:
            work_items, sub_core_ranges, notes = builders[scenario_name](device, CONFIG, cols, rows)
            results.append(_measure_scenario(device, CONFIG, scenario_name, work_items, sub_core_ranges, notes))
        except ValueError as error:
            results.append(_skipped_row(CONFIG, scenario_name, str(error)))

    _write_csv(CONFIG, results)
    _print_summary(CONFIG, results, f"{cols}x{rows}")
