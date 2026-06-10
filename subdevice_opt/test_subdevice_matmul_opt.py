# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Sub-device matmul perf harness.

Compares pairs of equal-shape 2D-mcast matmuls run as:
  1. full-grid sequential: N matmuls on the full core grid in queue order
  2. split sub-device parallel: N matmuls on N row-split sub-devices

Useful overrides:
    SUBDEVICE_OPT_M=2048
    SUBDEVICE_OPT_N=2048
    SUBDEVICE_OPT_K=2048
    SUBDEVICE_OPT_SHAPES=160x8192x8192,64x8192x8192
    SUBDEVICE_OPT_DTYPE=bf16
    SUBDEVICE_OPT_DTYPES=bf16,bf8
    SUBDEVICE_OPT_FIDELITY=HiFi2
    SUBDEVICE_OPT_FIDELITIES=LoFi,HiFi2
    SUBDEVICE_OPT_FP32_ACC=0
    SUBDEVICE_OPT_PACKER_L1_ACC=1
    SUBDEVICE_OPT_MATH_APPROX=0
    SUBDEVICE_OPT_WARMUP=1
    SUBDEVICE_OPT_ITERS=5
    SUBDEVICE_OPT_PAIRS_PER_ITER=1
    SUBDEVICE_OPT_NUM_CQS=2
    SUBDEVICE_OPT_NUM_SUBDEVICES=2  # max 8 in the current dispatch table
    SUBDEVICE_OPT_IN0_BLOCK_W_MAX=16
    SUBDEVICE_OPT_L1_TILE_BUDGET=1280
    SUBDEVICE_OPT_OUT_BLOCK_TILE_BUDGET=256
    SUBDEVICE_OPT_CSV=/home/tenstorrent/tt-metal/subdevice_opt/subdevice_matmul_opt.csv
"""

from __future__ import annotations

import csv
import os
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import pytest
import torch

import ttnn


_TILE = 32
_MAX_SUBDEVICES = 8
_MAX_COMMAND_QUEUES = 2
_DEFAULT_CSV = Path(__file__).resolve().with_name("subdevice_matmul_opt.csv")


@dataclass(frozen=True)
class SubdeviceMatmulConfig:
    m: int
    n: int
    k: int
    dtype: ttnn.DataType
    dtype_name: str
    math_fidelity: object
    math_fidelity_name: str
    fp32_dest_acc_en: bool
    packer_l1_acc: bool
    math_approx_mode: bool
    warmup_iterations: int
    measured_iterations: int
    pairs_per_iteration: int
    num_command_queues: int
    num_subdevices: int
    in0_block_w_max: int
    l1_tile_budget: int
    out_block_tile_budget: int
    csv_path: Path
    timeout_s: int


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def _split_env_list(name: str) -> list[str]:
    value = os.getenv(name)
    if value is None:
        return []
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
        raise ValueError(
            f"Unsupported SUBDEVICE_OPT_SHAPES entry={value!r}; use MxNxK, for example 160x8192x8192"
        )
    return int(parts[0]), int(parts[1]), int(parts[2])


def _parse_dtype(value: str) -> tuple[ttnn.DataType, str]:
    normalized = value.lower()
    if normalized in {"bf16", "bfloat16"}:
        return ttnn.bfloat16, "BFLOAT16"
    if normalized in {"bf8", "bfloat8", "bfloat8_b"}:
        return ttnn.bfloat8_b, "BFLOAT8_B"
    if normalized in {"bf4", "bfloat4", "bfloat4_b"}:
        return ttnn.bfloat4_b, "BFLOAT4_B"
    raise ValueError(f"Unsupported SUBDEVICE_OPT_DTYPE={value!r}; use bf16, bf8, or bf4")


def _parse_math_fidelity(value: str):
    normalized = value.lower().replace("_", "")
    mapping = {
        "lofi": (ttnn.MathFidelity.LoFi, "LoFi"),
        "hifi2": (ttnn.MathFidelity.HiFi2, "HiFi2"),
        "hifi3": (getattr(ttnn.MathFidelity, "HiFi3", ttnn.MathFidelity.HiFi2), "HiFi3"),
        "hifi4": (ttnn.MathFidelity.HiFi4, "HiFi4"),
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported SUBDEVICE_OPT_FIDELITY={value!r}; use LoFi, HiFi2, HiFi3, or HiFi4")
    return mapping[normalized]


def _load_config() -> SubdeviceMatmulConfig:
    dtype, dtype_name = _parse_dtype(os.getenv("SUBDEVICE_OPT_DTYPE", "bf16"))
    math_fidelity, math_fidelity_name = _parse_math_fidelity(os.getenv("SUBDEVICE_OPT_FIDELITY", "HiFi4"))
    config = SubdeviceMatmulConfig(
        m=_env_int("SUBDEVICE_OPT_M", 2048),
        n=_env_int("SUBDEVICE_OPT_N", 2048),
        k=_env_int("SUBDEVICE_OPT_K", 2048),
        dtype=dtype,
        dtype_name=dtype_name,
        math_fidelity=math_fidelity,
        math_fidelity_name=math_fidelity_name,
        fp32_dest_acc_en=_env_flag("SUBDEVICE_OPT_FP32_ACC", True),
        packer_l1_acc=_env_flag("SUBDEVICE_OPT_PACKER_L1_ACC", False),
        math_approx_mode=_env_flag("SUBDEVICE_OPT_MATH_APPROX", False),
        warmup_iterations=_env_int("SUBDEVICE_OPT_WARMUP", 10),
        measured_iterations=_env_int("SUBDEVICE_OPT_ITERS", 20),
        pairs_per_iteration=_env_int("SUBDEVICE_OPT_PAIRS_PER_ITER", 1),
        num_command_queues=_env_int("SUBDEVICE_OPT_NUM_CQS", 2),
        num_subdevices=_env_int("SUBDEVICE_OPT_NUM_SUBDEVICES", 2),
        in0_block_w_max=_env_int("SUBDEVICE_OPT_IN0_BLOCK_W_MAX", 16),
        l1_tile_budget=_env_int("SUBDEVICE_OPT_L1_TILE_BUDGET", 1280),
        out_block_tile_budget=_env_int("SUBDEVICE_OPT_OUT_BLOCK_TILE_BUDGET", 256),
        csv_path=Path(os.getenv("SUBDEVICE_OPT_CSV", str(_DEFAULT_CSV))),
        timeout_s=_env_int("SUBDEVICE_OPT_TIMEOUT", 300),
    )
    if config.num_command_queues > _MAX_COMMAND_QUEUES:
        raise ValueError(f"SUBDEVICE_OPT_NUM_CQS must be <= {_MAX_COMMAND_QUEUES}")
    return config


def _load_configs() -> list[SubdeviceMatmulConfig]:
    base_config = _load_config()
    shape_values = _split_env_list("SUBDEVICE_OPT_SHAPES")
    dtype_values = _split_env_list("SUBDEVICE_OPT_DTYPES")
    fidelity_values = _split_env_list("SUBDEVICE_OPT_FIDELITIES")

    shapes = [_parse_shape(value) for value in shape_values] or [(base_config.m, base_config.n, base_config.k)]
    dtypes = [_parse_dtype(value) for value in dtype_values] or [(base_config.dtype, base_config.dtype_name)]
    fidelities = [_parse_math_fidelity(value) for value in fidelity_values] or [
        (base_config.math_fidelity, base_config.math_fidelity_name)
    ]

    configs = []
    for m, n, k in shapes:
        for dtype, dtype_name in dtypes:
            for math_fidelity, math_fidelity_name in fidelities:
                configs.append(
                    replace(
                        base_config,
                        m=m,
                        n=n,
                        k=k,
                        dtype=dtype,
                        dtype_name=dtype_name,
                        math_fidelity=math_fidelity,
                        math_fidelity_name=math_fidelity_name,
                    )
                )
    return configs


CONFIGS = _load_configs()
CONFIG = CONFIGS[0]


def _validate_config(config: SubdeviceMatmulConfig) -> None:
    for name, value in (("M", config.m), ("N", config.n), ("K", config.k)):
        if value % _TILE != 0:
            raise ValueError(f"SUBDEVICE_OPT_{name} must be divisible by {_TILE}, got {value}")
    if config.measured_iterations < 1:
        raise ValueError("SUBDEVICE_OPT_ITERS must be >= 1")
    if config.warmup_iterations < 0:
        raise ValueError("SUBDEVICE_OPT_WARMUP must be >= 0")
    if config.pairs_per_iteration < 1:
        raise ValueError("SUBDEVICE_OPT_PAIRS_PER_ITER must be >= 1")
    if config.num_command_queues < 1:
        raise ValueError("SUBDEVICE_OPT_NUM_CQS must be >= 1")
    if config.num_command_queues > _MAX_COMMAND_QUEUES:
        raise ValueError(f"SUBDEVICE_OPT_NUM_CQS must be <= {_MAX_COMMAND_QUEUES}")
    if config.num_subdevices < 2:
        raise ValueError("SUBDEVICE_OPT_NUM_SUBDEVICES must be >= 2")
    if config.num_subdevices > _MAX_SUBDEVICES:
        raise ValueError(f"SUBDEVICE_OPT_NUM_SUBDEVICES must be <= {_MAX_SUBDEVICES}")


def _compute_kernel_config(config: SubdeviceMatmulConfig):
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


def _split_row_ranges(cols: int, rows: int, num_subdevices: int) -> list[tuple[int, int, ttnn.CoreRangeSet]]:
    if rows % num_subdevices != 0:
        raise ValueError(
            f"SUBDEVICE_OPT_NUM_SUBDEVICES={num_subdevices} must evenly divide device rows={rows}"
        )
    rows_per_subdevice = rows // num_subdevices
    ranges = []
    for index in range(num_subdevices):
        y0 = index * rows_per_subdevice
        y1 = y0 + rows_per_subdevice - 1
        ranges.append((y0, y1, _core_range(0, y0, cols - 1, y1)))
    return ranges


def _divisors_desc(value: int) -> list[int]:
    return [candidate for candidate in range(value, 0, -1) if value % candidate == 0]


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


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
    # Match tests/ttnn/unit_tests/benchmarks/test_benchmark.py::get_subblock_sizes.
    # fp32 dest accumulation has four destination registers; otherwise 8-tile
    # subblocks are valid candidates.
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
    config: SubdeviceMatmulConfig,
    grid_x: int,
    grid_y: int,
    allowed_worker_cores: ttnn.CoreRangeSet | None = None,
) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    m_tiles = config.m // _TILE
    n_tiles = config.n // _TILE
    k_tiles = config.k // _TILE
    per_core_m = _ceil_div(m_tiles, grid_y)
    per_core_n = _ceil_div(n_tiles, grid_x)
    in0_block_w = _choose_in0_block_w(
        k_tiles, config.in0_block_w_max, per_core_m, per_core_n, config.l1_tile_budget
    )
    out_block_h, out_block_w = _choose_out_block(per_core_m, per_core_n, config.out_block_tile_budget)
    out_subblock_h, out_subblock_w = _choose_out_subblock(
        out_block_h, out_block_w, config.fp32_dest_acc_en
    )
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
    if allowed_worker_cores is None:
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(**kwargs)
    try:
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            **kwargs, allowed_worker_cores=allowed_worker_cores
        )
    except TypeError as exc:
        if "allowed_worker_cores" not in str(exc):
            raise
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(**kwargs)


def _program_summary(config: SubdeviceMatmulConfig, grid_x: int, grid_y: int) -> dict[str, int]:
    m_tiles = config.m // _TILE
    n_tiles = config.n // _TILE
    k_tiles = config.k // _TILE
    per_core_m = _ceil_div(m_tiles, grid_y)
    per_core_n = _ceil_div(n_tiles, grid_x)
    out_block_h, out_block_w = _choose_out_block(per_core_m, per_core_n, config.out_block_tile_budget)
    out_subblock_h, out_subblock_w = _choose_out_subblock(
        out_block_h, out_block_w, config.fp32_dest_acc_en
    )
    return {
        "grid_x": grid_x,
        "grid_y": grid_y,
        "cores": grid_x * grid_y,
        "m_tiles": m_tiles,
        "n_tiles": n_tiles,
        "k_tiles": k_tiles,
        "per_core_m": per_core_m,
        "per_core_n": per_core_n,
        "out_block_h": out_block_h,
        "out_block_w": out_block_w,
        "out_subblock_h": out_subblock_h,
        "out_subblock_w": out_subblock_w,
        "blocks_x": _ceil_div(n_tiles, per_core_n),
        "blocks_y": _ceil_div(m_tiles, per_core_m),
        "in0_block_w": _choose_in0_block_w(
            k_tiles, config.in0_block_w_max, per_core_m, per_core_n, config.l1_tile_budget
        ),
    }


def _make_inputs(device, config: SubdeviceMatmulConfig, seed: int):
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn((1, 1, config.m, config.k), dtype=torch.bfloat16, generator=generator)
    b = torch.randn((1, 1, config.k, config.n), dtype=torch.bfloat16, generator=generator)
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


def _deallocate_all(tensors: Iterable[ttnn.Tensor]) -> None:
    for tensor in tensors:
        try:
            ttnn.deallocate(tensor)
        except Exception:
            try:
                tensor.deallocate()
            except Exception:
                pass


def _matmul(tt_a, tt_b, program_config, compute_kernel_config, dtype, sub_device_id=None, queue_id: int = 0):
    kwargs = {
        "queue_id": queue_id,
        "program_config": program_config,
        "memory_config": ttnn.DRAM_MEMORY_CONFIG,
        "dtype": dtype,
        "compute_kernel_config": compute_kernel_config,
    }
    if sub_device_id is not None:
        kwargs["sub_device_id"] = sub_device_id
    return ttnn.matmul(tt_a, tt_b, **kwargs)


def _run_full_sequential_once(device, inputs, program_config, compute_kernel_config, dtype, pairs_per_iteration: int) -> float:
    ttnn.synchronize_device(device)
    outputs = []
    start_s = time.perf_counter()
    for _ in range(pairs_per_iteration):
        for tt_a, tt_b in inputs:
            outputs.append(_matmul(tt_a, tt_b, program_config, compute_kernel_config, dtype, queue_id=0))
    ttnn.synchronize_device(device)
    elapsed_s = time.perf_counter() - start_s
    _deallocate_all(outputs)
    return elapsed_s


def _run_subdevice_parallel_once(
    device,
    inputs,
    sub_program_configs,
    compute_kernel_config,
    dtype,
    sub_device_ids: list[ttnn.SubDeviceId],
    pairs_per_iteration: int,
    queue_ids: list[int],
) -> float:
    ttnn.synchronize_device(device)
    outputs = []
    start_s = time.perf_counter()
    for _ in range(pairs_per_iteration):
        for (tt_a, tt_b), program_config, sub_device_id, queue_id in zip(
            inputs, sub_program_configs, sub_device_ids, queue_ids
        ):
            device.set_sub_device_stall_group([sub_device_id])
            outputs.append(
                _matmul(
                    tt_a,
                    tt_b,
                    program_config,
                    compute_kernel_config,
                    dtype,
                    sub_device_id=sub_device_id,
                    queue_id=queue_id,
                )
            )
    device.reset_sub_device_stall_group()
    ttnn.synchronize_device(device)
    elapsed_s = time.perf_counter() - start_s
    _deallocate_all(outputs)
    return elapsed_s


def _avg_ms(values_s: list[float]) -> float:
    return statistics.mean(values_s) * 1000.0


def _tflops(flops: float, seconds: float) -> float:
    return flops / max(seconds, 1e-12) / 1e12


def _write_csv(config: SubdeviceMatmulConfig, rows: list[dict]) -> None:
    config.csv_path.parent.mkdir(parents=True, exist_ok=True)
    with config.csv_path.open("w", newline="") as csv_file:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _compact_list(values: list[str]) -> str:
    if not values:
        return ""
    if all(value == values[0] for value in values):
        return f"{values[0]}x{len(values)}"
    return ",".join(values)


def _print_summary(result: dict) -> None:
    lines = [
        "",
        "=" * 88,
        "SUBDEVICE OPT MATMUL RESULT",
        "-" * 88,
        (
            f"shape=M{result['m']} N{result['n']} K{result['k']} | dtype={result['dtype']} | "
            f"fidelity={result['fidelity']} | fp32_acc={result['fp32_acc']} | packer_l1_acc={result['packer_l1_acc']}"
        ),
        (
            f"device_grid={result['device_grid']} | split={result['split']} | "
            f"cqs={result['num_command_queues']} | subdevice_parallel_cqs={result['subdevice_parallel_cqs']} | "
            f"subdevices={result['num_subdevices']} | matmul_groups_per_iter={result['pairs_per_iter']} | "
            f"matmuls_per_iter={result['matmuls_per_iter']} | "
            f"warmup={result['warmup']} | iters={result['iters']}"
        ),
        (
            "full_grid_config: "
            f"grid={result['full_grid']} cores={result['full_cores']} "
            f"per_core={result['full_per_core_n']}x{result['full_per_core_m']} "
            f"out_block={result['full_out_block_w']}x{result['full_out_block_h']} "
            f"out_subblock={result['full_out_subblock_w']}x{result['full_out_subblock_h']} "
            f"blocks={result['full_blocks_x']}x{result['full_blocks_y']} in0_block_w={result['full_in0_block_w']}"
        ),
        (
            "subdevice_config: "
            f"grids={result['sub_grids']} cores={result['sub_cores']} "
            f"per_core={result['sub_per_core']} out_block={result['sub_out_block']} "
            f"out_subblock={result['sub_out_subblock']} in0_block_w={result['sub_in0_block_w']}"
        ),
        "-" * 88,
        (
            f"full_grid_sequential: avg_ms={result['full_seq_avg_ms']:.3f} | "
            f"min_ms={result['full_seq_min_ms']:.3f} | max_ms={result['full_seq_max_ms']:.3f} | "
            f"TFLOPs/s={result['full_seq_tflops']:.3f}"
        ),
        (
            f"subdevice_parallel:  avg_ms={result['sub_parallel_avg_ms']:.3f} | "
            f"min_ms={result['sub_parallel_min_ms']:.3f} | max_ms={result['sub_parallel_max_ms']:.3f} | "
            f"TFLOPs/s={result['sub_parallel_tflops']:.3f}"
        ),
        (
            f"speedup(full_seq/sub_parallel)={result['speedup']:.3f} | "
            f"parallel_time/full_seq_time={result['parallel_ratio']:.3f}"
        ),
        "-" * 88,
        f"total_flops_per_iter={result['total_tflops_per_iter']:.6f} TFLOPs",
        f"csv={result['csv']}",
        "=" * 88,
    ]
    print("\n".join(lines), flush=True)


def _format_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    header_line = " | ".join(header.ljust(width) for header, width in zip(headers, widths))
    divider = "-+-".join("-" * width for width in widths)
    body = [" | ".join(value.ljust(width) for value, width in zip(row, widths)) for row in rows]
    return [header_line, divider, *body]


def _print_sweep_summary(results: list[dict], csv_path: Path) -> None:
    headers = [
        "case",
        "shape",
        "dtype",
        "fidelity",
        "split",
        "groups/iter",
        "full_ms",
        "sub_ms",
        "speedup",
        "full_TF/s",
        "sub_TF/s",
    ]
    rows = []
    for result in results:
        rows.append(
            [
                str(result["case"]),
                f"{result['m']}x{result['n']}x{result['k']}",
                result["dtype"],
                result["fidelity"],
                result["split"],
                str(result["pairs_per_iter"]),
                f"{result['full_seq_avg_ms']:.3f}",
                f"{result['sub_parallel_avg_ms']:.3f}",
                f"{result['speedup']:.3f}",
                f"{result['full_seq_tflops']:.3f}",
                f"{result['sub_parallel_tflops']:.3f}",
            ]
        )
    lines = [
        "",
        "=" * 116,
        "SUBDEVICE OPT MATMUL SWEEP RESULT",
        "-" * 116,
        (
            f"cases={len(results)} | cqs={results[0]['num_command_queues']} | "
            f"subdevices={results[0]['num_subdevices']} | warmup={results[0]['warmup']} | "
            f"iters={results[0]['iters']} | matmul_groups_per_iter={results[0]['pairs_per_iter']}"
        ),
        "-" * 116,
        *_format_table(headers, rows),
        "-" * 116,
        f"csv={csv_path}",
        "=" * 116,
    ]
    print("\n".join(lines), flush=True)


TEST_TIMEOUT = max(config.timeout_s for config in CONFIGS) * len(CONFIGS)
TEST_NUM_COMMAND_QUEUES = max(config.num_command_queues for config in CONFIGS)


def _run_config(device, config: SubdeviceMatmulConfig, case_index: int) -> tuple[dict, list[dict]]:
    _validate_config(config)

    grid = device.compute_with_storage_grid_size()
    cols, rows = int(grid.x), int(grid.y)
    if rows < 2:
        pytest.skip(f"Need at least 2 compute rows for split sub-device experiment, got {rows}")
    if rows % config.num_subdevices != 0:
        pytest.skip(
            f"SUBDEVICE_OPT_NUM_SUBDEVICES={config.num_subdevices} must evenly divide device rows={rows}"
        )

    sub_ranges = _split_row_ranges(cols, rows, config.num_subdevices)
    sub_device_ids = [ttnn.SubDeviceId(index) for index in range(config.num_subdevices)]
    sub_queue_ids = [index % config.num_command_queues for index in range(config.num_subdevices)]
    crs_full = _core_range(0, 0, cols - 1, rows - 1)

    full_program_config = _mcast2d_config(config, cols, rows, crs_full)
    sub_program_configs = [
        _mcast2d_config(config, cols, y1 - y0 + 1, core_range) for y0, y1, core_range in sub_ranges
    ]
    compute_kernel_config = _compute_kernel_config(config)

    inputs = [_make_inputs(device, config, seed=index) for index in range(config.num_subdevices)]
    ttnn.synchronize_device(device)

    full_seq_s: list[float] = []
    sub_parallel_s: list[float] = []
    csv_rows: list[dict] = []

    try:
        for _ in range(config.warmup_iterations):
            _run_full_sequential_once(
                device, inputs, full_program_config, compute_kernel_config, config.dtype, config.pairs_per_iteration
            )

        for iteration in range(config.measured_iterations):
            elapsed = _run_full_sequential_once(
                device, inputs, full_program_config, compute_kernel_config, config.dtype, config.pairs_per_iteration
            )
            full_seq_s.append(elapsed)
            csv_rows.append({"mode": "full_grid_sequential", "iteration": iteration, "duration_ms": elapsed * 1000.0})

        sub_devices = [ttnn.SubDevice([core_range]) for _, _, core_range in sub_ranges]
        manager = device.create_sub_device_manager(sub_devices, 0)
        device.load_sub_device_manager(manager)
        try:
            for _ in range(config.warmup_iterations):
                _run_subdevice_parallel_once(
                    device,
                    inputs,
                    sub_program_configs,
                    compute_kernel_config,
                    config.dtype,
                    sub_device_ids,
                    config.pairs_per_iteration,
                    sub_queue_ids,
                )

            for iteration in range(config.measured_iterations):
                elapsed = _run_subdevice_parallel_once(
                    device,
                    inputs,
                    sub_program_configs,
                    compute_kernel_config,
                    config.dtype,
                    sub_device_ids,
                    config.pairs_per_iteration,
                    sub_queue_ids,
                )
                sub_parallel_s.append(elapsed)
                csv_rows.append({"mode": "subdevice_parallel", "iteration": iteration, "duration_ms": elapsed * 1000.0})
        finally:
            device.reset_sub_device_stall_group()
            device.clear_loaded_sub_device_manager()
            device.remove_sub_device_manager(manager)
    finally:
        _deallocate_all(tensor for pair in inputs for tensor in pair)

    flops_per_matmul = 2.0 * config.m * config.n * config.k
    total_flops = float(config.pairs_per_iteration) * float(config.num_subdevices) * flops_per_matmul
    full_summary = _program_summary(config, cols, rows)
    sub_summaries = [_program_summary(config, cols, y1 - y0 + 1) for y0, y1, _ in sub_ranges]
    sub_grid_values = [f"{cols}x{y1 - y0 + 1}" for y0, y1, _ in sub_ranges]
    sub_core_values = [str(summary["cores"]) for summary in sub_summaries]
    sub_per_core_values = [f"{summary['per_core_n']}x{summary['per_core_m']}" for summary in sub_summaries]
    sub_out_block_values = [f"{summary['out_block_w']}x{summary['out_block_h']}" for summary in sub_summaries]
    sub_out_subblock_values = [
        f"{summary['out_subblock_w']}x{summary['out_subblock_h']}" for summary in sub_summaries
    ]
    sub_in0_block_w_values = [str(summary["in0_block_w"]) for summary in sub_summaries]

    avg_full_s = statistics.mean(full_seq_s)
    avg_parallel_s = statistics.mean(sub_parallel_s)
    result = {
        "case": case_index,
        "m": config.m,
        "n": config.n,
        "k": config.k,
        "dtype": config.dtype_name,
        "fidelity": config.math_fidelity_name,
        "fp32_acc": int(config.fp32_dest_acc_en),
        "packer_l1_acc": int(config.packer_l1_acc),
        "device_grid": f"{cols}x{rows}",
        "split": _compact_list(sub_grid_values),
        "num_command_queues": config.num_command_queues,
        "subdevice_parallel_cqs": "/".join(str(queue_id) for queue_id in sub_queue_ids),
        "num_subdevices": config.num_subdevices,
        "pairs_per_iter": config.pairs_per_iteration,
        "matmuls_per_iter": config.pairs_per_iteration * config.num_subdevices,
        "warmup": config.warmup_iterations,
        "iters": config.measured_iterations,
        "full_grid": f"{cols}x{rows}",
        "full_cores": full_summary["cores"],
        "full_per_core_m": full_summary["per_core_m"],
        "full_per_core_n": full_summary["per_core_n"],
        "full_out_block_h": full_summary["out_block_h"],
        "full_out_block_w": full_summary["out_block_w"],
        "full_out_subblock_h": full_summary["out_subblock_h"],
        "full_out_subblock_w": full_summary["out_subblock_w"],
        "full_blocks_x": full_summary["blocks_x"],
        "full_blocks_y": full_summary["blocks_y"],
        "full_in0_block_w": full_summary["in0_block_w"],
        "sub_grids": _compact_list(sub_grid_values),
        "sub_cores": _compact_list(sub_core_values),
        "sub_per_core": _compact_list(sub_per_core_values),
        "sub_out_block": _compact_list(sub_out_block_values),
        "sub_out_subblock": _compact_list(sub_out_subblock_values),
        "sub_in0_block_w": _compact_list(sub_in0_block_w_values),
        "full_seq_avg_ms": _avg_ms(full_seq_s),
        "full_seq_min_ms": min(full_seq_s) * 1000.0,
        "full_seq_max_ms": max(full_seq_s) * 1000.0,
        "full_seq_tflops": _tflops(total_flops, avg_full_s),
        "sub_parallel_avg_ms": _avg_ms(sub_parallel_s),
        "sub_parallel_min_ms": min(sub_parallel_s) * 1000.0,
        "sub_parallel_max_ms": max(sub_parallel_s) * 1000.0,
        "sub_parallel_tflops": _tflops(total_flops, avg_parallel_s),
        "speedup": avg_full_s / max(avg_parallel_s, 1e-12),
        "parallel_ratio": avg_parallel_s / max(avg_full_s, 1e-12),
        "total_tflops_per_iter": total_flops / 1e12,
        "csv": str(config.csv_path),
    }

    for row in csv_rows:
        row["case"] = case_index
        row["m"] = config.m
        row["n"] = config.n
        row["k"] = config.k
        row["dtype"] = config.dtype_name
        row["fidelity"] = config.math_fidelity_name
        row["num_command_queues"] = config.num_command_queues
        row["subdevice_parallel_cqs"] = "/".join(str(queue_id) for queue_id in sub_queue_ids)
        row["num_subdevices"] = config.num_subdevices
        row["pairs_per_iter"] = config.pairs_per_iteration
        row["matmuls_per_iter"] = config.pairs_per_iteration * config.num_subdevices
        row["total_tflops_per_iter"] = total_flops / 1e12
        row["tflops_per_s"] = _tflops(total_flops, row["duration_ms"] / 1000.0)

    return result, csv_rows


@pytest.mark.timeout(TEST_TIMEOUT)
@pytest.mark.parametrize("device_params", [{"num_command_queues": TEST_NUM_COMMAND_QUEUES}], indirect=True)
def test_subdevice_mcast2d_matmul_parallel_vs_full_grid_sequential(device, device_params):
    results: list[dict] = []
    csv_rows: list[dict] = []

    for case_index, config in enumerate(CONFIGS):
        result, rows = _run_config(device, config, case_index)
        results.append(result)
        csv_rows.extend(rows)

    _write_csv(CONFIG, csv_rows)
    _print_sweep_summary(results, CONFIG.csv_path)
