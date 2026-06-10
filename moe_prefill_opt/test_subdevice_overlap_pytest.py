# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Pytest timing litmus for TTNN sub-device overlap.

This is intentionally a softer test than the C++ semaphore litmus: Python does
not expose the low-level Metal Program/CreateKernel path, so this test infers
overlap by comparing:

  serial:   matmul on sub-device A, sync, matmul on sub-device B, sync
  overlap: matmul on sub-device A, matmul on sub-device B, one sync

If sub-device execution overlaps, the overlap timing should be much closer to
max(A, B) than to A + B.  The test writes the raw timings to CSV so the result
can be inspected even when the assertion threshold needs tuning for a system.
"""

from __future__ import annotations

import csv
import os
import statistics
import time
from pathlib import Path

import pytest
import torch

import ttnn


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _core_range(cols: int, start_y: int, rows: int) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, start_y),
                ttnn.CoreCoord(cols - 1, start_y + rows - 1),
            )
        }
    )


def _make_inputs(device, m: int, k: int, n: int):
    torch.manual_seed(0)
    a = torch.randn((1, 1, m, k), dtype=torch.bfloat16)
    b = torch.randn((1, 1, k, n), dtype=torch.bfloat16)
    tt_a = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return tt_a, tt_b


def _run_matmul(tt_a, tt_b, grid: ttnn.CoreGrid, sub_device_id: ttnn.SubDeviceId):
    return ttnn.matmul(
        tt_a,
        tt_b,
        core_grid=grid,
        sub_device_id=sub_device_id,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _timed_single(device, tt_a, tt_b, grid, sub_device_id, stall_group):
    ttnn.synchronize_device(device)
    device.set_sub_device_stall_group(stall_group)
    start_s = time.perf_counter()
    output = _run_matmul(tt_a, tt_b, grid, sub_device_id)
    ttnn.synchronize_device(device)
    return time.perf_counter() - start_s, output


def _timed_serial_pair(device, tt_a, tt_b, grid_a, grid_b, sd_a, sd_b):
    ttnn.synchronize_device(device)
    start_s = time.perf_counter()
    device.set_sub_device_stall_group([sd_a])
    out_a = _run_matmul(tt_a, tt_b, grid_a, sd_a)
    ttnn.synchronize_device(device)
    device.set_sub_device_stall_group([sd_b])
    out_b = _run_matmul(tt_a, tt_b, grid_b, sd_b)
    ttnn.synchronize_device(device)
    return time.perf_counter() - start_s, (out_a, out_b)


def _timed_overlap_pair(device, tt_a, tt_b, grid_a, grid_b, sd_a, sd_b):
    ttnn.synchronize_device(device)
    start_s = time.perf_counter()
    device.set_sub_device_stall_group([sd_a])
    out_a = _run_matmul(tt_a, tt_b, grid_a, sd_a)
    device.set_sub_device_stall_group([sd_b])
    out_b = _run_matmul(tt_a, tt_b, grid_b, sd_b)
    device.reset_sub_device_stall_group()
    ttnn.synchronize_device(device)
    return time.perf_counter() - start_s, (out_a, out_b)


@pytest.mark.timeout(_env_int("TTNN_SUBDEVICE_OVERLAP_TIMEOUT", 120))
def test_ttnn_matmul_subdevice_overlap_timing(device):
    grid = device.compute_with_storage_grid_size()
    cols, rows = grid.x, grid.y
    if rows < 2:
        pytest.skip(f"Need at least 2 compute rows for two sub-devices, got {rows}")

    rows_a = rows // 2
    rows_b = rows - rows_a
    sd_a = ttnn.SubDeviceId(0)
    sd_b = ttnn.SubDeviceId(1)
    crs_a = _core_range(cols, 0, rows_a)
    crs_b = _core_range(cols, rows_a, rows_b)
    manager = device.create_sub_device_manager([ttnn.SubDevice([crs_a]), ttnn.SubDevice([crs_b])], 0)
    device.load_sub_device_manager(manager)

    m = _env_int("TTNN_SUBDEVICE_OVERLAP_M", 2048)
    k = _env_int("TTNN_SUBDEVICE_OVERLAP_K", 2048)
    n = _env_int("TTNN_SUBDEVICE_OVERLAP_N", 2048)
    warmup = _env_int("TTNN_SUBDEVICE_OVERLAP_WARMUP", 1)
    iterations = _env_int("TTNN_SUBDEVICE_OVERLAP_ITERS", 5)
    threshold = _env_float("TTNN_SUBDEVICE_OVERLAP_MAX_SERIAL_RATIO", 0.85)
    csv_path = Path(
        os.getenv("TTNN_SUBDEVICE_OVERLAP_CSV", "/tmp/ttnn_subdevice_overlap_timing.csv")
    )

    grid_a = ttnn.CoreGrid(x=cols, y=rows_a)
    grid_b = ttnn.CoreGrid(x=cols, y=rows_b)

    try:
        device.set_sub_device_stall_group([sd_a, sd_b])
        tt_a, tt_b = _make_inputs(device, m, k, n)

        for _ in range(warmup):
            _timed_single(device, tt_a, tt_b, grid_a, sd_a, [sd_a])
            _timed_single(device, tt_a, tt_b, grid_b, sd_b, [sd_b])
            _timed_overlap_pair(device, tt_a, tt_b, grid_a, grid_b, sd_a, sd_b)

        rows_out = []
        single_a_s = []
        single_b_s = []
        serial_pair_s = []
        overlap_pair_s = []

        for iteration in range(iterations):
            elapsed_a, _ = _timed_single(device, tt_a, tt_b, grid_a, sd_a, [sd_a])
            elapsed_b, _ = _timed_single(device, tt_a, tt_b, grid_b, sd_b, [sd_b])
            elapsed_serial, _ = _timed_serial_pair(device, tt_a, tt_b, grid_a, grid_b, sd_a, sd_b)
            elapsed_overlap, _ = _timed_overlap_pair(device, tt_a, tt_b, grid_a, grid_b, sd_a, sd_b)
            single_a_s.append(elapsed_a)
            single_b_s.append(elapsed_b)
            serial_pair_s.append(elapsed_serial)
            overlap_pair_s.append(elapsed_overlap)
            rows_out.append(
                {
                    "iteration": iteration,
                    "single_a_ms": elapsed_a * 1000.0,
                    "single_b_ms": elapsed_b * 1000.0,
                    "single_sum_ms": (elapsed_a + elapsed_b) * 1000.0,
                    "serial_pair_ms": elapsed_serial * 1000.0,
                    "overlap_pair_ms": elapsed_overlap * 1000.0,
                    "overlap_vs_single_sum": elapsed_overlap / max(elapsed_a + elapsed_b, 1e-12),
                    "overlap_vs_serial_pair": elapsed_overlap / max(elapsed_serial, 1e-12),
                }
            )

        avg_single_sum_s = statistics.mean(a + b for a, b in zip(single_a_s, single_b_s))
        avg_serial_pair_s = statistics.mean(serial_pair_s)
        avg_overlap_s = statistics.mean(overlap_pair_s)
        overlap_vs_single_sum = avg_overlap_s / max(avg_single_sum_s, 1e-12)
        overlap_vs_serial_pair = avg_overlap_s / max(avg_serial_pair_s, 1e-12)

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as csv_file:
            fieldnames = [
                "iteration",
                "single_a_ms",
                "single_b_ms",
                "single_sum_ms",
                "serial_pair_ms",
                "overlap_pair_ms",
                "overlap_vs_single_sum",
                "overlap_vs_serial_pair",
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_out)
            writer.writerow(
                {
                    "iteration": "avg",
                    "single_a_ms": statistics.mean(single_a_s) * 1000.0,
                    "single_b_ms": statistics.mean(single_b_s) * 1000.0,
                    "single_sum_ms": avg_single_sum_s * 1000.0,
                    "serial_pair_ms": avg_serial_pair_s * 1000.0,
                    "overlap_pair_ms": avg_overlap_s * 1000.0,
                    "overlap_vs_single_sum": overlap_vs_single_sum,
                    "overlap_vs_serial_pair": overlap_vs_serial_pair,
                }
            )

        print(
            "TTNN sub-device overlap timing: "
            f"grid={cols}x{rows}, split={rows_a}+{rows_b}, shape={m}x{k}x{n}, "
            f"single_sum={avg_single_sum_s * 1000.0:.3f}ms, "
            f"serial_pair={avg_serial_pair_s * 1000.0:.3f}ms, "
            f"overlap_pair={avg_overlap_s * 1000.0:.3f}ms, "
            f"overlap/single_sum={overlap_vs_single_sum:.3f}, "
            f"overlap/serial_pair={overlap_vs_serial_pair:.3f}, "
            f"csv={csv_path}"
        )

        assert overlap_vs_single_sum < threshold, (
            "Did not observe enough timing overlap between sub-device matmuls: "
            f"overlap/single_sum={overlap_vs_single_sum:.3f}, threshold={threshold}. "
            f"See {csv_path} for raw timings."
        )
    finally:
        device.reset_sub_device_stall_group()
        device.clear_loaded_sub_device_manager()
        device.remove_sub_device_manager(manager)
