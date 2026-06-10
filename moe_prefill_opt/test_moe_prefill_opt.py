# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Perf harness for the isolated DeepSeek prefill MoE layer.

Example:
    pytest -q -s moe_prefill_opt/test_moe_prefill_opt.py

Useful overrides:
    MOE_PREFILL_OPT_MESH=1x1
    MOE_PREFILL_OPT_SEQ_LEN=4096
    MOE_PREFILL_OPT_EMB_DIM=2048
    MOE_PREFILL_OPT_HIDDEN_DIM=768
    MOE_PREFILL_OPT_NUM_EXPERTS=128
    MOE_PREFILL_OPT_TOPK=8
    MOE_PREFILL_OPT_WARMUP=1
    MOE_PREFILL_OPT_ITERS=3
    MOE_PREFILL_OPT_GATE_MODE=host_all
    MOE_PREFILL_OPT_ROUTING_PROFILE=spiky_soft
    MOE_PREFILL_OPT_DYNAMIC_EXPERT_CAPACITY=1
    MOE_PREFILL_OPT_MEASURE_EXPERT_TIME=1
    MOE_PREFILL_OPT_EXPERT_FFN_BREAKDOWN=1
    MOE_PREFILL_OPT_EXPERT_FFN_BREAKDOWN_WARMUP=1
    MOE_PREFILL_OPT_EXPERT_FFN_BREAKDOWN_ITERS=3
    MOE_PREFILL_OPT_EXPERT_FFN_SWEEP_ROWS=131072,65536,32768,16384,8192,1024,512,256,128
    MOE_PREFILL_OPT_EXPERT_FFN_SWEEP_CSV=/home/tenstorrent/tt-metal/moe_prefill_opt/moe_prefill_opt_expert_ffn_sweep.csv
    MOE_PREFILL_OPT_EXPERT_SUBDEVICE_PARALLEL=1
    MOE_PREFILL_OPT_EXPERT_BASELINE_FUSED_CPP=1
    MOE_PREFILL_OPT_EXPERT_SUBDEVICE_FUSED_CPP=1
    MOE_PREFILL_OPT_EXPERT_SUBDEVICE_CPP_MODE=stage_rr  # stage_rr or composite
    MOE_PREFILL_OPT_EXPERT_SUBDEVICE_RT_PROFILE=1
    MOE_PREFILL_OPT_EXPERT_SUBDEVICE_RT_PROFILE_CSV=/home/tenstorrent/tt-metal/moe_prefill_opt/moe_prefill_opt_expert_subdevices_rt_profile.csv
    MOE_PREFILL_OPT_EXPERT_SUBDEVICE_EQUAL_ROWS=32  # optional: force every expert input to [rows, emb_dim]
    MOE_PREFILL_OPT_EXPERT_SUBDEVICE_MAX_EXPERTS=0  # 0 means all active experts
    MOE_PREFILL_OPT_EXPERT_SUBDEVICE_LANE_TARGET_LOAD=1024  # optional: force lane target load
    MOE_PREFILL_OPT_EXPERT_SUBDEVICE_PHASE_OVERHEAD_LOAD=128  # scheduler phase-switch penalty
    MOE_PREFILL_OPT_EXPERT_SUBDEVICE_CSV=/home/tenstorrent/tt-metal/moe_prefill_opt/moe_prefill_opt_expert_subdevices.csv
    MOE_PREFILL_OPT_MCAST2D_CORE_LOG=1
    MOE_PREFILL_OPT_MCAST2D_CORE_LOG_CSV=/home/tenstorrent/tt-metal/moe_prefill_opt/moe_prefill_opt_mcast2d_actual_cores.csv
    MOE_PREFILL_OPT_TIMEOUT=0  # disable pytest-timeout for long Qwen-scale perf runs
    MOE_PREFILL_OPT_EXPERT_FREQ_PATH=/path/to/expert_frequency.json
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pytest
import torch
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# JIT build command dumps are very noisy and can bury the perf result block.
# Users can opt back in when debugging kernel builds.
if not os.getenv("MOE_PREFILL_OPT_SHOW_JIT_COMPILE_COMMANDS"):
    os.environ.pop("TT_METAL_LOG_KERNELS_COMPILE_COMMANDS", None)
    os.environ.pop("TT_METAL_BACKEND_DUMP_RUN_CMD", None)

import ttnn
from moe_prefill_opt.init_helpers_opt import (
    compute_constants,
    create_fabric_router_config,
    create_gate_weights,
    create_shared_expert_weights,
    create_torch_expert_weights,
    extract_mesh_config,
)
from moe_prefill_opt.tt_moe_gate_prefill_opt import GateComputeMode, sample_frequency_routing
from moe_prefill_opt.tt_moe_opt import TtMoe
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker


_RESULTS: list[dict] = []
_CSV_PATH = Path(__file__).resolve().with_name("moe_prefill_opt_perf.csv")
_TILE = 32
_DEFAULT_FREQ_MAX_PERCENT = 11.0
_DEFAULT_FREQ_MIN_PERCENT = 0.002


@dataclass(frozen=True)
class MoePerfConfig:
    name: str
    mesh_shape: tuple[int, int]
    seq_len_per_chip: int
    emb_dim: int
    hidden_dim: int
    num_routed_experts: int
    num_experts_per_tok: int
    dispatch_buffer_capacity_factor: int
    gate_mode: GateComputeMode
    warmup_iterations: int
    measured_iterations: int
    seed: int
    use_cache: bool
    overlap_shared_expert_with_dispatch: bool
    use_shared_expert: bool
    routing_profile: str
    expert_frequency_path: str
    frequency_max_percent: float
    frequency_min_percent: float
    dynamic_expert_capacity: bool
    measure_expert_times: bool
    expert_ffn_breakdown: bool
    expert_ffn_breakdown_warmup_iterations: int
    expert_ffn_breakdown_iterations: int
    expert_ffn_sweep_rows: tuple[int, ...]
    expert_ffn_sweep_csv_path: str
    expert_ffn_sweep_expert_id: int
    expert_subdevice_parallel: bool
    expert_baseline_fused_cpp: bool
    expert_subdevice_fused_cpp: bool
    expert_subdevice_cpp_mode: str
    expert_subdevice_rt_profile: bool
    expert_subdevice_rt_profile_csv_path: str
    expert_subdevice_equal_rows: int
    expert_subdevice_max_experts: int
    expert_subdevice_warmup_iterations: int
    expert_subdevice_iterations: int
    expert_subdevice_csv_path: str
    mcast2d_core_log: bool
    mcast2d_core_log_csv_path: str
    mcast2d_core_log_append: bool
    routed_activations_dtype: ttnn.DataType
    routed_weights_dtype: ttnn.DataType
    shared_activations_dtype: ttnn.DataType
    shared_weights_dtype: ttnn.DataType


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _parse_int_list(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    normalized = value.replace(";", ",").replace("x", ",")
    rows = []
    for part in normalized.split(","):
        part = part.strip()
        if not part:
            continue
        rows.append(int(part))
    return tuple(rows)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def _parse_mesh_shape(value: str) -> tuple[int, int]:
    normalized = value.lower().replace(",", "x")
    parts = normalized.split("x")
    if len(parts) != 2:
        raise ValueError(f"Expected mesh shape like '1x1', got {value!r}")
    return int(parts[0]), int(parts[1])


def _parse_gate_mode(value: str) -> GateComputeMode:
    normalized = value.lower()
    for mode in GateComputeMode:
        if normalized in {mode.value, mode.name.lower()}:
            return mode
    valid = ", ".join(mode.value for mode in GateComputeMode)
    raise ValueError(f"Unknown MOE_PREFILL_OPT_GATE_MODE={value!r}; valid values: {valid}")


def _parse_dtype(value: str) -> ttnn.DataType:
    normalized = value.lower()
    if normalized in {"bf16", "bfloat16"}:
        return ttnn.bfloat16
    if normalized in {"bf8", "bfloat8", "bfloat8_b"}:
        return ttnn.bfloat8_b
    if normalized in {"bf4", "bfloat4", "bfloat4_b"}:
        return ttnn.bfloat4_b
    raise ValueError(f"Unsupported dtype {value!r}; use bf16, bf8, or bf4")


def _dtype_nbytes(dtype: ttnn.DataType) -> float:
    if dtype == ttnn.bfloat16:
        return 2.0
    if dtype == ttnn.bfloat8_b:
        return 1.0
    if dtype == ttnn.bfloat4_b:
        return 0.5
    return 2.0


def _load_config_from_env() -> MoePerfConfig:
    legacy_subdevice_fused_cpp = _env_flag("MOE_PREFILL_OPT_EXPERT_SUBDEVICE_FUSED_CPP", True)
    return MoePerfConfig(
        name=os.getenv("MOE_PREFILL_OPT_NAME", "qwen3_30b_a3b_prefill"),
        mesh_shape=_parse_mesh_shape(os.getenv("MOE_PREFILL_OPT_MESH", "1x1")),
        seq_len_per_chip=_env_int("MOE_PREFILL_OPT_SEQ_LEN", 4096),
        emb_dim=_env_int("MOE_PREFILL_OPT_EMB_DIM", 2048),
        hidden_dim=_env_int("MOE_PREFILL_OPT_HIDDEN_DIM", 768),
        num_routed_experts=_env_int("MOE_PREFILL_OPT_NUM_EXPERTS", 128),
        num_experts_per_tok=_env_int("MOE_PREFILL_OPT_TOPK", 8),
        dispatch_buffer_capacity_factor=_env_int("MOE_PREFILL_OPT_DISPATCH_CAPACITY_FACTOR", 8),
        gate_mode=_parse_gate_mode(os.getenv("MOE_PREFILL_OPT_GATE_MODE", "host_all")),
        warmup_iterations=_env_int("MOE_PREFILL_OPT_WARMUP", 1),
        measured_iterations=_env_int("MOE_PREFILL_OPT_ITERS", 3),
        seed=_env_int("MOE_PREFILL_OPT_SEED", 42),
        use_cache=_env_flag("MOE_PREFILL_OPT_USE_CACHE", True),
        overlap_shared_expert_with_dispatch=_env_flag("MOE_PREFILL_OPT_OVERLAP_SHARED_DISPATCH", True),
        use_shared_expert=_env_flag("MOE_PREFILL_OPT_USE_SHARED_EXPERT", False),
        routing_profile=os.getenv("MOE_PREFILL_OPT_ROUTING_PROFILE", "spiky_soft"),
        expert_frequency_path=os.getenv("MOE_PREFILL_OPT_EXPERT_FREQ_PATH", ""),
        frequency_max_percent=_env_float("MOE_PREFILL_OPT_FREQ_MAX_PERCENT", _DEFAULT_FREQ_MAX_PERCENT),
        frequency_min_percent=_env_float("MOE_PREFILL_OPT_FREQ_MIN_PERCENT", _DEFAULT_FREQ_MIN_PERCENT),
        dynamic_expert_capacity=_env_flag("MOE_PREFILL_OPT_DYNAMIC_EXPERT_CAPACITY", True),
        measure_expert_times=_env_flag("MOE_PREFILL_OPT_MEASURE_EXPERT_TIME", True),
        expert_ffn_breakdown=_env_flag("MOE_PREFILL_OPT_EXPERT_FFN_BREAKDOWN", False),
        expert_ffn_breakdown_warmup_iterations=_env_int("MOE_PREFILL_OPT_EXPERT_FFN_BREAKDOWN_WARMUP", 1),
        expert_ffn_breakdown_iterations=_env_int(
            "MOE_PREFILL_OPT_EXPERT_FFN_BREAKDOWN_ITERS",
            _env_int("MOE_PREFILL_OPT_ITERS", 3),
        ),
        expert_ffn_sweep_rows=_parse_int_list(os.getenv("MOE_PREFILL_OPT_EXPERT_FFN_SWEEP_ROWS", "")),
        expert_ffn_sweep_csv_path=os.getenv(
            "MOE_PREFILL_OPT_EXPERT_FFN_SWEEP_CSV",
            str(Path(__file__).resolve().with_name("moe_prefill_opt_expert_ffn_sweep.csv")),
        ),
        expert_ffn_sweep_expert_id=_env_int("MOE_PREFILL_OPT_EXPERT_FFN_SWEEP_EXPERT_ID", 0),
        expert_subdevice_parallel=_env_flag("MOE_PREFILL_OPT_EXPERT_SUBDEVICE_PARALLEL", False),
        expert_baseline_fused_cpp=_env_flag(
            "MOE_PREFILL_OPT_EXPERT_BASELINE_FUSED_CPP",
            legacy_subdevice_fused_cpp,
        ),
        expert_subdevice_fused_cpp=legacy_subdevice_fused_cpp,
        expert_subdevice_cpp_mode=os.getenv("MOE_PREFILL_OPT_EXPERT_SUBDEVICE_CPP_MODE", "stage_rr").lower(),
        expert_subdevice_rt_profile=_env_flag("MOE_PREFILL_OPT_EXPERT_SUBDEVICE_RT_PROFILE", False),
        expert_subdevice_rt_profile_csv_path=os.getenv(
            "MOE_PREFILL_OPT_EXPERT_SUBDEVICE_RT_PROFILE_CSV",
            str(Path(__file__).resolve().with_name("moe_prefill_opt_expert_subdevices_rt_profile.csv")),
        ),
        expert_subdevice_equal_rows=_env_int("MOE_PREFILL_OPT_EXPERT_SUBDEVICE_EQUAL_ROWS", 0),
        expert_subdevice_max_experts=_env_int("MOE_PREFILL_OPT_EXPERT_SUBDEVICE_MAX_EXPERTS", 0),
        expert_subdevice_warmup_iterations=_env_int(
            "MOE_PREFILL_OPT_EXPERT_SUBDEVICE_WARMUP",
            _env_int("MOE_PREFILL_OPT_EXPERT_FFN_BREAKDOWN_WARMUP", 1),
        ),
        expert_subdevice_iterations=_env_int(
            "MOE_PREFILL_OPT_EXPERT_SUBDEVICE_ITERS",
            _env_int("MOE_PREFILL_OPT_EXPERT_FFN_BREAKDOWN_ITERS", _env_int("MOE_PREFILL_OPT_ITERS", 3)),
        ),
        expert_subdevice_csv_path=os.getenv(
            "MOE_PREFILL_OPT_EXPERT_SUBDEVICE_CSV",
            str(Path(__file__).resolve().with_name("moe_prefill_opt_expert_subdevices.csv")),
        ),
        mcast2d_core_log=_env_flag(
            "MOE_PREFILL_OPT_MCAST2D_CORE_LOG",
            _env_flag("MOE_PREFILL_OPT_EXPERT_FFN_BREAKDOWN", False)
            or bool(os.getenv("MOE_PREFILL_OPT_EXPERT_FFN_SWEEP_ROWS", "").strip()),
        ),
        mcast2d_core_log_csv_path=os.getenv(
            "MOE_PREFILL_OPT_MCAST2D_CORE_LOG_CSV",
            os.getenv(
                "TTNN_MCAST2D_CORE_LOG_PATH",
                str(Path(__file__).resolve().with_name("moe_prefill_opt_mcast2d_actual_cores.csv")),
            ),
        ),
        mcast2d_core_log_append=_env_flag("MOE_PREFILL_OPT_MCAST2D_CORE_LOG_APPEND", False),
        # routed_activations_dtype=_parse_dtype(os.getenv("MOE_PREFILL_OPT_ROUTED_ACT_DTYPE", "bf8")),
        # routed_weights_dtype=_parse_dtype(os.getenv("MOE_PREFILL_OPT_ROUTED_WEIGHT_DTYPE", "bf16")),
        routed_activations_dtype=_parse_dtype(os.getenv("MOE_PREFILL_OPT_ROUTED_ACT_DTYPE", "bf16")),
        routed_weights_dtype=_parse_dtype(os.getenv("MOE_PREFILL_OPT_ROUTED_WEIGHT_DTYPE", "bf8")),
        shared_activations_dtype=_parse_dtype(os.getenv("MOE_PREFILL_OPT_SHARED_ACT_DTYPE", "bf16")),
        shared_weights_dtype=_parse_dtype(os.getenv("MOE_PREFILL_OPT_SHARED_WEIGHT_DTYPE", "bf16")),
    )


CONFIG = _load_config_from_env()


def _device_params_for_config(config: MoePerfConfig) -> dict:
    # DeepSeek dispatch/combine wrappers query the fabric control plane even for
    # all-local 1x1 runs, so initialize fabric context for the opt harness too.
    device_params = {
        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        "fabric_router_config": create_fabric_router_config(max_payload_size=config.emb_dim),
    }
    if config.expert_subdevice_rt_profile:
        # The RT profiler needs a reserved tensix from the worker-dispatch pool.
        # With ETH dispatch IsProgramRealtimeProfilerActive() stays false and no
        # device-side records are produced.
        device_params["dispatch_core_type"] = ttnn.DispatchCoreType.WORKER
    return device_params


def _validate_config(config: MoePerfConfig, mesh_device: ttnn.MeshDevice) -> None:
    for name, value in (
        ("seq_len_per_chip", config.seq_len_per_chip),
        ("emb_dim", config.emb_dim),
        ("hidden_dim", config.hidden_dim),
    ):
        assert value % _TILE == 0, f"{name}={value} must be a multiple of {_TILE}"
    # assert config.num_routed_experts % 8 == 0, "num_routed_experts must be divisible by DeepSeek n_group=8"
    assert config.num_experts_per_tok <= config.num_routed_experts
    assert config.num_routed_experts % mesh_device.get_num_devices() == 0
    assert config.measured_iterations > 0
    if config.expert_subdevice_equal_rows > 0:
        assert (
            config.expert_subdevice_equal_rows % _TILE == 0
        ), f"MOE_PREFILL_OPT_EXPERT_SUBDEVICE_EQUAL_ROWS={config.expert_subdevice_equal_rows} must be a multiple of {_TILE}"

    device_grouped_gate_modes = {
        GateComputeMode.DEVICE,
        GateComputeMode.DEVICE_FP32,
        GateComputeMode.HOST_MATMUL,
    }
    if config.gate_mode in device_grouped_gate_modes and (
        config.num_routed_experts != 256 or config.num_experts_per_tok != 8
    ):
        pytest.skip("device grouped-gate path currently expects 256 experts and topk=8; use host_all for small configs")


def _cache_dir(config: MoePerfConfig, mesh_device: ttnn.MeshDevice) -> Path:
    base = Path(os.getenv("MOE_PREFILL_OPT_CACHE_DIR", "/tmp/moe_prefill_opt_cache"))
    dtype_tag = (
        f"ra{config.routed_activations_dtype}_rw{config.routed_weights_dtype}_"
        f"sa{config.shared_activations_dtype}_sw{config.shared_weights_dtype}_"
        f"shared{int(config.use_shared_expert)}"
    ).replace("DataType.", "").replace(" ", "")
    return base / (
        f"{config.num_routed_experts}experts_{mesh_device.shape[0]}x{mesh_device.shape[1]}mesh_"
        f"{config.seq_len_per_chip}seq_{config.emb_dim}emb_{config.hidden_dim}hid_{dtype_tag}"
    )


def _prepare_weights(config: MoePerfConfig, mesh_device: ttnn.MeshDevice, experts_per_chip: int) -> tuple:
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    cache_path = _cache_dir(config, mesh_device) if config.use_cache else None
    if cache_path is None:
        logger.info("Creating MoE weights without persistent cache")
        return (
            create_gate_weights(config.num_routed_experts, config.emb_dim),
            create_torch_expert_weights(config.num_routed_experts, config.emb_dim, config.hidden_dim),
            create_shared_expert_weights(config.emb_dim, config.hidden_dim) if config.use_shared_expert else None,
            None,
        )

    cache_path.mkdir(parents=True, exist_ok=True)
    init_checker(cache_path)
    if TtMoe.check_cache_complete(
        cache_path,
        layer_idx=0,
        experts_per_chip=experts_per_chip,
        include_shared_expert=config.use_shared_expert,
    ):
        logger.info(f"Using existing MoE opt cache: {cache_path}")
        return None, None, None, cache_path

    logger.info(f"Building MoE opt cache: {cache_path}")
    gate_weights = create_gate_weights(config.num_routed_experts, config.emb_dim)
    routed_expert_weights = create_torch_expert_weights(
        config.num_routed_experts,
        config.emb_dim,
        config.hidden_dim,
    )
    shared_expert_weights = (
        create_shared_expert_weights(config.emb_dim, config.hidden_dim) if config.use_shared_expert else None
    )
    TtMoe.build_ttnn_cache(
        gate_weights=gate_weights,
        routed_expert_weights=routed_expert_weights,
        shared_expert_weights=shared_expert_weights,
        experts_per_chip=experts_per_chip,
        emb_dim=config.emb_dim,
        hidden_dim=config.hidden_dim,
        mesh_device=mesh_device,
        routed_expert_weights_dtype=config.routed_weights_dtype,
        shared_expert_weights_dtype=config.shared_weights_dtype,
        cache_path=cache_path,
        layer_idx=0,
    )
    return None, None, None, cache_path


def _frequency_from_mapping(data: dict, num_experts: int) -> torch.Tensor:
    for key in ("frequencies", "frequency", "counts", "expert_frequencies", "expert_counts"):
        if key in data:
            return torch.tensor(data[key], dtype=torch.float32)

    values = []
    for expert_idx in range(num_experts):
        if str(expert_idx) in data:
            values.append(float(data[str(expert_idx)]))
        elif expert_idx in data:
            values.append(float(data[expert_idx]))
        else:
            values.append(0.0)
    if any(value > 0 for value in values):
        return torch.tensor(values, dtype=torch.float32)
    raise ValueError("Frequency JSON object must contain a frequencies/counts list or expert-id keys")


def _load_frequency_file(path: Path, num_experts: int) -> torch.Tensor:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, list):
            frequency = torch.tensor(data, dtype=torch.float32)
        elif isinstance(data, dict):
            frequency = _frequency_from_mapping(data, num_experts)
        else:
            raise ValueError(f"Unsupported JSON frequency format in {path}")
    else:
        values = []
        indexed_values = {}
        with path.open(newline="") as csv_file:
            for row in csv.reader(csv_file):
                if not row:
                    continue
                try:
                    if len(row) == 1:
                        values.append(float(row[0]))
                    else:
                        indexed_values[int(row[0])] = float(row[1])
                except ValueError:
                    continue
        if indexed_values:
            frequency = torch.tensor([indexed_values.get(idx, 0.0) for idx in range(num_experts)], dtype=torch.float32)
        else:
            frequency = torch.tensor(values, dtype=torch.float32)

    if frequency.numel() != num_experts:
        raise ValueError(f"{path} has {frequency.numel()} frequency entries, expected {num_experts}")
    if torch.any(frequency < 0):
        raise ValueError(f"{path} contains negative frequencies")
    if frequency.sum() <= 0:
        raise ValueError(f"{path} contains no positive frequency")
    return frequency


def _long_tail_range_frequency(
    num_experts: int,
    seed: int,
    max_percent: float,
    min_percent: float,
) -> torch.Tensor:
    max_probability = max_percent / 100.0
    min_probability = min_percent / 100.0
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if num_experts == 1:
        return torch.ones(1, dtype=torch.float32)
    if min_probability <= 0 or max_probability <= 0:
        raise ValueError("frequency bounds must be positive")
    if min_probability > max_probability:
        raise ValueError("MOE_PREFILL_OPT_FREQ_MIN_PERCENT must be <= MOE_PREFILL_OPT_FREQ_MAX_PERCENT")
    if min_probability * num_experts > 1.0:
        raise ValueError(
            f"min frequency {min_percent}% is too large for {num_experts} experts to sum to 100%"
        )
    if max_probability * num_experts < 1.0:
        raise ValueError(
            f"max frequency {max_percent}% is too small for {num_experts} experts to sum to 100%"
        )

    positions = torch.linspace(0.0, 1.0, num_experts, dtype=torch.float64)
    base = torch.full((num_experts,), min_probability, dtype=torch.float64)
    residual = 1.0 - float(base.sum().item())
    if residual < -1e-12:
        raise ValueError(
            f"min frequency {min_percent}% is too large for {num_experts} experts to sum to 100%"
        )
    if residual <= 1e-12:
        probabilities = base
    else:
        # Treat max_percent as a target/cap. Exact max_percent is infeasible for
        # cases like max=100%, min>0, so the achievable top probability is bounded
        # by the mandatory min mass reserved for the remaining experts.
        achievable_top = min(max_probability, 1.0 - min_probability * (num_experts - 1))
        target_top_share = (achievable_top - min_probability) / residual
        target_top_share = max(1.0 / num_experts, min(1.0, target_top_share))

        low_alpha = 0.0
        high_alpha = 512.0
        for _ in range(128):
            alpha = (low_alpha + high_alpha) / 2.0
            weights = torch.pow(1.0 - positions, alpha)
            weights = weights / weights.sum()
            if float(weights[0].item()) < target_top_share:
                low_alpha = alpha
            else:
                high_alpha = alpha

        weights = torch.pow(1.0 - positions, high_alpha)
        weights = weights / weights.sum()
        probabilities = base + residual * weights
        probabilities = torch.clamp(probabilities, min=0.0)
        probabilities = probabilities / probabilities.sum()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 2026)
    return probabilities.float()[torch.randperm(num_experts, generator=generator)]


def _spiky_soft_frequency(
    num_experts: int,
    seed: int,
    max_percent: float,
    min_percent: float,
) -> torch.Tensor:
    max_probability = max_percent / 100.0
    min_probability = min_percent / 100.0
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if num_experts == 1:
        return torch.ones(1, dtype=torch.float32)
    if min_probability <= 0 or max_probability <= 0:
        raise ValueError("frequency bounds must be positive")
    if min_probability > max_probability:
        raise ValueError("MOE_PREFILL_OPT_FREQ_MIN_PERCENT must be <= MOE_PREFILL_OPT_FREQ_MAX_PERCENT")
    if min_probability * num_experts > 1.0:
        raise ValueError(
            f"min frequency {min_percent}% is too large for {num_experts} experts to sum to 100%"
        )
    if max_probability * num_experts < 1.0:
        raise ValueError(
            f"max frequency {max_percent}% is too small for {num_experts} experts to sum to 100%"
        )

    positions = torch.linspace(0.0, 1.0, num_experts, dtype=torch.float64)
    shape = torch.full((num_experts,), 1e-8, dtype=torch.float64)
    spike_positions = torch.tensor(
        [0.10, 0.34, 0.63, 0.82, 0.90, 0.95],
        dtype=torch.float64,
    )
    spike_heights = torch.tensor(
        [0.28, 0.22, 0.74, 1.00, 0.26, 0.68],
        dtype=torch.float64,
    )
    spike_width = 0.003
    for center, height in zip(spike_positions.tolist(), spike_heights.tolist()):
        shape += height * torch.exp(-0.5 * torch.square((positions - center) / spike_width))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 4515)
    shape += 0.001 * torch.rand(num_experts, generator=generator, dtype=torch.float64)
    shape = shape.clamp_min(1e-12)

    residual = 1.0 - min_probability * num_experts
    if residual <= 1e-12:
        return torch.full((num_experts,), 1.0 / num_experts, dtype=torch.float32)

    achievable_top = min(max_probability, 1.0 - min_probability * (num_experts - 1))
    target_top_share = (achievable_top - min_probability) / residual
    target_top_share = max(1.0 / num_experts, min(1.0, target_top_share))

    low_alpha = 0.0
    high_alpha = 128.0
    for _ in range(96):
        alpha = (low_alpha + high_alpha) / 2.0
        weights = torch.pow(shape, alpha)
        weights = weights / weights.sum()
        if float(weights.max().item()) < target_top_share:
            low_alpha = alpha
        else:
            high_alpha = alpha

    weights = torch.pow(shape, high_alpha)
    weights = weights / weights.sum()
    probabilities = min_probability + residual * weights
    probabilities = torch.clamp(probabilities, min=0.0)
    probabilities = probabilities / probabilities.sum()
    return probabilities.float()


def _load_expert_frequency_profile(config: MoePerfConfig) -> torch.Tensor | None:
    if config.expert_frequency_path:
        path = Path(config.expert_frequency_path).expanduser()
        frequency = _load_frequency_file(path, config.num_routed_experts)
        logger.info(f"Loaded expert frequency profile from {path}")
        return frequency

    profile = config.routing_profile.lower()
    if profile in {"", "none", "gate"}:
        return None
    if profile in {"balanced", "uniform"}:
        return torch.ones(config.num_routed_experts, dtype=torch.float32)
    if profile in {"spiky_soft", "soft_spiky", "attention_offload_like"}:
        return _spiky_soft_frequency(
            config.num_routed_experts,
            config.seed,
            config.frequency_max_percent,
            config.frequency_min_percent,
        )
    if profile in {"long_tail_range", "qwen3_long_tail", "qwen3_activity"}:
        return _long_tail_range_frequency(
            config.num_routed_experts,
            config.seed,
            config.frequency_max_percent,
            config.frequency_min_percent,
        )
    if profile in {"synthetic_skew", "skew", "qwen3_proxy"}:
        logger.warning(
            "Using synthetic skew routing profile. For measured Qwen3 routing, set "
            "MOE_PREFILL_OPT_EXPERT_FREQ_PATH to a per-expert frequency file."
        )
        return _long_tail_range_frequency(
            config.num_routed_experts,
            config.seed,
            config.frequency_max_percent,
            config.frequency_min_percent,
        )
    raise ValueError(
        f"Unknown MOE_PREFILL_OPT_ROUTING_PROFILE={config.routing_profile!r}; "
        "use balanced, spiky_soft, synthetic_skew, none, or provide MOE_PREFILL_OPT_EXPERT_FREQ_PATH"
    )


def _frequency_summary(frequency: torch.Tensor | None, limit: int = 8, largest: bool = True) -> str:
    if frequency is None:
        return "gate_logits"
    probs = frequency.float() / frequency.float().sum().clamp_min(1e-12)
    values, indices = torch.topk(probs, min(limit, probs.numel()), largest=largest)
    return ", ".join(f"{int(idx)}:{float(value) * 100:.2f}%" for idx, value in zip(indices, values))


def _selection_frequency_summaries(
    config: MoePerfConfig,
    dispatch_group_size: int,
    routing_frequency: torch.Tensor | None,
) -> tuple[str, str]:
    if routing_frequency is None:
        return "gate_logits", "gate_logits"

    counts = _expert_assignment_counts(config, dispatch_group_size, routing_frequency)
    ratios = counts / counts.sum().clamp_min(1.0)
    return _frequency_summary(ratios, largest=True), _frequency_summary(ratios, largest=False)


def _expert_assignment_counts(
    config: MoePerfConfig,
    dispatch_group_size: int,
    routing_frequency: torch.Tensor | None,
) -> torch.Tensor | None:
    if routing_frequency is None:
        return None

    num_tokens = dispatch_group_size * config.seq_len_per_chip
    indices, _ = sample_frequency_routing(
        routing_frequency,
        num_tokens=num_tokens,
        topk=config.num_experts_per_tok,
        seed=config.seed + num_tokens * 1009 + config.num_experts_per_tok,
    )
    return torch.bincount(indices.flatten(), minlength=config.num_routed_experts).float()


def _expert_ffn_rows_from_counts(config: MoePerfConfig, dispatch_group_size: int, counts: torch.Tensor) -> torch.Tensor:
    if config.dynamic_expert_capacity:
        rows = ((counts.to(torch.int64) + _TILE - 1) // _TILE) * _TILE
        rows[counts == 0] = 0
        return rows.float()
    return torch.full_like(counts, float(dispatch_group_size * config.seq_len_per_chip))


def _estimated_routed_ffn_rows(
    config: MoePerfConfig,
    dispatch_group_size: int,
    routing_frequency: torch.Tensor | None,
) -> float:
    tokens = dispatch_group_size * config.seq_len_per_chip
    routed_assignments = tokens * config.num_experts_per_tok
    if not config.dynamic_expert_capacity:
        return float(config.num_routed_experts * tokens)
    if routing_frequency is None:
        return float(routed_assignments)

    counts = _expert_assignment_counts(config, dispatch_group_size, routing_frequency)
    return float(_expert_ffn_rows_from_counts(config, dispatch_group_size, counts).sum().item())


def _plot_expert_counts_and_roofline(
    config: MoePerfConfig,
    dispatch_group_size: int,
    routing_frequency: torch.Tensor | None,
    durations_s: list[float],
    result: dict,
    expert_times_s: list[float] | None = None,
) -> str:
    counts = _expert_assignment_counts(config, dispatch_group_size, routing_frequency)
    if counts is None:
        logger.warning("Skipping expert plot: no frequency routing profile is available")
        return ""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning(f"Skipping expert plot: matplotlib is unavailable ({exc})")
        return ""

    plot_dir = Path(os.getenv("MOE_PREFILL_OPT_PLOT_DIR", str(Path(__file__).resolve().with_name("plots"))))
    plot_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in result["name"])
    plot_path = plot_dir / f"{safe_name}_{result['mesh_shape']}_expert_counts_roofline.png"

    avg_s = statistics.mean(durations_s)
    rows = _expert_ffn_rows_from_counts(config, dispatch_group_size, counts)
    act_b = _dtype_nbytes(config.routed_activations_dtype)
    weight_b = _dtype_nbytes(config.routed_weights_dtype)
    expert_flops = 6.0 * rows * config.emb_dim * config.hidden_dim
    expert_bytes = rows * (2 * config.emb_dim + 3 * config.hidden_dim) * act_b
    expert_bytes += 3 * config.emb_dim * config.hidden_dim * weight_b
    intensity = expert_flops / expert_bytes.clamp_min(1.0)

    expert_times = torch.zeros(config.num_routed_experts, dtype=torch.float32)
    if expert_times_s:
        timing_values = torch.tensor(expert_times_s[: config.num_routed_experts], dtype=torch.float32)
        expert_times[: timing_values.numel()] = timing_values
    active_mask = rows > 0
    timed_mask = active_mask & (expert_times > 0)

    effective_tflops_s = torch.zeros_like(expert_flops)
    if bool(timed_mask.any()):
        effective_tflops_s[timed_mask] = expert_flops[timed_mask] / expert_times[timed_mask].clamp_min(1e-12) / 1e12
        scatter_mask = timed_mask
        result["expert_timing_source"] = result.get("expert_timing_source") or "per_expert_ffn_sync"
    else:
        logger.warning("Expert timing is unavailable; roofline falls back to measured MoE latency")
        effective_tflops_s[active_mask] = expert_flops[active_mask] / max(avg_s, 1e-12) / 1e12
        scatter_mask = active_mask
        result["expert_timing_source"] = "moe_latency_fallback"

    expert_ids = torch.arange(config.num_routed_experts)
    ratios = counts / counts.sum().clamp_min(1.0) * 100.0

    result["expert_timing_active_experts"] = int(timed_mask.sum().item())
    result["expert_timing_total_ffn_ms"] = float(expert_times.sum().item() * 1000.0)

    max_tflops_idx = None
    min_tflops_idx = None
    if bool(scatter_mask.any()):
        valid_indices = torch.nonzero(scatter_mask, as_tuple=False).flatten()
        valid_tflops = effective_tflops_s[valid_indices]
        max_tflops_idx = int(valid_indices[int(torch.argmax(valid_tflops).item())].item())
        min_tflops_idx = int(valid_indices[int(torch.argmin(valid_tflops).item())].item())
        result["max_expert_tflops_expert"] = max_tflops_idx
        result["max_expert_tflops_per_s"] = float(effective_tflops_s[max_tflops_idx].item())
        result["min_expert_tflops_expert"] = min_tflops_idx
        result["min_expert_tflops_per_s"] = float(effective_tflops_s[min_tflops_idx].item())
    else:
        result["max_expert_tflops_expert"] = ""
        result["max_expert_tflops_per_s"] = 0.0
        result["min_expert_tflops_expert"] = ""
        result["min_expert_tflops_per_s"] = 0.0

    fig, (ax_counts, ax_roofline) = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)

    ax_counts.bar(expert_ids.tolist(), counts.tolist(), color="#2563eb", alpha=0.78)
    ax_counts.set_title("Expert Routed Assignments")
    ax_counts.set_xlabel("Expert ID")
    ax_counts.set_ylabel("Top-k assignment count")
    ax_counts.grid(axis="y", alpha=0.25)

    ax_ratio = ax_counts.twinx()
    ax_ratio.plot(expert_ids.tolist(), ratios.tolist(), color="#dc2626", linewidth=1.2)
    ax_ratio.set_ylabel("Selected ratio (%)")

    top_values, top_indices = torch.topk(counts, min(6, counts.numel()), largest=True)
    bottom_values, bottom_indices = torch.topk(counts, min(6, counts.numel()), largest=False)
    summary_text = [
        "top: "
        + ", ".join(f"E{int(idx)}={int(value)} ({float(ratios[int(idx)]):.2f}%)" for value, idx in zip(top_values, top_indices)),
        "bottom: "
        + ", ".join(
            f"E{int(idx)}={int(value)} ({float(ratios[int(idx)]):.2f}%)"
            for value, idx in zip(bottom_values, bottom_indices)
        ),
    ]
    ax_counts.text(
        0.01,
        0.98,
        "\n".join(summary_text),
        transform=ax_counts.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#d4d4d8"},
    )

    if bool(scatter_mask.any()):
        scatter = ax_roofline.scatter(
            intensity[scatter_mask].tolist(),
            effective_tflops_s[scatter_mask].tolist(),
            c=counts[scatter_mask].tolist(),
            cmap="viridis",
            s=34,
            alpha=0.88,
            edgecolors="none",
        )
        fig.colorbar(scatter, ax=ax_roofline, label="Assignment count")

        x_min = max(float(intensity[scatter_mask].min().item()) * 0.7, 1e-4)
        x_max = max(float(intensity[scatter_mask].max().item()) * 1.5, x_min * 10)
        roof_x = torch.logspace(math.log10(x_min), math.log10(x_max), 200)

        peak_tflops = _env_float("MOE_PREFILL_OPT_ROOFLINE_PEAK_TFLOPS", 680.0)
        peak_bw_gb_s = _env_float("MOE_PREFILL_OPT_ROOFLINE_PEAK_BW_GB_S", 448.0)
        if peak_tflops > 0 and peak_bw_gb_s > 0:
            roof_label = "configured roofline"
        else:
            peak_tflops = max(float(result["estimated_tflops_per_s"]), float(effective_tflops_s[scatter_mask].max().item())) * 1.25
            peak_bw_gb_s = max(float(result["estimated_memory_bw_gb_s"]), 1e-9)
            roof_label = "run envelope"

        roof_y = torch.minimum(torch.full_like(roof_x, peak_tflops), roof_x * peak_bw_gb_s / 1000.0)
        ax_roofline.plot(roof_x.tolist(), roof_y.tolist(), color="#111827", linewidth=1.5, label=roof_label)

        for idx in top_indices[:4].tolist():
            if scatter_mask[idx]:
                ax_roofline.annotate(
                    f"E{idx}",
                    (float(intensity[idx].item()), float(effective_tflops_s[idx].item())),
                    fontsize=8,
                    xytext=(4, 4),
                    textcoords="offset points",
                )

        if max_tflops_idx is not None:
            ax_roofline.scatter(
                [float(intensity[max_tflops_idx].item())],
                [float(effective_tflops_s[max_tflops_idx].item())],
                marker="*",
                s=230,
                color="#f59e0b",
                edgecolors="#111827",
                linewidths=0.8,
                label=f"max TFLOP/s E{max_tflops_idx}",
                zorder=4,
            )
            ax_roofline.annotate(
                f"max E{max_tflops_idx}\n{float(effective_tflops_s[max_tflops_idx].item()):.3f} TF/s",
                (float(intensity[max_tflops_idx].item()), float(effective_tflops_s[max_tflops_idx].item())),
                fontsize=9,
                xytext=(8, 8),
                textcoords="offset points",
                weight="bold",
            )
        if min_tflops_idx is not None and min_tflops_idx != max_tflops_idx:
            ax_roofline.scatter(
                [float(intensity[min_tflops_idx].item())],
                [float(effective_tflops_s[min_tflops_idx].item())],
                marker="X",
                s=140,
                color="#ef4444",
                edgecolors="#111827",
                linewidths=0.8,
                label=f"min TFLOP/s E{min_tflops_idx}",
                zorder=4,
            )
            ax_roofline.annotate(
                f"min E{min_tflops_idx}\n{float(effective_tflops_s[min_tflops_idx].item()):.3f} TF/s",
                (float(intensity[min_tflops_idx].item()), float(effective_tflops_s[min_tflops_idx].item())),
                fontsize=9,
                xytext=(8, -18),
                textcoords="offset points",
                weight="bold",
            )

        timing_lines = [f"timing: {result['expert_timing_source']}"]
        if max_tflops_idx is not None:
            timing_lines.append(
                f"max: E{max_tflops_idx} {float(effective_tflops_s[max_tflops_idx].item()):.3f} TF/s, "
                f"{float(expert_times[max_tflops_idx].item()) * 1000.0:.3f} ms"
            )
        if min_tflops_idx is not None:
            timing_lines.append(
                f"min: E{min_tflops_idx} {float(effective_tflops_s[min_tflops_idx].item()):.3f} TF/s, "
                f"{float(expert_times[min_tflops_idx].item()) * 1000.0:.3f} ms"
            )
        ax_roofline.text(
            0.02,
            0.98,
            "\n".join(timing_lines),
            transform=ax_roofline.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "#d4d4d8"},
        )

        ax_roofline.set_xscale("log")
        ax_roofline.set_yscale("log")

    ax_roofline.set_title("Per-Expert Roofline Position")
    ax_roofline.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax_roofline.set_ylabel("Expert FFN TFLOP/s (per-expert timed)")
    ax_roofline.grid(True, which="both", alpha=0.25)
    ax_roofline.legend(loc="best")

    fig.suptitle(
        f"{result['name']} | mesh={result['mesh_shape']} | seq/chip={config.seq_len_per_chip} | "
        f"topk={config.num_experts_per_tok} | dynamic_capacity={'on' if config.dynamic_expert_capacity else 'off'}",
        fontsize=13,
    )
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return str(plot_path)

def _create_input(config: MoePerfConfig, mesh_device: ttnn.MeshDevice, dispatch_group_size: int) -> ttnn.Tensor:
    torch.manual_seed(config.seed + 1)
    x = torch.randn(dispatch_group_size, config.seq_len_per_chip, config.emb_dim, dtype=torch.bfloat16)
    return ttnn.from_torch(
        x,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(0, -1)),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )


def _estimated_flops(
    config: MoePerfConfig,
    dispatch_group_size: int,
    routing_frequency: torch.Tensor | None,
) -> dict[str, float]:
    tokens = dispatch_group_size * config.seq_len_per_chip
    routed_assignments = tokens * config.num_experts_per_tok
    routed_ffn_rows = _estimated_routed_ffn_rows(config, dispatch_group_size, routing_frequency)
    gate_flops = 2 * tokens * config.emb_dim * config.num_routed_experts
    shared_ffn_flops = 6 * tokens * config.emb_dim * config.hidden_dim if config.use_shared_expert else 0
    routed_ffn_flops = 6 * routed_ffn_rows * config.emb_dim * config.hidden_dim
    reduce_flops = 2 * routed_assignments * config.emb_dim
    total = gate_flops + shared_ffn_flops + routed_ffn_flops + reduce_flops
    return {
        "tokens": float(tokens),
        "routed_assignments": float(routed_assignments),
        "routed_ffn_rows": float(routed_ffn_rows),
        "gate_flops": float(gate_flops),
        "shared_ffn_flops": float(shared_ffn_flops),
        "routed_ffn_flops": float(routed_ffn_flops),
        "reduce_flops": float(reduce_flops),
        "total_flops": float(total),
    }


def _estimated_memory_bytes(
    config: MoePerfConfig,
    dispatch_group_size: int,
    routing_frequency: torch.Tensor | None,
) -> dict[str, float]:
    tokens = dispatch_group_size * config.seq_len_per_chip
    routed_assignments = tokens * config.num_experts_per_tok
    routed_ffn_rows = _estimated_routed_ffn_rows(config, dispatch_group_size, routing_frequency)
    act_b = _dtype_nbytes(config.routed_activations_dtype)
    shared_act_b = _dtype_nbytes(config.shared_activations_dtype)
    routed_w_b = _dtype_nbytes(config.routed_weights_dtype)
    shared_w_b = _dtype_nbytes(config.shared_weights_dtype)
    bf16_b = 2.0

    gate_bytes = (
        tokens * config.emb_dim * bf16_b
        + config.emb_dim * config.num_routed_experts * bf16_b
        + tokens * config.num_routed_experts * bf16_b
    )
    dispatch_combine_bytes = 4 * routed_assignments * config.emb_dim * bf16_b
    routed_ffn_activation_bytes = routed_ffn_rows * (2 * config.emb_dim + 3 * config.hidden_dim) * act_b
    routed_ffn_weight_bytes = config.num_routed_experts * 3 * config.emb_dim * config.hidden_dim * routed_w_b
    shared_ffn_activation_bytes = (
        tokens * (2 * config.emb_dim + 3 * config.hidden_dim) * shared_act_b if config.use_shared_expert else 0
    )
    shared_ffn_weight_bytes = 3 * config.emb_dim * config.hidden_dim * shared_w_b if config.use_shared_expert else 0
    reduce_bytes = (routed_assignments * config.emb_dim + tokens * config.emb_dim) * bf16_b
    total = (
        gate_bytes
        + dispatch_combine_bytes
        + routed_ffn_activation_bytes
        + routed_ffn_weight_bytes
        + shared_ffn_activation_bytes
        + shared_ffn_weight_bytes
        + reduce_bytes
    )
    return {
        "gate_bytes": float(gate_bytes),
        "dispatch_combine_bytes": float(dispatch_combine_bytes),
        "routed_ffn_activation_bytes": float(routed_ffn_activation_bytes),
        "routed_ffn_weight_bytes": float(routed_ffn_weight_bytes),
        "shared_ffn_activation_bytes": float(shared_ffn_activation_bytes),
        "shared_ffn_weight_bytes": float(shared_ffn_weight_bytes),
        "reduce_bytes": float(reduce_bytes),
        "total_bytes": float(total),
    }


def _run_forward_once(tt_moe: TtMoe, tt_x: ttnn.Tensor, mesh_device: ttnn.MeshDevice) -> float:
    ttnn.synchronize_device(mesh_device)
    start = time.perf_counter()
    output, _ = tt_moe(tt_x, return_intermediates=False)
    ttnn.synchronize_device(mesh_device)
    duration_s = time.perf_counter() - start
    ttnn.deallocate(output)
    return duration_s


def _format_expert_ffn_breakdown_summary(breakdowns: list[dict]) -> str:
    if not breakdowns:
        return ""

    def core_summary(grid_key: str, cores_key: str) -> str:
        pairs = sorted(
            {
                (str(item.get(grid_key, "unknown")), int(item.get(cores_key, 0)))
                for item in breakdowns
                if item.get(cores_key, 0)
            },
            key=lambda pair: (pair[1], pair[0]),
        )
        if not pairs:
            return "unknown"
        if len(pairs) == 1:
            grid, cores = pairs[0]
            return f"{grid}({cores})"
        core_values = [cores for _, cores in pairs]
        return f"{min(core_values)}-{max(core_values)} cores"

    duration_keys = ("gate_mm_silu_ms", "up_mm_ms", "mul_ms", "down_mm_ms", "total_ms", "matmul_only_ms")
    flop_keys = ("gate_mm_silu_flops", "up_mm_flops", "down_mm_flops", "total_matmul_flops")
    pass_ids = sorted({int(item.get("iteration", 0)) for item in breakdowns})
    num_iters = max(1, len(pass_ids))
    expert_ids = sorted({int(item.get("expert_id", -1)) for item in breakdowns})

    totals = {key: 0.0 for key in duration_keys}
    flops = {key: 0.0 for key in flop_keys}
    for breakdown in breakdowns:
        for key in duration_keys:
            totals[key] += float(breakdown.get(key, 0.0))
        for key in flop_keys:
            flops[key] += float(breakdown.get(key, 0.0))

    avg_durations = {key: totals[key] / num_iters for key in duration_keys}

    def effective_tflops(flop_key: str, ms_key: str) -> float:
        return flops[flop_key] / max(totals[ms_key] / 1000.0, 1e-12) / 1e12

    total_tflops = flops["total_matmul_flops"] / max(totals["total_ms"] / 1000.0, 1e-12) / 1e12
    device_grid = str(breakdowns[0].get("device_core_grid", "unknown"))
    device_cores = int(breakdowns[0].get("device_compute_cores", 0))
    device_summary = f"{device_grid}({device_cores})" if device_cores else device_grid
    stage_source = str(breakdowns[0].get("stage_source", "unknown"))
    ffn_path = str(breakdowns[0].get("ffn_path", "unknown"))
    matmul_program = str(breakdowns[0].get("matmul_program", "unknown"))
    matmul_factory = str(breakdowns[0].get("matmul_factory", "unknown"))
    return (
        f"source={stage_source} | path={ffn_path} | program={matmul_program} | factory={matmul_factory} | "
        f"experts={len(expert_ids)} | iters={num_iters} | "
        f"avg_total_ms={avg_durations['total_ms']:.3f} | "
        f"avg_gate_mm_silu_ms={avg_durations['gate_mm_silu_ms']:.3f} | "
        f"avg_up_mm_ms={avg_durations['up_mm_ms']:.3f} | "
        f"avg_mul_reshard_ms={avg_durations['mul_ms']:.3f} | "
        f"avg_down_mm_ms={avg_durations['down_mm_ms']:.3f} | "
        f"avg_matmul_tflops=gate={effective_tflops('gate_mm_silu_flops', 'gate_mm_silu_ms'):.3f} "
        f"up={effective_tflops('up_mm_flops', 'up_mm_ms'):.3f} "
        f"down={effective_tflops('down_mm_flops', 'down_mm_ms'):.3f} "
        f"matmul_only={effective_tflops('total_matmul_flops', 'matmul_only_ms'):.3f} "
        f"total={total_tflops:.3f} | "
        f"cores=device={device_summary} | "
        f"gate/up={core_summary('gate_up_core_grid', 'gate_up_cores')} | "
        f"mul={core_summary('mul_core_grid', 'mul_cores')} | "
        f"down={core_summary('down_core_grid', 'down_cores')}"
    )


def _log_expert_ffn_average_breakdowns(breakdowns: list[dict]) -> None:
    by_expert: dict[int, list[dict]] = {}
    for breakdown in breakdowns:
        by_expert.setdefault(int(breakdown.get("expert_id", -1)), []).append(breakdown)

    duration_keys = ("gate_mm_silu_ms", "up_mm_ms", "mul_ms", "down_mm_ms", "total_ms", "matmul_only_ms")
    flop_keys = ("gate_mm_silu_flops", "up_mm_flops", "down_mm_flops", "total_matmul_flops")
    for expert_id, expert_breakdowns in sorted(by_expert.items()):
        samples = len(expert_breakdowns)
        if samples == 0:
            continue
        totals = {key: 0.0 for key in duration_keys}
        flops = {key: 0.0 for key in flop_keys}
        for breakdown in expert_breakdowns:
            for key in duration_keys:
                totals[key] += float(breakdown.get(key, 0.0))
            for key in flop_keys:
                flops[key] += float(breakdown.get(key, 0.0))
        avg = {key: totals[key] / samples for key in duration_keys}

        def effective_tflops(flop_key: str, ms_key: str) -> float:
            return flops[flop_key] / max(totals[ms_key] / 1000.0, 1e-12) / 1e12

        total_tflops = flops["total_matmul_flops"] / max(totals["total_ms"] / 1000.0, 1e-12) / 1e12
        first = expert_breakdowns[0]
        logger.info(
            "EXPERT_FFN_BREAKDOWN_AVG "
            f"E{expert_id} samples={samples} rows={int(first.get('rows', 0))} "
            f"avg_gate_mm_silu={avg['gate_mm_silu_ms']:.3f}ms "
            f"avg_up_mm={avg['up_mm_ms']:.3f}ms "
            f"avg_mul_reshard={avg['mul_ms']:.3f}ms "
            f"avg_down_mm={avg['down_mm_ms']:.3f}ms "
            f"source={first.get('stage_source', 'unknown')} "
            f"path={first.get('ffn_path', 'unknown')} "
            f"program={first.get('matmul_program', 'unknown')} "
            f"factory={first.get('matmul_factory', 'unknown')} "
            f"avg_total={avg['total_ms']:.3f}ms "
            f"avg_matmul_tflops=gate={effective_tflops('gate_mm_silu_flops', 'gate_mm_silu_ms'):.3f} "
            f"up={effective_tflops('up_mm_flops', 'up_mm_ms'):.3f} "
            f"down={effective_tflops('down_mm_flops', 'down_mm_ms'):.3f} "
            f"matmul_only={effective_tflops('total_matmul_flops', 'matmul_only_ms'):.3f} "
            f"total={total_tflops:.3f}"
        )

def _create_direct_expert_input(config: MoePerfConfig, mesh_device: ttnn.MeshDevice, rows: int) -> ttnn.Tensor:
    return ttnn.zeros(
        [rows, config.emb_dim],
        dtype=config.routed_activations_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _measure_expert_ffn_fused_opt(
    config: MoePerfConfig,
    tt_moe: TtMoe,
    mesh_device: ttnn.MeshDevice,
    dispatch_group_size: int,
    routing_frequency: torch.Tensor | None,
) -> tuple[list[float], float, str]:
    routed_expert = getattr(tt_moe, "routed_expert", None)
    if routed_expert is None or not hasattr(routed_expert, "_expert_ffn"):
        logger.warning("Skipping fused expert FFN timing: routed expert hook is unavailable")
        return [], 0.0, ""

    counts = _expert_assignment_counts(config, dispatch_group_size, routing_frequency)
    if counts is None:
        counts = torch.full(
            (config.num_routed_experts,),
            float(dispatch_group_size * config.seq_len_per_chip * config.num_experts_per_tok) / config.num_routed_experts,
        )
    rows_by_expert = _expert_ffn_rows_from_counts(config, dispatch_group_size, counts).to(torch.int64)

    op_name = (
        "routed_expert_ffn_opt"
        if hasattr(ttnn.experimental.deepseek_prefill, "routed_expert_ffn_opt")
        else "routed_expert_ffn"
    )
    iterations = max(1, int(config.expert_ffn_breakdown_iterations))
    warmup_iterations = max(0, int(config.expert_ffn_breakdown_warmup_iterations))
    timing_sums = [0.0 for _ in range(config.num_routed_experts)]
    timing_counts = [0 for _ in range(config.num_routed_experts)]

    previous_breakdown = getattr(routed_expert, "breakdown_expert_ffn", False)
    previous_measure = getattr(routed_expert, "measure_expert_time", False)
    routed_expert.breakdown_expert_ffn = False
    routed_expert.measure_expert_time = False

    total_pass_s = 0.0
    try:
        for warmup_idx in range(warmup_iterations):
            logger.info(f"MoE opt fused {op_name} warmup {warmup_idx + 1}/{warmup_iterations}")
            for expert_id in range(min(config.num_routed_experts, len(routed_expert.gate_projs))):
                rows = int(rows_by_expert[expert_id].item())
                if rows <= 0:
                    continue
                tt_input = _create_direct_expert_input(config, mesh_device, rows)
                ttnn.synchronize_device(mesh_device)
                output = routed_expert._expert_ffn(
                    tt_input,
                    routed_expert.gate_projs[expert_id],
                    routed_expert.up_projs[expert_id],
                    routed_expert.down_projs[expert_id],
                    out=None,
                    expert_id=expert_id,
                )
                ttnn.synchronize_device(mesh_device)
                ttnn.deallocate(output)
                ttnn.deallocate(tt_input)

        ttnn.synchronize_device(mesh_device)
        pass_start_s = time.perf_counter()
        for iteration in range(iterations):
            logger.info(f"MoE opt fused {op_name} pass {iteration + 1}/{iterations}")
            for expert_id in range(min(config.num_routed_experts, len(routed_expert.gate_projs))):
                rows = int(rows_by_expert[expert_id].item())
                if rows <= 0:
                    continue
                tt_input = _create_direct_expert_input(config, mesh_device, rows)
                ttnn.synchronize_device(mesh_device)
                start_s = time.perf_counter()
                output = routed_expert._expert_ffn(
                    tt_input,
                    routed_expert.gate_projs[expert_id],
                    routed_expert.up_projs[expert_id],
                    routed_expert.down_projs[expert_id],
                    out=None,
                    expert_id=expert_id,
                )
                ttnn.synchronize_device(mesh_device)
                elapsed_s = time.perf_counter() - start_s
                timing_sums[expert_id] += elapsed_s
                timing_counts[expert_id] += 1
                ttnn.deallocate(output)
                ttnn.deallocate(tt_input)
        ttnn.synchronize_device(mesh_device)
        total_pass_s = time.perf_counter() - pass_start_s
    finally:
        ttnn.synchronize_device(mesh_device)
        routed_expert.breakdown_expert_ffn = previous_breakdown
        routed_expert.measure_expert_time = previous_measure

    timings = [
        timing_sums[idx] / timing_counts[idx] if timing_counts[idx] else 0.0
        for idx in range(config.num_routed_experts)
    ]
    avg_pass_s = total_pass_s / iterations
    timed_total_s = sum(timing_sums) / iterations
    active_experts = sum(1 for count in timing_counts if count)
    rows_total = int(sum(int(rows_by_expert[idx].item()) for idx, count in enumerate(timing_counts) if count))
    total_flops = 3.0 * 2.0 * rows_total * config.emb_dim * config.hidden_dim
    tflops_per_s = total_flops / max(timed_total_s, 1e-12) / 1e12
    summary = (
        f"op={op_name} | experts={active_experts} | iters={iterations} | "
        f"avg_op_ms={timed_total_s * 1000.0:.3f} | "
        f"avg_pass_ms={avg_pass_s * 1000.0:.3f} | "
        f"rows={rows_total} | tflops={tflops_per_s:.3f}"
    )
    logger.info("EXPERT_FFN_FUSED_OP " + summary)
    return timings, avg_pass_s, summary


def _measure_expert_ffn_direct_breakdown(
    config: MoePerfConfig,
    tt_moe: TtMoe,
    mesh_device: ttnn.MeshDevice,
    dispatch_group_size: int,
    routing_frequency: torch.Tensor | None,
) -> tuple[list[float], float, str]:
    routed_expert = getattr(tt_moe, "routed_expert", None)
    if routed_expert is None or not hasattr(routed_expert, "_expert_ffn"):
        logger.warning("Skipping direct expert FFN breakdown: routed expert hook is unavailable")
        return [], 0.0, ""

    counts = _expert_assignment_counts(config, dispatch_group_size, routing_frequency)
    if counts is None:
        counts = torch.full(
            (config.num_routed_experts,),
            float(dispatch_group_size * config.seq_len_per_chip * config.num_experts_per_tok) / config.num_routed_experts,
        )
    rows_by_expert = _expert_ffn_rows_from_counts(config, dispatch_group_size, counts).to(torch.int64)

    iterations = max(1, int(config.expert_ffn_breakdown_iterations))
    timing_sums = [0.0 for _ in range(config.num_routed_experts)]
    timing_counts = [0 for _ in range(config.num_routed_experts)]
    previous_breakdown = getattr(routed_expert, "breakdown_expert_ffn", False)
    previous_record_breakdown = getattr(routed_expert, "record_expert_ffn_breakdown", True)
    previous_measure = getattr(routed_expert, "measure_expert_time", False)
    routed_expert.breakdown_expert_ffn = True
    routed_expert.measure_expert_time = False
    routed_expert.last_expert_ffn_breakdowns = []

    ttnn.synchronize_device(mesh_device)
    total_pass_s = 0.0
    try:
        warmup_iterations = max(0, int(config.expert_ffn_breakdown_warmup_iterations))
        routed_expert.record_expert_ffn_breakdown = False
        for warmup_idx in range(warmup_iterations):
            logger.info(f"MoE opt direct _expert_ffn breakdown warmup {warmup_idx + 1}/{warmup_iterations}")
            for expert_id in range(min(config.num_routed_experts, len(routed_expert.gate_projs))):
                rows = int(rows_by_expert[expert_id].item())
                if rows <= 0:
                    continue
                tt_input = _create_direct_expert_input(config, mesh_device, rows)
                ttnn.synchronize_device(mesh_device)
                output = routed_expert._expert_ffn(
                    tt_input,
                    routed_expert.gate_projs[expert_id],
                    routed_expert.up_projs[expert_id],
                    routed_expert.down_projs[expert_id],
                    out=None,
                    expert_id=expert_id,
                )
                ttnn.synchronize_device(mesh_device)
                ttnn.deallocate(output)
                ttnn.deallocate(tt_input)

        routed_expert.record_expert_ffn_breakdown = True
        routed_expert.last_expert_ffn_breakdowns = []
        ttnn.synchronize_device(mesh_device)
        pass_start_s = time.perf_counter()
        for iteration in range(iterations):
            logger.info(f"MoE opt direct _expert_ffn breakdown pass {iteration + 1}/{iterations}")
            for expert_id in range(min(config.num_routed_experts, len(routed_expert.gate_projs))):
                rows = int(rows_by_expert[expert_id].item())
                if rows <= 0:
                    continue
                tt_input = _create_direct_expert_input(config, mesh_device, rows)
                ttnn.synchronize_device(mesh_device)
                output = routed_expert._expert_ffn(
                    tt_input,
                    routed_expert.gate_projs[expert_id],
                    routed_expert.up_projs[expert_id],
                    routed_expert.down_projs[expert_id],
                    out=None,
                    expert_id=expert_id,
                )
                ttnn.synchronize_device(mesh_device)
                if routed_expert.last_expert_ffn_breakdowns:
                    breakdown = routed_expert.last_expert_ffn_breakdowns[-1]
                    breakdown["iteration"] = iteration
                    timing_sums[expert_id] += float(breakdown["total_ms"]) / 1000.0
                    timing_counts[expert_id] += 1
                ttnn.deallocate(output)
                ttnn.deallocate(tt_input)
        ttnn.synchronize_device(mesh_device)
        total_pass_s = time.perf_counter() - pass_start_s
    finally:
        ttnn.synchronize_device(mesh_device)
        routed_expert.breakdown_expert_ffn = previous_breakdown
        routed_expert.record_expert_ffn_breakdown = previous_record_breakdown
        routed_expert.measure_expert_time = previous_measure

    timings = [
        timing_sums[idx] / timing_counts[idx] if timing_counts[idx] else 0.0
        for idx in range(config.num_routed_experts)
    ]
    _log_expert_ffn_average_breakdowns(routed_expert.last_expert_ffn_breakdowns)
    avg_pass_s = total_pass_s / iterations
    return timings, avg_pass_s, _format_expert_ffn_breakdown_summary(routed_expert.last_expert_ffn_breakdowns)


def _run_direct_expert_ffn_once(
    routed_expert,
    config: MoePerfConfig,
    mesh_device: ttnn.MeshDevice,
    expert_id: int,
    rows: int,
):
    tt_input = _create_direct_expert_input(config, mesh_device, rows)
    ttnn.synchronize_device(mesh_device)
    output = routed_expert._expert_ffn(
        tt_input,
        routed_expert.gate_projs[expert_id],
        routed_expert.up_projs[expert_id],
        routed_expert.down_projs[expert_id],
        out=None,
        expert_id=expert_id,
    )
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(output)
    ttnn.deallocate(tt_input)


@dataclass(frozen=True)
class _ExpertSubDeviceGroup:
    sub_device_id: ttnn.SubDeviceId
    cores: ttnn.CoreRangeSet
    core_grid: ttnn.CoreGrid
    expert_id: int
    token_count: int
    ffn_rows: int
    core_rows: int
    start_y: int


@dataclass(frozen=True)
class _ExpertSubDeviceLaneGroup:
    sub_device_id: ttnn.SubDeviceId
    cores: ttnn.CoreRangeSet
    core_grid: ttnn.CoreGrid
    core_rows: int
    start_y: int
    assignments: tuple["_ExpertSubDeviceAssignment", ...]


@dataclass(frozen=True)
class _ExpertSubDeviceAssignment:
    expert_id: int
    token_count: int
    ffn_rows: int
    core_rows: int


@dataclass(frozen=True)
class _ExpertSubDeviceWave:
    assignments: tuple[_ExpertSubDeviceAssignment, ...]
    total_core_rows: int
    total_subdevices: int
    total_tokens: int
    total_ffn_rows: int


@dataclass(frozen=True)
class _ExpertSubDeviceLane:
    core_rows: int
    assignments: tuple[_ExpertSubDeviceAssignment, ...]
    total_tokens: int
    total_ffn_rows: int


@dataclass(frozen=True)
class _ExpertSubDevicePhase:
    lanes: tuple[_ExpertSubDeviceLane, ...]
    total_core_rows: int
    total_subdevices: int
    total_tokens: int
    total_ffn_rows: int


def _core_range_set(cols: int, rows: int, start_y: int = 0) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, start_y),
                ttnn.CoreCoord(cols - 1, start_y + rows - 1),
            )
        }
    )


def _allocate_direct_expert_input(config: MoePerfConfig, mesh_device: ttnn.MeshDevice, rows: int) -> ttnn.Tensor:
    return ttnn.empty(
        ttnn.Shape((rows, config.emb_dim)),
        dtype=config.routed_activations_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _forget_live_tensor(live_tensors: list[ttnn.Tensor], tensor: ttnn.Tensor) -> None:
    for idx in range(len(live_tensors) - 1, -1, -1):
        if live_tensors[idx] is tensor:
            live_tensors.pop(idx)
            return


def _deallocate_tracked_tensor(live_tensors: list[ttnn.Tensor], tensor: ttnn.Tensor) -> None:
    _forget_live_tensor(live_tensors, tensor)
    ttnn.deallocate(tensor)


def _routed_expert_ffn_opt_supports_subdevice() -> bool:
    fused_op = getattr(ttnn.experimental.deepseek_prefill, "routed_expert_ffn_opt", None)
    return fused_op is not None and "sub_device_id" in str(getattr(fused_op, "__doc__", ""))


def _routed_expert_ffn_opt_available() -> bool:
    return getattr(ttnn.experimental.deepseek_prefill, "routed_expert_ffn_opt", None) is not None


def _sub_device_id_to_int(sub_device_id) -> int:
    try:
        return int(sub_device_id)
    except Exception:
        text = str(sub_device_id)
        digits = "".join(ch for ch in text if ch.isdigit())
        return int(digits) if digits else -1


def _rt_profiler_inactive_hint() -> str:
    hints = []
    pci_root = Path("/sys/bus/pci/devices")
    try:
        for vendor_path in pci_root.glob("*/vendor"):
            try:
                if vendor_path.read_text().strip().lower() != "0x1e52":
                    continue
                device_dir = vendor_path.parent
                iommu_type_path = device_dir / "iommu_group" / "type"
                iommu_type = iommu_type_path.read_text().strip() if iommu_type_path.exists() else "missing"
                hints.append(f"TT PCI {device_dir.name} iommu_group/type={iommu_type}")
            except OSError:
                continue
    except OSError:
        pass

    if hints:
        return "; ".join(hints)
    return "no Tenstorrent PCI IOMMU hint found"


class _ExpertSubDeviceRtProfiler:
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.summary_csv_path = self.csv_path.with_name(f"{self.csv_path.stem}_summary{self.csv_path.suffix}")
        self.active = False
        self.current_iteration = -1
        self.current_phase = -1
        self._handle = None
        self._lock = threading.Lock()
        self._records: list[dict] = []
        self._issues: list[dict] = []
        self._issue_index = 0
        self.status = "not_started"
        self.inactive_reason = ""

    def start(self) -> bool:
        try:
            if not ttnn.device.IsProgramRealtimeProfilerActive():
                self.status = "inactive"
                self.inactive_reason = (
                    "IsProgramRealtimeProfilerActive returned false; " + _rt_profiler_inactive_hint()
                )
                logger.warning(
                    "Expert sub-device RT profile requested, but the real-time profiler is not active on this device"
                )
                return False
        except Exception:
            self.status = "state_query_failed"
            self.inactive_reason = "failed to query IsProgramRealtimeProfilerActive"
            logger.warning("Failed to query real-time profiler state", exc_info=True)
            return False

        def collect(record):
            with self._lock:
                self._records.append(
                    {
                        "program_id": int(record.program_id),
                        "chip_id": int(record.chip_id),
                        "start_timestamp": int(record.start_timestamp),
                        "end_timestamp": int(record.end_timestamp),
                        "frequency_cycles_per_ns": float(record.frequency),
                        "kernel_sources": "|".join(str(source) for source in record.kernel_sources),
                    }
                )

        try:
            self._handle = ttnn.device.RegisterProgramRealtimeProfilerCallback(collect)
            self.active = True
            self.status = "active"
            self.inactive_reason = ""
            return True
        except Exception:
            self.status = "callback_registration_failed"
            self.inactive_reason = "RegisterProgramRealtimeProfilerCallback failed"
            logger.warning("Failed to register real-time profiler callback", exc_info=True)
            self._handle = None
            self.active = False
            return False

    def stop(self) -> None:
        if self._handle is None:
            self.active = False
            return
        # Let the D2H receiver thread drain records from the final synchronize.
        time.sleep(0.2)
        try:
            ttnn.device.UnregisterProgramRealtimeProfilerCallback(self._handle)
        except Exception:
            logger.warning("Failed to unregister real-time profiler callback", exc_info=True)
        finally:
            self._handle = None
            self.active = False

    def set_context(self, iteration: int, phase: int) -> None:
        self.current_iteration = int(iteration)
        self.current_phase = int(phase)

    def record_issue(
        self,
        stage: str,
        lane_depth: int,
        lane_idx: int,
        group: "_ExpertSubDeviceLaneGroup",
        assignment: "_ExpertSubDeviceAssignment",
    ) -> None:
        if not self.active:
            return
        self._issues.append(
            {
                "issue_index": self._issue_index,
                "iteration": self.current_iteration,
                "phase": self.current_phase,
                "lane_depth": int(lane_depth),
                "lane": int(lane_idx),
                "sub_device_id": _sub_device_id_to_int(group.sub_device_id),
                "stage": stage,
                "expert_id": int(assignment.expert_id),
                "token_count": int(assignment.token_count),
                "ffn_rows": int(assignment.ffn_rows),
                "core_rows": int(group.core_rows),
                "start_y": int(group.start_y),
            }
        )
        self._issue_index += 1

    def _matched_rows(self) -> list[dict]:
        with self._lock:
            records = list(self._records)
        issues = list(self._issues)
        records.sort(key=lambda row: (row["program_id"], row["start_timestamp"], row["end_timestamp"]))
        rows = []
        for record_idx, record in enumerate(records):
            issue = issues[record_idx] if record_idx < len(issues) else {}
            frequency = record["frequency_cycles_per_ns"]
            duration_ns = (
                (record["end_timestamp"] - record["start_timestamp"]) / frequency if frequency > 0.0 else 0.0
            )
            rows.append(
                {
                    **record,
                    "duration_ns": duration_ns,
                    "matched_issue": bool(issue),
                    "issue_index": issue.get("issue_index", ""),
                    "iteration": issue.get("iteration", ""),
                    "phase": issue.get("phase", ""),
                    "lane_depth": issue.get("lane_depth", ""),
                    "lane": issue.get("lane", ""),
                    "sub_device_id": issue.get("sub_device_id", ""),
                    "stage": issue.get("stage", ""),
                    "expert_id": issue.get("expert_id", ""),
                    "token_count": issue.get("token_count", ""),
                    "ffn_rows": issue.get("ffn_rows", ""),
                    "core_rows": issue.get("core_rows", ""),
                    "start_y": issue.get("start_y", ""),
                }
            )
        return rows

    @staticmethod
    def _overlap_summary(rows: list[dict]) -> list[dict]:
        grouped: dict[tuple[int, int], list[dict]] = {}
        for row in rows:
            if not row["matched_issue"] or row["iteration"] == "" or row["phase"] == "" or row["sub_device_id"] == "":
                continue
            grouped.setdefault((int(row["iteration"]), int(row["phase"])), []).append(row)

        summaries = []
        for (iteration, phase), group_rows in sorted(grouped.items()):
            overlap_pairs = 0
            for idx, lhs in enumerate(group_rows):
                lhs_start = int(lhs["start_timestamp"])
                lhs_end = int(lhs["end_timestamp"])
                lhs_sub_device = int(lhs["sub_device_id"])
                for rhs in group_rows[idx + 1 :]:
                    if lhs_sub_device == int(rhs["sub_device_id"]):
                        continue
                    rhs_start = int(rhs["start_timestamp"])
                    rhs_end = int(rhs["end_timestamp"])
                    if lhs_start < rhs_end and rhs_start < lhs_end:
                        overlap_pairs += 1

            events = []
            for row in group_rows:
                events.append((int(row["start_timestamp"]), 1, int(row["sub_device_id"])))
                events.append((int(row["end_timestamp"]), 0, int(row["sub_device_id"])))
            events.sort()

            active_by_sub_device: dict[int, int] = {}
            active_programs = 0
            max_concurrent_programs = 0
            max_concurrent_subdevices = 0
            for _, event_type, sub_device_id in events:
                if event_type == 0:
                    active_by_sub_device[sub_device_id] = max(0, active_by_sub_device.get(sub_device_id, 0) - 1)
                    if active_by_sub_device[sub_device_id] == 0:
                        active_by_sub_device.pop(sub_device_id, None)
                    active_programs = max(0, active_programs - 1)
                else:
                    active_by_sub_device[sub_device_id] = active_by_sub_device.get(sub_device_id, 0) + 1
                    active_programs += 1
                    max_concurrent_programs = max(max_concurrent_programs, active_programs)
                    max_concurrent_subdevices = max(max_concurrent_subdevices, len(active_by_sub_device))

            summaries.append(
                {
                    "iteration": iteration,
                    "phase": phase,
                    "program_records": len(group_rows),
                    "unique_subdevices": len({int(row["sub_device_id"]) for row in group_rows}),
                    "overlap_pairs": overlap_pairs,
                    "max_concurrent_programs": max_concurrent_programs,
                    "max_concurrent_subdevices": max_concurrent_subdevices,
                    "window_start_timestamp": min(int(row["start_timestamp"]) for row in group_rows),
                    "window_end_timestamp": max(int(row["end_timestamp"]) for row in group_rows),
                }
            )
        return summaries

    def write_reports(self) -> str:
        rows = self._matched_rows()
        summaries = self._overlap_summary(rows)
        with self._lock:
            record_count = len(self._records)
        issue_count = len(self._issues)
        matched_count = min(record_count, issue_count)

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        detail_fieldnames = list(rows[0].keys()) if rows else [
            "program_id",
            "chip_id",
            "start_timestamp",
            "end_timestamp",
            "frequency_cycles_per_ns",
            "kernel_sources",
            "duration_ns",
            "matched_issue",
            "issue_index",
            "iteration",
            "phase",
            "lane_depth",
            "lane",
            "sub_device_id",
            "stage",
            "expert_id",
            "token_count",
            "ffn_rows",
            "core_rows",
            "start_y",
        ]
        with self.csv_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=detail_fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        summary_fieldnames = [
            "status",
            "inactive_reason",
            "record_count",
            "issue_count",
            "matched_count",
            "iteration",
            "phase",
            "program_records",
            "unique_subdevices",
            "overlap_pairs",
            "max_concurrent_programs",
            "max_concurrent_subdevices",
            "window_start_timestamp",
            "window_end_timestamp",
        ]
        summary_rows = [
            {
                "status": self.status,
                "inactive_reason": self.inactive_reason,
                "record_count": record_count,
                "issue_count": issue_count,
                "matched_count": matched_count,
                **summary,
            }
            for summary in summaries
        ]
        if not summary_rows:
            summary_rows = [
                {
                    "status": self.status,
                    "inactive_reason": self.inactive_reason,
                    "record_count": record_count,
                    "issue_count": issue_count,
                    "matched_count": matched_count,
                    "iteration": "",
                    "phase": "",
                    "program_records": 0,
                    "unique_subdevices": 0,
                    "overlap_pairs": 0,
                    "max_concurrent_programs": 0,
                    "max_concurrent_subdevices": 0,
                    "window_start_timestamp": "",
                    "window_end_timestamp": "",
                }
            ]
        with self.summary_csv_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=summary_fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

        max_concurrent_subdevices = max(
            (summary["max_concurrent_subdevices"] for summary in summaries),
            default=0,
        )
        overlap_pairs = sum(summary["overlap_pairs"] for summary in summaries)
        reason = f",reason={self.inactive_reason}" if self.inactive_reason else ""
        return (
            f"rt_profile=status={self.status}{reason},records={record_count},issues={issue_count},matched={matched_count},"
            f"max_concurrent_subdevices={max_concurrent_subdevices},overlap_pairs={overlap_pairs},"
            f"csv={self.csv_path},summary_csv={self.summary_csv_path}"
        )


def _allocate_tracked_expert_stage_output(
    input_tensor: ttnn.Tensor,
    rows: int,
    width: int,
    memory_config: ttnn.MemoryConfig,
    live_tensors: list[ttnn.Tensor],
) -> ttnn.Tensor:
    output = ttnn.empty(
        ttnn.Shape((rows, width)),
        dtype=input_tensor.dtype,
        layout=ttnn.TILE_LAYOUT,
        device=input_tensor.device(),
        memory_config=memory_config,
    )
    live_tensors.append(output)
    return output


def _div_up_int(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _largest_divisor(value: int, max_divisor: int) -> int:
    for candidate in range(max(1, min(value, max_divisor)), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def _subdevice_block_sharded_memory_config(
    cols: int,
    rows: int,
    start_y: int,
    shard_shape: tuple[int, int],
) -> ttnn.MemoryConfig:
    shard_spec = ttnn.ShardSpec(
        _core_range_set(cols, rows, start_y=start_y),
        list(shard_shape),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard_spec)


def _is_routed_expert_small_m(rows: int) -> bool:
    tile = int(ttnn.TILE_SIZE)
    m_tiles = max(1, _div_up_int(rows, tile))
    return m_tiles <= _ROUTED_EXPERT_SMALL_M_TILE_THRESHOLD


def _routed_expert_gate_up_stage_kwargs(
    group,
    rows: int,
    n_dim: int,
    fused_activation=None,
) -> dict:
    if not _is_routed_expert_small_m(rows):
        return {"core_grid": group.core_grid}

    tile = int(ttnn.TILE_SIZE)
    grid_x_max = min(11, int(group.core_grid.x))
    m_tiles = max(1, _div_up_int(rows, tile))
    n_tiles = max(1, _div_up_int(n_dim, tile))
    grid_y = _routed_expert_small_m_grid_y(m_tiles)
    per_core_m = _div_up_int(m_tiles, grid_y)
    per_core_n = _div_up_int(n_tiles, grid_x_max)
    grid_x = max(1, min(grid_x_max, _div_up_int(n_tiles, per_core_n)))
    allowed_worker_cores = _core_range_set(grid_x, grid_y)
    program_config_kwargs = {
        "compute_with_storage_grid_size": ttnn.CoreCoord(grid_x, grid_y),
        "in0_block_w": 16,
        "out_subblock_h": 1,
        "out_subblock_w": per_core_n,
        "out_block_h": per_core_m,
        "out_block_w": per_core_n,
        "per_core_M": per_core_m,
        "per_core_N": per_core_n,
        "transpose_mcast": False,
        "fuse_batch": False,
        "allowed_worker_cores": allowed_worker_cores,
    }
    if fused_activation is not None:
        program_config_kwargs["fused_activation"] = fused_activation

    return {
        "memory_config": _subdevice_block_sharded_memory_config(
            grid_x,
            grid_y,
            group.start_y,
            (per_core_m * tile, per_core_n * tile),
        ),
        "program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(**program_config_kwargs),
    }


def _routed_expert_gate_stage(
    input_tensor: ttnn.Tensor,
    gate_proj: ttnn.Tensor,
    compute_kernel_config,
    group,
    rows: int,
    hidden_dim: int,
    live_tensors: list[ttnn.Tensor],
    sub_device_id: ttnn.SubDeviceId | None = None,
) -> ttnn.Tensor:
    is_small_m = _is_routed_expert_small_m(rows)
    matmul_kwargs = _routed_expert_gate_up_stage_kwargs(
        group,
        rows,
        hidden_dim,
        fused_activation=(ttnn.UnaryOpType.SILU, False) if is_small_m else None,
    )
    if sub_device_id is not None:
        matmul_kwargs["sub_device_id"] = sub_device_id

    if not is_small_m:
        return ttnn.matmul(
            input_tensor,
            gate_proj,
            activation="silu",
            compute_kernel_config=compute_kernel_config,
            **matmul_kwargs,
        )

    # Keep small-M gate as one fused matmul stage; passing activation="silu" here
    # currently lowers through unary in Python and breaks offset sub-device grids.
    preallocated_output = _allocate_tracked_expert_stage_output(
        input_tensor,
        rows,
        hidden_dim,
        matmul_kwargs["memory_config"],
        live_tensors,
    )
    matmul_kwargs["optional_output_tensor"] = preallocated_output
    gate_output = ttnn.matmul(
        input_tensor,
        gate_proj,
        compute_kernel_config=compute_kernel_config,
        **matmul_kwargs,
    )
    if gate_output is preallocated_output:
        _forget_live_tensor(live_tensors, preallocated_output)
    return gate_output


def _routed_expert_up_stage(
    input_tensor: ttnn.Tensor,
    up_proj: ttnn.Tensor,
    compute_kernel_config,
    group,
    rows: int,
    hidden_dim: int,
    live_tensors: list[ttnn.Tensor],
    sub_device_id: ttnn.SubDeviceId | None = None,
) -> ttnn.Tensor:
    matmul_kwargs = _routed_expert_gate_up_stage_kwargs(group, rows, hidden_dim)
    if sub_device_id is not None:
        matmul_kwargs["sub_device_id"] = sub_device_id

    preallocated_output = None
    if _is_routed_expert_small_m(rows):
        preallocated_output = _allocate_tracked_expert_stage_output(
            input_tensor,
            rows,
            hidden_dim,
            matmul_kwargs["memory_config"],
            live_tensors,
        )
        matmul_kwargs["optional_output_tensor"] = preallocated_output

    up_output = ttnn.matmul(
        input_tensor,
        up_proj,
        compute_kernel_config=compute_kernel_config,
        **matmul_kwargs,
    )
    if preallocated_output is not None and up_output is preallocated_output:
        _forget_live_tensor(live_tensors, preallocated_output)
    return up_output


def _routed_expert_down_stage_kwargs(
    group,
    rows: int,
    n_dim: int,
    k_dim: int,
) -> dict:
    if not _is_routed_expert_small_m(rows):
        return {"core_grid": group.core_grid}

    tile = int(ttnn.TILE_SIZE)
    grid_x = min(11, int(group.core_grid.x))
    m_tiles = max(1, _div_up_int(rows, tile))
    n_tiles = max(1, _div_up_int(n_dim, tile))
    k_tiles = max(1, _div_up_int(k_dim, tile))
    grid_y = _routed_expert_small_m_grid_y(m_tiles)
    per_core_m = _div_up_int(m_tiles, grid_y)
    per_core_n = _div_up_int(n_tiles, grid_x)
    # The C++ stage uses best_in0_block_w(); for the 768-wide routed expert path this selects full K.
    down_in0_block_w = k_tiles

    return {
        "program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
            in0_block_w=down_in0_block_w,
            out_subblock_h=1,
            out_subblock_w=_largest_divisor(per_core_n, 8),
            out_block_h=per_core_m,
            out_block_w=per_core_n,
            per_core_M=per_core_m,
            per_core_N=per_core_n,
            transpose_mcast=False,
            fuse_batch=False,
            allowed_worker_cores=_core_range_set(grid_x, grid_y),
        )
    }


def _routed_expert_mul_reshard(
    gate_output: ttnn.Tensor,
    up_output: ttnn.Tensor,
    rows: int,
    live_tensors: list[ttnn.Tensor],
    sub_device_id: ttnn.SubDeviceId | None = None,
) -> ttnn.Tensor:
    # Match routed_expert_ffn_opt_mul_reshard: multiply first, then reshard the
    # activated result for the small-M path. The inputs already carry an
    # absolute shard grid; binary_ng can infer the owning sub-device from that
    # grid, while passing sub_device_id forces the whole sub-device grid and can
    # trip all_cores.contains(used_cores) for smaller gate/up shard grids.
    ttnn.multiply_(gate_output, up_output)
    _deallocate_tracked_tensor(live_tensors, up_output)

    if not _is_routed_expert_small_m(rows):
        return gate_output

    activated = ttnn.to_memory_config(gate_output, ttnn.L1_MEMORY_CONFIG)
    live_tensors.append(activated)
    _deallocate_tracked_tensor(live_tensors, gate_output)
    return activated


_ROUTED_EXPERT_SMALL_M_TILE_THRESHOLD = 64
_ROUTED_EXPERT_SMALL_M_GRID_Y_MAX = 8


def _routed_expert_small_m_grid_y(m_tiles: int) -> int:
    grid_y_upper = max(1, min(_div_up_int(m_tiles, 4), _ROUTED_EXPERT_SMALL_M_GRID_Y_MAX))
    per_core_m = _div_up_int(m_tiles, grid_y_upper)
    return max(1, min(_ROUTED_EXPERT_SMALL_M_GRID_Y_MAX, _div_up_int(m_tiles, per_core_m)))


def _expert_subdevice_core_rows(ffn_rows: int, max_core_rows: int = 10) -> int:
    m_tiles = max(1, _div_up_int(ffn_rows, int(ttnn.TILE_SIZE)))
    if m_tiles > _ROUTED_EXPERT_SMALL_M_TILE_THRESHOLD:
        return max_core_rows
    return _routed_expert_small_m_grid_y(m_tiles)


def _build_expert_subdevice_assignments(
    config: MoePerfConfig,
    counts_by_expert: torch.Tensor,
    rows_by_expert: torch.Tensor,
    max_core_rows: int = 10,
) -> list[_ExpertSubDeviceAssignment]:
    assignments = []
    for expert_id, token_count_tensor in enumerate(counts_by_expert):
        token_count = int(token_count_tensor.item())
        if token_count <= 0:
            continue
        assignments.append(
            _ExpertSubDeviceAssignment(
                expert_id=expert_id,
                token_count=token_count,
                ffn_rows=int(rows_by_expert[expert_id].item()),
                core_rows=_expert_subdevice_core_rows(int(rows_by_expert[expert_id].item()), max_core_rows),
            )
        )

    assignments.sort(key=lambda item: (item.core_rows, item.token_count), reverse=True)
    if config.expert_subdevice_max_experts > 0:
        assignments = assignments[: config.expert_subdevice_max_experts]
    return assignments


def _pack_expert_subdevice_waves(
    assignments: list[_ExpertSubDeviceAssignment],
    max_core_rows: int,
    max_subdevices: int,
) -> list[_ExpertSubDeviceWave]:
    waves: list[list[_ExpertSubDeviceAssignment]] = []
    wave_core_rows: list[int] = []
    for assignment in assignments:
        if assignment.core_rows > max_core_rows:
            raise ValueError(
                f"Expert {assignment.expert_id} needs {assignment.core_rows} core rows, "
                f"but the device only has {max_core_rows}"
            )

        best_wave_idx = None
        best_remaining = max_core_rows + 1
        for wave_idx, used_rows in enumerate(wave_core_rows):
            if len(waves[wave_idx]) >= max_subdevices:
                continue
            remaining = max_core_rows - used_rows
            if assignment.core_rows <= remaining and remaining - assignment.core_rows < best_remaining:
                best_wave_idx = wave_idx
                best_remaining = remaining - assignment.core_rows

        if best_wave_idx is None:
            waves.append([assignment])
            wave_core_rows.append(assignment.core_rows)
        else:
            waves[best_wave_idx].append(assignment)
            wave_core_rows[best_wave_idx] += assignment.core_rows

    return [
        _ExpertSubDeviceWave(
            assignments=tuple(sorted(wave, key=lambda item: (item.core_rows, item.token_count, item.expert_id))),
            total_core_rows=sum(item.core_rows for item in wave),
            total_subdevices=len(wave),
            total_tokens=sum(item.token_count for item in wave),
            total_ffn_rows=sum(item.ffn_rows for item in wave),
        )
        for wave in waves
    ]


def _make_expert_subdevice_lane(
    core_rows: int,
    assignments: list[_ExpertSubDeviceAssignment],
) -> _ExpertSubDeviceLane:
    ordered = tuple(sorted(assignments, key=lambda item: item.token_count, reverse=True))
    return _ExpertSubDeviceLane(
        core_rows=core_rows,
        assignments=ordered,
        total_tokens=sum(item.token_count for item in ordered),
        total_ffn_rows=sum(item.ffn_rows for item in ordered),
    )


def _expert_subdevice_lane_load(lane: _ExpertSubDeviceLane) -> float:
    return sum(
        float(assignment.ffn_rows) / max(1, min(assignment.core_rows, lane.core_rows))
        for assignment in lane.assignments
    )


def _expert_subdevice_lane_target_load(assignments: list[_ExpertSubDeviceAssignment]) -> float:
    override = os.getenv("MOE_PREFILL_OPT_EXPERT_SUBDEVICE_LANE_TARGET_LOAD")
    if override:
        return max(1.0, float(override))
    if not assignments:
        return 1536.0

    wide_lane_loads = [
        float(assignment.ffn_rows) / max(1, assignment.core_rows)
        for assignment in assignments
        if assignment.core_rows >= 8
    ]
    # Keep bucket lanes around the full-chip/near-full-chip expert duration so
    # leftover rows can be backfilled instead of creating many short tail phases.
    return max(1024.0, min(1536.0, max(wide_lane_loads, default=1024.0)))


def _expert_subdevice_lane_target_candidates(assignments: list[_ExpertSubDeviceAssignment]) -> list[float]:
    override = os.getenv("MOE_PREFILL_OPT_EXPERT_SUBDEVICE_LANE_TARGET_LOAD")
    if override:
        return [max(1.0, float(override))]

    base_target = _expert_subdevice_lane_target_load(assignments)
    candidate_targets = {
        768.0,
        896.0,
        1024.0,
        1152.0,
        1280.0,
        1536.0,
        1792.0,
        2048.0,
        base_target,
        max(1.0, base_target * 0.75),
        max(1.0, base_target * 1.25),
    }
    return sorted(candidate_targets)


def _expert_subdevice_phase_load(phase: _ExpertSubDevicePhase) -> float:
    return max((_expert_subdevice_lane_load(lane) for lane in phase.lanes), default=0.0)


def _expert_subdevice_phase_plan_load(phases: list[_ExpertSubDevicePhase]) -> float:
    return sum(_expert_subdevice_phase_load(phase) for phase in phases)


def _split_expert_subdevice_lanes(
    assignments: list[_ExpertSubDeviceAssignment],
    core_rows: int,
    lane_count: int,
) -> list[_ExpertSubDeviceLane]:
    if not assignments or lane_count <= 0:
        return []

    lane_items: list[list[_ExpertSubDeviceAssignment]] = [[] for _ in range(lane_count)]
    lane_rows = [0 for _ in range(lane_count)]
    for assignment in sorted(assignments, key=lambda item: item.ffn_rows, reverse=True):
        lane_idx = min(range(lane_count), key=lambda idx: lane_rows[idx])
        lane_items[lane_idx].append(assignment)
        lane_rows[lane_idx] += assignment.ffn_rows

    return [
        _make_expert_subdevice_lane(core_rows, lane)
        for lane in lane_items
        if lane
    ]


def _take_expert_subdevice_lane_fill(
    assignments: list[_ExpertSubDeviceAssignment],
    target_ffn_rows: float,
) -> list[_ExpertSubDeviceAssignment]:
    if not assignments:
        return []

    lane: list[_ExpertSubDeviceAssignment] = []
    lane_rows = 0
    while assignments and (not lane or lane_rows < target_ffn_rows):
        assignment = assignments.pop(0)
        lane.append(assignment)
        lane_rows += assignment.ffn_rows
    return lane


def _build_expert_subdevice_phases_for_target(
    assignments: list[_ExpertSubDeviceAssignment],
    max_core_rows: int,
    max_subdevices: int,
    target_lane_load: float,
) -> list[_ExpertSubDevicePhase]:
    buckets: dict[int, list[_ExpertSubDeviceAssignment]] = {}
    for assignment in assignments:
        if assignment.core_rows > max_core_rows:
            raise ValueError(
                f"Expert {assignment.expert_id} needs {assignment.core_rows} core rows, "
                f"but the device only has {max_core_rows}"
            )
        buckets.setdefault(assignment.core_rows, []).append(assignment)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: item.ffn_rows, reverse=True)

    candidate_lanes: list[_ExpertSubDeviceLane] = []
    for core_rows, bucket in sorted(buckets.items(), reverse=True):
        bucket_load = sum(float(assignment.ffn_rows) / max(1, core_rows) for assignment in bucket)
        lane_count = max(1, math.ceil(bucket_load / target_lane_load))
        if core_rows <= max_core_rows // 2:
            lane_count = max(lane_count, min(len(bucket), max_subdevices, max(1, max_core_rows // core_rows)))
        lane_count = min(len(bucket), lane_count)
        candidate_lanes.extend(_split_expert_subdevice_lanes(bucket, core_rows=core_rows, lane_count=lane_count))

    candidate_lanes.sort(
        key=lambda lane: (_expert_subdevice_lane_load(lane), lane.core_rows, lane.total_ffn_rows),
        reverse=True,
    )

    phase_lanes: list[list[_ExpertSubDeviceLane]] = []
    phase_core_rows: list[int] = []
    for lane in candidate_lanes:
        best_phase_idx = None
        best_score = None
        lane_load = _expert_subdevice_lane_load(lane)
        for phase_idx, lanes in enumerate(phase_lanes):
            if len(lanes) >= max_subdevices:
                continue
            next_core_rows = phase_core_rows[phase_idx] + lane.core_rows
            if next_core_rows > max_core_rows:
                continue
            next_max_load = max([_expert_subdevice_lane_load(existing) for existing in lanes] + [lane_load])
            phase_max_load = max(_expert_subdevice_lane_load(existing) for existing in lanes)
            load_growth = max(0.0, next_max_load - phase_max_load)
            score = (
                load_growth,
                max_core_rows - next_core_rows,
                next_max_load,
                len(lanes),
            )
            if best_score is None or score < best_score:
                best_score = score
                best_phase_idx = phase_idx

        if best_phase_idx is None:
            phase_lanes.append([lane])
            phase_core_rows.append(lane.core_rows)
        else:
            phase_lanes[best_phase_idx].append(lane)
            phase_core_rows[best_phase_idx] += lane.core_rows

    return [
        _ExpertSubDevicePhase(
            lanes=tuple(sorted(lanes, key=lambda lane: lane.core_rows)),
            total_core_rows=sum(lane.core_rows for lane in lanes),
            total_subdevices=len(lanes),
            total_tokens=sum(lane.total_tokens for lane in lanes),
            total_ffn_rows=sum(lane.total_ffn_rows for lane in lanes),
        )
        for lanes in phase_lanes
        if lanes
    ]


def _build_expert_subdevice_phases(
    assignments: list[_ExpertSubDeviceAssignment],
    max_core_rows: int,
    max_subdevices: int,
) -> list[_ExpertSubDevicePhase]:
    phase_overhead_load = _env_float("MOE_PREFILL_OPT_EXPERT_SUBDEVICE_PHASE_OVERHEAD_LOAD", 128.0)
    best_target = None
    best_phases: list[_ExpertSubDevicePhase] | None = None
    best_score: tuple[float, float, int, float] | None = None
    for target_lane_load in _expert_subdevice_lane_target_candidates(assignments):
        phases = _build_expert_subdevice_phases_for_target(
            assignments,
            max_core_rows,
            max_subdevices,
            target_lane_load,
        )
        phase_load = _expert_subdevice_phase_plan_load(phases)
        score = (
            phase_load + phase_overhead_load * len(phases),
            phase_load,
            len(phases),
            target_lane_load,
        )
        if best_score is None or score < best_score:
            best_score = score
            best_target = target_lane_load
            best_phases = phases

    phases = best_phases or []
    logger.info(
        "MoE opt expert sub-device scheduler: "
        f"selected_lane_target_load={best_target:.1f}, "
        f"estimated_phase_load={_expert_subdevice_phase_plan_load(phases):.1f}, "
        f"phase_overhead_load={phase_overhead_load:.1f}"
    )
    return phases


def _make_subdevice_groups_for_wave(
    wave: _ExpertSubDeviceWave,
    cols: int,
) -> list[_ExpertSubDeviceGroup]:
    groups = []
    start_y = 0
    for group_id, assignment in enumerate(wave.assignments):
        cores = _core_range_set(cols, assignment.core_rows, start_y=start_y)
        groups.append(
            _ExpertSubDeviceGroup(
                sub_device_id=ttnn.SubDeviceId(group_id),
                cores=cores,
                core_grid=ttnn.CoreGrid(x=cols, y=assignment.core_rows),
                expert_id=assignment.expert_id,
                token_count=assignment.token_count,
                ffn_rows=assignment.ffn_rows,
                core_rows=assignment.core_rows,
                start_y=start_y,
            )
        )
        start_y += assignment.core_rows
    return groups


def _make_subdevice_lane_groups_for_phase(
    phase: _ExpertSubDevicePhase,
    cols: int,
) -> list[_ExpertSubDeviceLaneGroup]:
    groups = []
    start_y = 0
    for group_id, lane in enumerate(phase.lanes):
        cores = _core_range_set(cols, lane.core_rows, start_y=start_y)
        groups.append(
            _ExpertSubDeviceLaneGroup(
                sub_device_id=ttnn.SubDeviceId(group_id),
                cores=cores,
                core_grid=ttnn.CoreGrid(x=cols, y=lane.core_rows),
                core_rows=lane.core_rows,
                start_y=start_y,
                assignments=lane.assignments,
            )
        )
        start_y += lane.core_rows
    return groups


def _run_staged_expert_ffn_wave_on_subdevices(
    config: MoePerfConfig,
    routed_expert,
    mesh_device: ttnn.MeshDevice,
    wave: list[tuple[int, int, _ExpertSubDeviceGroup]],
) -> None:
    live_tensors: list[ttnn.Tensor] = []
    inputs: dict[int, ttnn.Tensor] = {}
    gate_outputs: dict[int, ttnn.Tensor] = {}
    up_outputs: dict[int, ttnn.Tensor] = {}
    activated_outputs: dict[int, ttnn.Tensor] = {}
    final_outputs: dict[int, ttnn.Tensor] = {}

    try:
        for expert_id, rows, _ in wave:
            input_tensor = _allocate_direct_expert_input(config, mesh_device, rows)
            inputs[expert_id] = input_tensor
            live_tensors.append(input_tensor)

        for expert_id, rows, group in wave:
            gate_outputs[expert_id] = _routed_expert_gate_stage(
                inputs[expert_id],
                routed_expert.gate_projs[expert_id],
                routed_expert.compute_kernel_config,
                group,
                rows,
                config.hidden_dim,
                live_tensors,
                group.sub_device_id,
            )
            live_tensors.append(gate_outputs[expert_id])

        for expert_id, rows, group in wave:
            up_outputs[expert_id] = _routed_expert_up_stage(
                inputs[expert_id],
                routed_expert.up_projs[expert_id],
                routed_expert.compute_kernel_config,
                group,
                rows,
                config.hidden_dim,
                live_tensors,
                group.sub_device_id,
            )
            live_tensors.append(up_outputs[expert_id])

        for expert_id, rows, group in wave:
            activated_outputs[expert_id] = _routed_expert_mul_reshard(
                gate_outputs[expert_id],
                up_outputs[expert_id],
                rows,
                live_tensors,
                group.sub_device_id,
            )

        for expert_id, rows, group in wave:
            final_outputs[expert_id] = ttnn.matmul(
                activated_outputs[expert_id],
                routed_expert.down_projs[expert_id],
                compute_kernel_config=routed_expert.compute_kernel_config,
                sub_device_id=group.sub_device_id,
                **_routed_expert_down_stage_kwargs(group, rows, config.emb_dim, config.hidden_dim),
            )
            live_tensors.append(final_outputs[expert_id])

        ttnn.synchronize_device(mesh_device, sub_device_ids=[group.sub_device_id for _, _, group in wave])
    finally:
        for tensor in reversed(live_tensors):
            try:
                ttnn.deallocate(tensor)
            except Exception:
                logger.warning("Failed to deallocate a sub-device expert FFN temporary", exc_info=True)


def _run_staged_expert_ffn_phase_on_subdevices(
    config: MoePerfConfig,
    routed_expert,
    mesh_device: ttnn.MeshDevice,
    lane_groups: list[_ExpertSubDeviceLaneGroup],
    rt_profile: _ExpertSubDeviceRtProfiler | None = None,
) -> None:
    live_tensors: list[ttnn.Tensor] = []
    max_lane_depth = max((len(group.assignments) for group in lane_groups), default=0)

    try:
        for lane_idx in range(max_lane_depth):
            for group_idx, group in enumerate(lane_groups):
                if lane_idx >= len(group.assignments):
                    continue

                assignment = group.assignments[lane_idx]
                expert_id = assignment.expert_id
                rows = assignment.ffn_rows

                input_tensor = _allocate_direct_expert_input(config, mesh_device, rows)
                live_tensors.append(input_tensor)

                if rt_profile is not None:
                    rt_profile.record_issue("gate", lane_idx, group_idx, group, assignment)
                gate_output = _routed_expert_gate_stage(
                    input_tensor,
                    routed_expert.gate_projs[expert_id],
                    routed_expert.compute_kernel_config,
                    group,
                    rows,
                    config.hidden_dim,
                    live_tensors,
                    group.sub_device_id,
                )
                live_tensors.append(gate_output)

                if rt_profile is not None:
                    rt_profile.record_issue("up", lane_idx, group_idx, group, assignment)
                up_output = _routed_expert_up_stage(
                    input_tensor,
                    routed_expert.up_projs[expert_id],
                    routed_expert.compute_kernel_config,
                    group,
                    rows,
                    config.hidden_dim,
                    live_tensors,
                    group.sub_device_id,
                )
                live_tensors.append(up_output)

                if rt_profile is not None:
                    rt_profile.record_issue("mul", lane_idx, group_idx, group, assignment)
                    rt_profile.record_issue("reshard", lane_idx, group_idx, group, assignment)
                activated = _routed_expert_mul_reshard(
                    gate_output,
                    up_output,
                    rows,
                    live_tensors,
                    group.sub_device_id,
                )

                if rt_profile is not None:
                    rt_profile.record_issue("down", lane_idx, group_idx, group, assignment)
                output = ttnn.matmul(
                    activated,
                    routed_expert.down_projs[expert_id],
                    compute_kernel_config=routed_expert.compute_kernel_config,
                    sub_device_id=group.sub_device_id,
                    **_routed_expert_down_stage_kwargs(group, rows, config.emb_dim, config.hidden_dim),
                )
                live_tensors.append(output)

        ttnn.synchronize_device(mesh_device, sub_device_ids=[group.sub_device_id for group in lane_groups])
    finally:
        for tensor in reversed(live_tensors):
            try:
                ttnn.deallocate(tensor)
            except Exception:
                logger.warning("Failed to deallocate a sub-device expert FFN temporary", exc_info=True)


def _run_fused_expert_ffn_phase_on_subdevices(
    config: MoePerfConfig,
    routed_expert,
    mesh_device: ttnn.MeshDevice,
    lane_groups: list[_ExpertSubDeviceLaneGroup],
    rt_profile: _ExpertSubDeviceRtProfiler | None = None,
) -> None:
    fused_op = getattr(ttnn.experimental.deepseek_prefill, "routed_expert_ffn_opt", None)
    if fused_op is None:
        raise RuntimeError("routed_expert_ffn_opt is unavailable")

    grid = mesh_device.compute_with_storage_grid_size()
    max_core_rows = int(grid.y)
    live_tensors: list[ttnn.Tensor] = []
    max_lane_depth = max((len(group.assignments) for group in lane_groups), default=0)

    try:
        for lane_idx in range(max_lane_depth):
            for group_idx, group in enumerate(lane_groups):
                if lane_idx >= len(group.assignments):
                    continue

                assignment = group.assignments[lane_idx]
                expert_id = assignment.expert_id
                input_tensor = _allocate_direct_expert_input(config, mesh_device, assignment.ffn_rows)
                live_tensors.append(input_tensor)

                fused_kwargs = {"compute_kernel_config": routed_expert.compute_kernel_config}
                # Full-height phases are single-lane 11xmax_core_rows runs. Leaving
                # sub_device_id unset preserves the original large-M fused fallback
                # config while the loaded sub-device manager contains only this lane.
                if group.core_rows < max_core_rows:
                    fused_kwargs["sub_device_id"] = group.sub_device_id

                if rt_profile is not None:
                    for stage in ("gate", "up", "mul", "reshard", "down"):
                        rt_profile.record_issue(stage, lane_idx, group_idx, group, assignment)
                output = fused_op(
                    input_tensor,
                    routed_expert.gate_projs[expert_id],
                    routed_expert.up_projs[expert_id],
                    routed_expert.down_projs[expert_id],
                    **fused_kwargs,
                )
                live_tensors.append(output)

        ttnn.synchronize_device(mesh_device, sub_device_ids=[group.sub_device_id for group in lane_groups])
    finally:
        for tensor in reversed(live_tensors):
            try:
                ttnn.deallocate(tensor)
            except Exception:
                logger.warning("Failed to deallocate a fused sub-device expert FFN temporary", exc_info=True)


def _subdevice_kwargs_for_group(
    group: _ExpertSubDeviceLaneGroup,
    max_core_rows: int,
) -> dict:
    # Full-height phases are single-lane 11xmax_core_rows runs. Leaving
    # sub_device_id unset preserves the original large-M fused fallback
    # config while the loaded sub-device manager contains only this lane.
    if group.core_rows >= max_core_rows:
        return {}
    return {"sub_device_id": group.sub_device_id}


def _run_cpp_stage_roundrobin_expert_ffn_phase_on_subdevices(
    config: MoePerfConfig,
    routed_expert,
    mesh_device: ttnn.MeshDevice,
    lane_groups: list[_ExpertSubDeviceLaneGroup],
    rt_profile: _ExpertSubDeviceRtProfiler | None = None,
) -> None:
    deepseek = ttnn.experimental.deepseek_prefill
    gate_op = getattr(deepseek, "routed_expert_ffn_opt_gate", None)
    up_op = getattr(deepseek, "routed_expert_ffn_opt_up", None)
    mul_op = getattr(deepseek, "routed_expert_ffn_opt_mul_reshard", None)
    down_op = getattr(deepseek, "routed_expert_ffn_opt_down", None)
    if any(op is None for op in (gate_op, up_op, mul_op, down_op)):
        raise RuntimeError("routed_expert_ffn_opt stage functions are unavailable")

    grid = mesh_device.compute_with_storage_grid_size()
    max_core_rows = int(grid.y)
    live_tensors: list[ttnn.Tensor] = []
    max_lane_depth = max((len(group.assignments) for group in lane_groups), default=0)

    try:
        for lane_idx in range(max_lane_depth):
            slots = []
            for group_idx, group in enumerate(lane_groups):
                if lane_idx >= len(group.assignments):
                    continue

                assignment = group.assignments[lane_idx]
                input_tensor = _allocate_direct_expert_input(config, mesh_device, assignment.ffn_rows)
                live_tensors.append(input_tensor)
                slots.append(
                    {
                        "group": group,
                        "group_idx": group_idx,
                        "assignment": assignment,
                        "input": input_tensor,
                        "subdevice_kwargs": _subdevice_kwargs_for_group(group, max_core_rows),
                    }
                )

            for slot in slots:
                assignment = slot["assignment"]
                if rt_profile is not None:
                    rt_profile.record_issue("gate", lane_idx, slot["group_idx"], slot["group"], assignment)
                slot["gate"] = gate_op(
                    slot["input"],
                    routed_expert.gate_projs[assignment.expert_id],
                    compute_kernel_config=routed_expert.compute_kernel_config,
                    **slot["subdevice_kwargs"],
                )
                live_tensors.append(slot["gate"])

            for slot in slots:
                assignment = slot["assignment"]
                if rt_profile is not None:
                    rt_profile.record_issue("up", lane_idx, slot["group_idx"], slot["group"], assignment)
                slot["up"] = up_op(
                    slot["input"],
                    routed_expert.up_projs[assignment.expert_id],
                    compute_kernel_config=routed_expert.compute_kernel_config,
                    **slot["subdevice_kwargs"],
                )
                live_tensors.append(slot["up"])

            for slot in slots:
                if rt_profile is not None:
                    rt_profile.record_issue("mul", lane_idx, slot["group_idx"], slot["group"], slot["assignment"])
                    rt_profile.record_issue("reshard", lane_idx, slot["group_idx"], slot["group"], slot["assignment"])
                slot["activated"] = mul_op(
                    slot["gate"],
                    slot["up"],
                    **slot["subdevice_kwargs"],
                )
                _forget_live_tensor(live_tensors, slot["gate"])
                _forget_live_tensor(live_tensors, slot["up"])
                live_tensors.append(slot["activated"])

            for slot in slots:
                assignment = slot["assignment"]
                if rt_profile is not None:
                    rt_profile.record_issue("down", lane_idx, slot["group_idx"], slot["group"], assignment)
                slot["output"] = down_op(
                    slot["activated"],
                    routed_expert.down_projs[assignment.expert_id],
                    compute_kernel_config=routed_expert.compute_kernel_config,
                    **slot["subdevice_kwargs"],
                )
                live_tensors.append(slot["output"])

        ttnn.synchronize_device(mesh_device, sub_device_ids=[group.sub_device_id for group in lane_groups])
    finally:
        for tensor in reversed(live_tensors):
            try:
                ttnn.deallocate(tensor)
            except Exception:
                logger.warning("Failed to deallocate a C++ stage round-robin sub-device expert FFN temporary", exc_info=True)


def _run_staged_expert_ffn_serial_baseline(
    config: MoePerfConfig,
    routed_expert,
    mesh_device: ttnn.MeshDevice,
    assignments: list[_ExpertSubDeviceAssignment],
) -> None:
    grid = mesh_device.compute_with_storage_grid_size()
    cols, max_core_rows = int(grid.x), int(grid.y)

    for assignment in assignments:
        live_tensors: list[ttnn.Tensor] = []
        core_rows = _expert_subdevice_core_rows(assignment.ffn_rows, max_core_rows)
        baseline_group = _ExpertSubDeviceLaneGroup(
            sub_device_id=ttnn.SubDeviceId(0),
            cores=_core_range_set(cols, core_rows),
            core_grid=ttnn.CoreGrid(x=cols, y=core_rows),
            core_rows=core_rows,
            start_y=0,
            assignments=(assignment,),
        )
        try:
            input_tensor = _allocate_direct_expert_input(config, mesh_device, assignment.ffn_rows)
            live_tensors.append(input_tensor)
            gate_output = _routed_expert_gate_stage(
                input_tensor,
                routed_expert.gate_projs[assignment.expert_id],
                routed_expert.compute_kernel_config,
                baseline_group,
                assignment.ffn_rows,
                config.hidden_dim,
                live_tensors,
            )
            live_tensors.append(gate_output)
            up_output = _routed_expert_up_stage(
                input_tensor,
                routed_expert.up_projs[assignment.expert_id],
                routed_expert.compute_kernel_config,
                baseline_group,
                assignment.ffn_rows,
                config.hidden_dim,
                live_tensors,
            )
            live_tensors.append(up_output)
            activated = _routed_expert_mul_reshard(
                gate_output,
                up_output,
                assignment.ffn_rows,
                live_tensors,
            )
            output = ttnn.matmul(
                activated,
                routed_expert.down_projs[assignment.expert_id],
                compute_kernel_config=routed_expert.compute_kernel_config,
                **_routed_expert_down_stage_kwargs(
                    baseline_group,
                    assignment.ffn_rows,
                    config.emb_dim,
                    config.hidden_dim,
                ),
            )
            live_tensors.append(output)
            ttnn.synchronize_device(mesh_device)
        finally:
            for tensor in reversed(live_tensors):
                try:
                    ttnn.deallocate(tensor)
                except Exception:
                    logger.warning("Failed to deallocate a baseline expert FFN temporary", exc_info=True)


def _run_fused_expert_ffn_serial_baseline(
    config: MoePerfConfig,
    routed_expert,
    mesh_device: ttnn.MeshDevice,
    assignments: list[_ExpertSubDeviceAssignment],
) -> None:
    fused_op = getattr(ttnn.experimental.deepseek_prefill, "routed_expert_ffn_opt", None)
    if fused_op is None:
        raise RuntimeError("routed_expert_ffn_opt is unavailable")

    for assignment in assignments:
        live_tensors: list[ttnn.Tensor] = []
        try:
            input_tensor = _allocate_direct_expert_input(config, mesh_device, assignment.ffn_rows)
            live_tensors.append(input_tensor)
            output = fused_op(
                input_tensor,
                routed_expert.gate_projs[assignment.expert_id],
                routed_expert.up_projs[assignment.expert_id],
                routed_expert.down_projs[assignment.expert_id],
                compute_kernel_config=routed_expert.compute_kernel_config,
            )
            live_tensors.append(output)
            ttnn.synchronize_device(mesh_device)
        finally:
            for tensor in reversed(live_tensors):
                try:
                    ttnn.deallocate(tensor)
                except Exception:
                    logger.warning("Failed to deallocate a fused baseline expert FFN temporary", exc_info=True)


def _expert_subdevice_bucket_summary(assignments: list[_ExpertSubDeviceAssignment]) -> str:
    bucket_totals: dict[int, dict[str, int]] = {}
    for assignment in assignments:
        totals = bucket_totals.setdefault(assignment.core_rows, {"experts": 0, "tokens": 0, "ffn_rows": 0})
        totals["experts"] += 1
        totals["tokens"] += assignment.token_count
        totals["ffn_rows"] += assignment.ffn_rows
    return "; ".join(
        f"11x{core_rows}:experts={totals['experts']},tokens={totals['tokens']},ffn_rows={totals['ffn_rows']}"
        for core_rows, totals in sorted(bucket_totals.items(), reverse=True)
        if totals["experts"]
    )


def _write_expert_subdevice_csv(
    config: MoePerfConfig,
    phases: list[_ExpertSubDevicePhase],
    baseline_avg_s: float,
    subdevice_avg_s: float,
    speedup: float,
    max_core_rows: int,
    baseline_execution_mode: str,
    subdevice_execution_mode: str,
) -> str:
    csv_path = Path(config.expert_subdevice_csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    baseline_fusion_enabled = baseline_execution_mode.startswith("cpp_")
    subdevice_fusion_enabled = subdevice_execution_mode.startswith("cpp_")
    for phase_idx, phase in enumerate(phases):
        for lane_idx, lane in enumerate(phase.lanes):
            for lane_order, assignment in enumerate(lane.assignments):
                records.append(
                    {
                        "name": config.name,
                        "baseline_execution_mode": baseline_execution_mode,
                        "subdevice_execution_mode": subdevice_execution_mode,
                        "baseline_fusion_enabled": baseline_fusion_enabled,
                        "subdevice_fusion_enabled": subdevice_fusion_enabled,
                        "forced_equal_rows": config.expert_subdevice_equal_rows,
                        "execution_mode": f"baseline_{baseline_execution_mode}__subdevice_{subdevice_execution_mode}",
                        "phase": phase_idx,
                        "lane": lane_idx,
                        "lane_order": lane_order,
                        "expert_id": assignment.expert_id,
                        "token_count": assignment.token_count,
                        "ffn_rows": assignment.ffn_rows,
                        "subdevice_core_grid": f"11x{lane.core_rows}",
                        "core_rows": lane.core_rows,
                        "lane_estimated_load": _expert_subdevice_lane_load(lane),
                        "lane_total_experts": len(lane.assignments),
                        "lane_total_tokens": lane.total_tokens,
                        "lane_total_ffn_rows": lane.total_ffn_rows,
                        "phase_total_subdevices": phase.total_subdevices,
                        "phase_total_core_rows": phase.total_core_rows,
                        "phase_core_utilization": phase.total_core_rows / max(1, max_core_rows),
                        "phase_estimated_load": _expert_subdevice_phase_load(phase),
                        "phase_total_tokens": phase.total_tokens,
                        "phase_total_ffn_rows": phase.total_ffn_rows,
                        "baseline_avg_ms": baseline_avg_s * 1000.0,
                        "subdevice_avg_ms": subdevice_avg_s * 1000.0,
                        "speedup_vs_baseline": speedup,
                    }
                )

    if not records:
        return ""

    fieldnames = list(records[0].keys())
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return str(csv_path)


def _measure_expert_ffn_subdevice_parallel(
    config: MoePerfConfig,
    tt_moe: TtMoe,
    mesh_device: ttnn.MeshDevice,
    dispatch_group_size: int,
    routing_frequency: torch.Tensor | None,
) -> str:
    if not config.expert_subdevice_parallel:
        return ""

    if mesh_device.get_num_devices() != 1:
        logger.warning("Skipping expert sub-device parallel run: only 1x1 mesh/device is supported for this opt test")
        return "skipped: requires 1x1 mesh"

    routed_expert = getattr(tt_moe, "routed_expert", None)
    if routed_expert is None or not routed_expert.gate_projs:
        logger.warning("Skipping expert sub-device parallel run: routed expert weights are unavailable")
        return "skipped: routed expert unavailable"

    if config.expert_subdevice_equal_rows > 0:
        counts = torch.full(
            (config.num_routed_experts,),
            float(config.expert_subdevice_equal_rows),
            dtype=torch.float32,
        )
        rows_by_expert = torch.full(
            (config.num_routed_experts,),
            int(config.expert_subdevice_equal_rows),
            dtype=torch.int64,
        )
        logger.info(
            "MoE opt expert sub-device equal-row override: "
            f"all experts use token_count={config.expert_subdevice_equal_rows}, "
            f"ffn_rows={config.expert_subdevice_equal_rows}, input_shape="
            f"[{config.expert_subdevice_equal_rows}, {config.emb_dim}]"
        )
    else:
        counts = _expert_assignment_counts(config, dispatch_group_size, routing_frequency)
        if routing_frequency is None and config.dynamic_expert_capacity:
            counts = torch.full(
                (config.num_routed_experts,),
                float(dispatch_group_size * config.seq_len_per_chip * config.num_experts_per_tok)
                / config.num_routed_experts,
            )
        rows_by_expert = _expert_ffn_rows_from_counts(config, dispatch_group_size, counts).to(torch.int64)
    grid = mesh_device.compute_with_storage_grid_size()
    cols, max_core_rows = int(grid.x), int(grid.y)
    if cols < 11 or max_core_rows < 10:
        logger.warning(
            "Expert sub-device bucket run was requested for an 11x10-style workload, "
            f"but this device reports {cols}x{max_core_rows} compute cores"
        )

    assignments = _build_expert_subdevice_assignments(config, counts, rows_by_expert, max_core_rows)
    if not assignments:
        logger.warning("Skipping expert sub-device parallel run: no active experts")
        return "skipped: no active experts"
    max_subdevices = 8
    phases = _build_expert_subdevice_phases(assignments, max_core_rows, max_subdevices)
    fused_op_available = _routed_expert_ffn_opt_available()
    fused_op_supports_subdevice = _routed_expert_ffn_opt_supports_subdevice()
    subdevice_cpp_mode = config.expert_subdevice_cpp_mode
    if subdevice_cpp_mode not in {"stage_rr", "composite"}:
        logger.warning(
            f"Unknown MOE_PREFILL_OPT_EXPERT_SUBDEVICE_CPP_MODE={subdevice_cpp_mode!r}; using stage_rr"
        )
        subdevice_cpp_mode = "stage_rr"

    use_baseline_fused_cpp = config.expert_baseline_fused_cpp and fused_op_available
    use_subdevice_fused_cpp = config.expert_subdevice_fused_cpp and fused_op_supports_subdevice
    if config.expert_baseline_fused_cpp and not use_baseline_fused_cpp:
        logger.warning(
            "MOE_PREFILL_OPT_EXPERT_BASELINE_FUSED_CPP=1 was requested, but routed_expert_ffn_opt "
            "is unavailable; falling baseline back to Python staged FFN"
        )
    if config.expert_subdevice_fused_cpp and not use_subdevice_fused_cpp:
        subdevice_fallback_reason = (
            "routed_expert_ffn_opt is unavailable"
            if not fused_op_available
            else "the loaded routed_expert_ffn_opt binding does not expose sub_device_id"
        )
        logger.warning(
            "MOE_PREFILL_OPT_EXPERT_SUBDEVICE_FUSED_CPP=1 was requested, but "
            f"{subdevice_fallback_reason}; falling sub-device path back to Python staged FFN"
        )

    baseline_execution_mode = "cpp_fused" if use_baseline_fused_cpp else "python_staged"
    subdevice_execution_mode = (
        f"cpp_{subdevice_cpp_mode}"
        if use_subdevice_fused_cpp
        else "python_staged"
    )
    baseline_runner = (
        _run_fused_expert_ffn_serial_baseline
        if use_baseline_fused_cpp
        else _run_staged_expert_ffn_serial_baseline
    )
    if use_subdevice_fused_cpp and subdevice_cpp_mode == "stage_rr":
        subdevice_runner = _run_cpp_stage_roundrobin_expert_ffn_phase_on_subdevices
    elif use_subdevice_fused_cpp:
        subdevice_runner = _run_fused_expert_ffn_phase_on_subdevices
    else:
        subdevice_runner = _run_staged_expert_ffn_phase_on_subdevices
    bucket_summary = _expert_subdevice_bucket_summary(assignments)
    phase_summary = "; ".join(
        f"p{idx}:core_rows={phase.total_core_rows}/{max_core_rows},subdevs={phase.total_subdevices}/{max_subdevices},"
        f"tokens={phase.total_tokens},experts={sum(len(lane.assignments) for lane in phase.lanes)},"
        f"load={_expert_subdevice_phase_load(phase):.1f},"
        f"lanes={','.join(f'11x{lane.core_rows}:experts={len(lane.assignments)},load={_expert_subdevice_lane_load(lane):.1f}' for lane in phase.lanes)}"
        for idx, phase in enumerate(phases[:8])
    )
    if len(phases) > 8:
        phase_summary += f"; ... +{len(phases) - 8} phases"
    logger.info(f"MoE opt expert sub-device buckets: {bucket_summary}")
    logger.info(
        "MoE opt expert sub-device execution modes: "
        f"baseline={baseline_execution_mode}, subdevice={subdevice_execution_mode}"
    )
    logger.info(f"MoE opt expert sub-device phases: {phase_summary}")

    warmup_iterations = max(0, int(config.expert_subdevice_warmup_iterations))
    iterations = max(1, int(config.expert_subdevice_iterations))
    baseline_durations_s: list[float] = []
    subdevice_durations_s: list[float] = []
    phase_runtimes: list[tuple[_ExpertSubDevicePhase, list[_ExpertSubDeviceLaneGroup], ttnn.SubDeviceManagerId]] = []
    rt_profile = (
        _ExpertSubDeviceRtProfiler(config.expert_subdevice_rt_profile_csv_path)
        if config.expert_subdevice_rt_profile
        else None
    )
    rt_profile_summary = ""

    try:
        for phase in phases:
            groups = _make_subdevice_lane_groups_for_phase(phase, cols)
            manager = mesh_device.create_sub_device_manager([ttnn.SubDevice([group.cores]) for group in groups], 0)
            phase_runtimes.append((phase, groups, manager))

        for warmup_idx in range(warmup_iterations):
            logger.info(
                f"MoE opt expert sub-device baseline warmup {warmup_idx + 1}/{warmup_iterations}"
            )
            baseline_runner(config, routed_expert, mesh_device, assignments)

        for iteration in range(iterations):
            logger.info(f"MoE opt expert sub-device baseline pass {iteration + 1}/{iterations}")
            ttnn.synchronize_device(mesh_device)
            start_s = time.perf_counter()
            baseline_runner(config, routed_expert, mesh_device, assignments)
            ttnn.synchronize_device(mesh_device)
            baseline_durations_s.append(time.perf_counter() - start_s)

        for warmup_idx in range(warmup_iterations):
            logger.info(
                f"MoE opt expert sub-device lane-phase warmup {warmup_idx + 1}/{warmup_iterations}"
            )
            for _, groups, manager in phase_runtimes:
                group_ids = [group.sub_device_id for group in groups]
                mesh_device.load_sub_device_manager(manager)
                mesh_device.set_sub_device_stall_group(group_ids)
                subdevice_runner(
                    config,
                    routed_expert,
                    mesh_device,
                    groups,
                    rt_profile=None,
                )
                mesh_device.reset_sub_device_stall_group()
                mesh_device.clear_loaded_sub_device_manager()

        rt_profile_active = False
        if rt_profile is not None:
            ttnn.synchronize_device(mesh_device)
            rt_profile_active = rt_profile.start()

        try:
            for iteration in range(iterations):
                logger.info(f"MoE opt expert sub-device lane-phase pass {iteration + 1}/{iterations}")
                ttnn.synchronize_device(mesh_device)
                start_s = time.perf_counter()
                for phase_idx, (_, groups, manager) in enumerate(phase_runtimes):
                    group_ids = [group.sub_device_id for group in groups]
                    mesh_device.load_sub_device_manager(manager)
                    mesh_device.set_sub_device_stall_group(group_ids)
                    if rt_profile is not None and rt_profile_active:
                        rt_profile.set_context(iteration, phase_idx)
                    subdevice_runner(
                        config,
                        routed_expert,
                        mesh_device,
                        groups,
                        rt_profile=rt_profile if rt_profile_active else None,
                    )
                    mesh_device.reset_sub_device_stall_group()
                    mesh_device.clear_loaded_sub_device_manager()
                ttnn.synchronize_device(mesh_device)
                subdevice_durations_s.append(time.perf_counter() - start_s)
        finally:
            if rt_profile is not None and rt_profile_active:
                ttnn.synchronize_device(mesh_device)
                rt_profile.stop()
                rt_profile_summary = rt_profile.write_reports()
            elif rt_profile is not None:
                rt_profile_summary = rt_profile.write_reports()
            if rt_profile_summary:
                logger.info(f"MoE opt expert sub-device {rt_profile_summary}")
    finally:
        for _, _, manager in phase_runtimes:
            mesh_device.reset_sub_device_stall_group()
            mesh_device.clear_loaded_sub_device_manager()
            mesh_device.remove_sub_device_manager(manager)

    active_experts = len(assignments)
    active_tokens = sum(assignment.token_count for assignment in assignments)
    active_rows = sum(assignment.ffn_rows for assignment in assignments)
    baseline_avg_s = statistics.mean(baseline_durations_s)
    subdevice_avg_s = statistics.mean(subdevice_durations_s)
    speedup = baseline_avg_s / max(subdevice_avg_s, 1e-12)
    total_matmul_flops = 6.0 * active_rows * config.emb_dim * config.hidden_dim
    baseline_tflops = total_matmul_flops / max(baseline_avg_s, 1e-12) / 1e12
    subdevice_tflops = total_matmul_flops / max(subdevice_avg_s, 1e-12) / 1e12
    csv_path = _write_expert_subdevice_csv(
        config,
        phases,
        baseline_avg_s,
        subdevice_avg_s,
        speedup,
        max_core_rows,
        baseline_execution_mode,
        subdevice_execution_mode,
    )
    rt_profile_suffix = f" | {rt_profile_summary}" if rt_profile_summary else ""
    equal_rows_suffix = (
        f" | equal_rows={config.expert_subdevice_equal_rows} | input_shape=[{config.expert_subdevice_equal_rows},{config.emb_dim}]"
        if config.expert_subdevice_equal_rows > 0
        else ""
    )
    return (
        f"bucketed_lane_11x{max_core_rows} | experts={active_experts} | tokens={active_tokens} | rows={active_rows} | "
        f"phases={len(phases)} | max_subdevices={max_subdevices} | "
        f"baseline_fusion={'on' if use_baseline_fused_cpp else 'off'} | "
        f"subdevice_fusion={'on' if use_subdevice_fused_cpp else 'off'} | "
        f"baseline_mode={baseline_execution_mode} | subdevice_mode={subdevice_execution_mode} | iters={iterations} | "
        f"baseline={baseline_avg_s * 1000.0:.3f}ms "
        f"({baseline_tflops:.3f} TF/s) | subdevice={subdevice_avg_s * 1000.0:.3f}ms "
        f"({subdevice_tflops:.3f} TF/s) | speedup={speedup:.3f}x{equal_rows_suffix} | buckets=[{bucket_summary}] | "
        f"phases=[{phase_summary}] | csv={csv_path}{rt_profile_suffix}"
    )


def _measure_expert_ffn_row_sweep(config: MoePerfConfig, tt_moe: TtMoe, mesh_device: ttnn.MeshDevice) -> str:
    if not config.expert_ffn_sweep_rows:
        return ""

    routed_expert = getattr(tt_moe, "routed_expert", None)
    if routed_expert is None or not hasattr(routed_expert, "_expert_ffn"):
        logger.warning("Skipping expert FFN sweep: routed expert hook is unavailable")
        return ""
    if not routed_expert.gate_projs:
        logger.warning("Skipping expert FFN sweep: no routed expert weights are loaded")
        return ""

    expert_id = max(0, min(int(config.expert_ffn_sweep_expert_id), len(routed_expert.gate_projs) - 1))
    warmup_iterations = max(0, int(config.expert_ffn_breakdown_warmup_iterations))
    iterations = max(1, int(config.expert_ffn_breakdown_iterations))
    csv_path = Path(config.expert_ffn_sweep_csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    previous_breakdown = getattr(routed_expert, "breakdown_expert_ffn", False)
    previous_record_breakdown = getattr(routed_expert, "record_expert_ffn_breakdown", True)
    previous_measure = getattr(routed_expert, "measure_expert_time", False)
    records: list[dict] = []

    try:
        for rows in config.expert_ffn_sweep_rows:
            rows = int(rows)
            if rows <= 0:
                continue
            logger.info(
                f"MoE opt expert FFN sweep rows={rows} expert={expert_id} "
                f"warmup={warmup_iterations} iters={iterations}"
            )

            routed_expert.breakdown_expert_ffn = False
            routed_expert.measure_expert_time = False
            for warmup_idx in range(warmup_iterations):
                logger.info(f"MoE opt expert FFN sweep fused warmup rows={rows} {warmup_idx + 1}/{warmup_iterations}")
                _run_direct_expert_ffn_once(routed_expert, config, mesh_device, expert_id, rows)

            fused_times_s = []
            for iteration in range(iterations):
                tt_input = _create_direct_expert_input(config, mesh_device, rows)
                ttnn.synchronize_device(mesh_device)
                start_s = time.perf_counter()
                output = routed_expert._expert_ffn(
                    tt_input,
                    routed_expert.gate_projs[expert_id],
                    routed_expert.up_projs[expert_id],
                    routed_expert.down_projs[expert_id],
                    out=None,
                    expert_id=expert_id,
                )
                ttnn.synchronize_device(mesh_device)
                fused_times_s.append(time.perf_counter() - start_s)
                ttnn.deallocate(output)
                ttnn.deallocate(tt_input)

            routed_expert.breakdown_expert_ffn = True
            routed_expert.measure_expert_time = False
            routed_expert.record_expert_ffn_breakdown = False
            routed_expert.last_expert_ffn_breakdowns = []
            for warmup_idx in range(warmup_iterations):
                logger.info(f"MoE opt expert FFN sweep stage warmup rows={rows} {warmup_idx + 1}/{warmup_iterations}")
                _run_direct_expert_ffn_once(routed_expert, config, mesh_device, expert_id, rows)

            routed_expert.record_expert_ffn_breakdown = True
            routed_expert.last_expert_ffn_breakdowns = []
            for iteration in range(iterations):
                _run_direct_expert_ffn_once(routed_expert, config, mesh_device, expert_id, rows)
                if routed_expert.last_expert_ffn_breakdowns:
                    routed_expert.last_expert_ffn_breakdowns[-1]["iteration"] = iteration

            breakdowns = list(routed_expert.last_expert_ffn_breakdowns)
            if not breakdowns:
                logger.warning(f"Skipping sweep row={rows}: no staged breakdown was recorded")
                continue

            duration_keys = ("gate_mm_silu_ms", "up_mm_ms", "mul_ms", "down_mm_ms", "total_ms", "matmul_only_ms")
            totals = {key: sum(float(item.get(key, 0.0)) for item in breakdowns) for key in duration_keys}
            samples = len(breakdowns)
            avg = {key: totals[key] / samples for key in duration_keys}
            matmul_flops = 2.0 * rows * config.emb_dim * config.hidden_dim
            total_matmul_flops = 3.0 * matmul_flops

            def tflops(flops_value: float, ms_value: float) -> float:
                return flops_value / max(ms_value / 1000.0, 1e-12) / 1e12

            fused_avg_s = statistics.mean(fused_times_s)
            first = breakdowns[0]
            record = {
                "name": config.name,
                "expert_id": expert_id,
                "rows": rows,
                "k_hidden_dim": config.hidden_dim,
                "emb_dim": config.emb_dim,
                "input_dtype": str(config.routed_activations_dtype).replace("DataType.", ""),
                "weight_dtype": str(config.routed_weights_dtype).replace("DataType.", ""),
                "warmup_iterations": warmup_iterations,
                "measured_iterations": iterations,
                "fused_op_avg_ms": fused_avg_s * 1000.0,
                "fused_op_tflops": total_matmul_flops / max(fused_avg_s, 1e-12) / 1e12,
                "stage_total_ms": avg["total_ms"],
                "gate_mm_silu_ms": avg["gate_mm_silu_ms"],
                "up_mm_ms": avg["up_mm_ms"],
                "mul_reshard_ms": avg["mul_ms"],
                "down_mm_ms": avg["down_mm_ms"],
                "gate_tflops": tflops(matmul_flops * samples, totals["gate_mm_silu_ms"]),
                "up_tflops": tflops(matmul_flops * samples, totals["up_mm_ms"]),
                "down_tflops": tflops(matmul_flops * samples, totals["down_mm_ms"]),
                "matmul_only_tflops": tflops(total_matmul_flops * samples, totals["matmul_only_ms"]),
                "total_tflops": tflops(total_matmul_flops * samples, totals["total_ms"]),
                "device_core_grid": first.get("device_core_grid", "unknown"),
                "device_compute_cores": int(first.get("device_compute_cores", 0)),
                "gate_up_core_grid": first.get("gate_up_core_grid", "unknown"),
                "gate_up_cores": int(first.get("gate_up_cores", 0)),
                "mul_core_grid": first.get("mul_core_grid", "unknown"),
                "mul_cores": int(first.get("mul_cores", 0)),
                "down_core_grid": first.get("down_core_grid", "unknown"),
                "down_cores": int(first.get("down_cores", 0)),
                "stage_source": first.get("stage_source", "unknown"),
                "ffn_path": first.get("ffn_path", "unknown"),
                "matmul_program": first.get("matmul_program", "unknown"),
                "matmul_factory": first.get("matmul_factory", "unknown"),
            }
            records.append(record)
    finally:
        routed_expert.breakdown_expert_ffn = previous_breakdown
        routed_expert.record_expert_ffn_breakdown = previous_record_breakdown
        routed_expert.measure_expert_time = previous_measure

    if not records:
        return ""

    fieldnames = list(records[0].keys())
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    lines = [
        "",
        "=" * 88,
        "MOE PREFILL OPT EXPERT FFN SWEEP",
        "-" * 88,
        f"csv={csv_path}",
        "rows | fused_ms | fused_TF/s | stage_ms | gate_TF/s | up_TF/s | down_TF/s | total_TF/s | cores",
    ]
    for record in records:
        cores = f"g/u={record['gate_up_core_grid']}({record['gate_up_cores']}) down={record['down_core_grid']}({record['down_cores']})"
        lines.append(
            f"{record['rows']} | "
            f"{record['fused_op_avg_ms']:.3f} | "
            f"{record['fused_op_tflops']:.3f} | "
            f"{record['stage_total_ms']:.3f} | "
            f"{record['gate_tflops']:.3f} | "
            f"{record['up_tflops']:.3f} | "
            f"{record['down_tflops']:.3f} | "
            f"{record['total_tflops']:.3f} | "
            f"{cores} | {record['ffn_path']}"
        )
    lines.append("=" * 88)
    print("\n".join(lines), flush=True)
    return str(csv_path)


def _measure_expert_ffn_times(
    config: MoePerfConfig,
    tt_moe: TtMoe,
    tt_x: ttnn.Tensor,
    mesh_device: ttnn.MeshDevice,
) -> tuple[list[float], float]:
    if not config.measure_expert_times:
        return [], 0.0

    routed_expert = getattr(tt_moe, "routed_expert", None)
    if routed_expert is None or not hasattr(routed_expert, "measure_expert_time"):
        logger.warning("Skipping expert timing pass: routed expert timing hook is unavailable")
        return [], 0.0

    logger.info("MoE opt expert timing pass 1/1")
    previous = routed_expert.measure_expert_time
    routed_expert.measure_expert_time = True
    try:
        timing_pass_s = _run_forward_once(tt_moe, tt_x, mesh_device)
        timings = list(getattr(routed_expert, "last_expert_timings_s", []))
    finally:
        routed_expert.measure_expert_time = previous
    return timings, timing_pass_s


def _configure_mcast2d_actual_core_logging(config: MoePerfConfig) -> str:
    if not config.mcast2d_core_log:
        os.environ.pop("TTNN_MCAST2D_CORE_LOG", None)
        os.environ.pop("TTNN_MCAST2D_CORE_LOG_PATH", None)
        os.environ.pop("TTNN_MCAST2D_RUN_LABEL", None)
        return ""

    csv_path = Path(config.mcast2d_core_log_csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not config.mcast2d_core_log_append:
        csv_path.unlink(missing_ok=True)

    os.environ["TTNN_MCAST2D_CORE_LOG"] = "1"
    os.environ["TTNN_MCAST2D_CORE_LOG_PATH"] = str(csv_path)
    os.environ.setdefault("TTNN_MCAST2D_RUN_LABEL", config.name)
    return str(csv_path)


def _write_csv() -> None:
    if not _RESULTS:
        return
    fieldnames = list(_RESULTS[0].keys())
    with _CSV_PATH.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_RESULTS)


def _fmt_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _print_result_summary(result: dict, durations_s: list[float], flops: dict, memory: dict) -> None:
    duration_ms = [duration * 1000 for duration in durations_s]
    duration_text = ", ".join(f"{duration:.3f}" for duration in duration_ms)
    if result.get("expert_timing_source") == "direct_expert_ffn_breakdown":
        expert_timing_line = (
            "expert_timing: "
            f"enabled={'on' if result['measure_expert_times'] else 'off'} | "
            f"source={result.get('expert_timing_source', 'n/a')} | "
            f"iters={result.get('expert_ffn_breakdown_iterations', 1)} | "
            f"avg_pass_ms={_fmt_float(result.get('expert_timing_pass_ms', 0.0))}"
        )
    else:
        expert_timing_line = (
            "expert_timing: "
            f"enabled={'on' if result['measure_expert_times'] else 'off'} | "
            f"source={result.get('expert_timing_source', 'n/a')} | "
            f"pass_ms={_fmt_float(result.get('expert_timing_pass_ms', 0.0))} | "
            f"max=E{result.get('max_expert_tflops_expert', '')}:{_fmt_float(result.get('max_expert_tflops_per_s', 0.0))} TF/s | "
            f"min=E{result.get('min_expert_tflops_expert', '')}:{_fmt_float(result.get('min_expert_tflops_per_s', 0.0))} TF/s"
        )
    lines = [
        "",
        "=" * 88,
        "MOE PREFILL OPT RESULT",
        "-" * 88,
        (
            f"name={result['name']} | mesh={result['mesh_shape']} | gate={result['gate_mode']} | "
            f"iters={result['measured_iterations']} | warmup={result['warmup_iterations']}"
        ),
        (
            f"seq/chip={result['seq_len_per_chip']} | emb={result['emb_dim']} | hidden={result['hidden_dim']} | "
            f"experts={result['num_routed_experts']} | experts/chip={result['experts_per_chip']} | "
            f"topk={result['num_experts_per_tok']}"
        ),
        (
            f"shared_expert={'on' if result['use_shared_expert'] else 'off'} | "
            f"dynamic_expert_capacity={'on' if result['dynamic_expert_capacity'] else 'off'} | "
            f"routing_profile={result['routing_profile']} | "
            f"target_range={_fmt_float(result['frequency_max_percent'], 3)}%-{_fmt_float(result['frequency_min_percent'], 3)}%"
        ),
        f"target_top_freq_experts: {result['target_top_frequency_summary']}",
        f"target_bottom_freq_experts: {result['target_bottom_frequency_summary']}",
        f"selected_top_freq_experts: {result['selected_top_frequency_summary']}",
        f"selected_bottom_freq_experts: {result['selected_bottom_frequency_summary']}",
        "-" * 88,
        (
            "latency_ms: "
            f"avg={_fmt_float(result['avg_ms'])} | p50={_fmt_float(result['median_ms'])} | "
            f"min={_fmt_float(result['min_ms'])} | max={_fmt_float(result['max_ms'])}"
        ),
        f"iterations_ms: [{duration_text}]",
        "-" * 88,
        (
            "throughput: "
            f"est_total_tflops={_fmt_float(result['estimated_tflops'], 6)} | "
            f"est_tflops_per_s={_fmt_float(result['estimated_tflops_per_s'])} | "
            f"est_memory_gb={_fmt_float(result['estimated_memory_gb'], 6)} | "
            f"est_memory_bw_gb_s={_fmt_float(result['estimated_memory_bw_gb_s'])}"
        ),
        expert_timing_line,
        (
            "expert_ffn_fused_op: " + result.get("expert_ffn_fused_op_summary", "")
            if result.get("expert_ffn_fused_op_summary")
            else "expert_ffn_fused_op: off"
        ),
        (
            "expert_ffn_staged_breakdown: " + result.get("expert_ffn_breakdown_summary", "")
            if result.get("expert_ffn_breakdown_summary")
            else "expert_ffn_staged_breakdown: off"
        ),
        (
            "expert_subdevice_parallel: " + result.get("expert_subdevice_parallel_summary", "")
            if result.get("expert_subdevice_parallel_summary")
            else "expert_subdevice_parallel: off"
        ),
        (
            f"tokens={result['tokens']} | routed_assignments={result['routed_assignments']} | "
            f"routed_ffn_rows={result['routed_ffn_rows']} | "
            f"dispatch_buffer_tokens={result['actual_dispatch_buffer_token_size']} | "
            f"configured_dispatch_buffer_tokens={result['max_dispatch_buffer_token_size']} | "
            f"max_tokens_per_expert={result['max_dispatched_tokens_per_expert']}"
        ),
        "-" * 88,
        (
            "flops_breakdown_tflops: "
            f"gate={_fmt_float(flops['gate_flops'] / 1e12, 6)} | "
            f"shared_ffn={_fmt_float(flops['shared_ffn_flops'] / 1e12, 6)} | "
            f"routed_ffn={_fmt_float(flops['routed_ffn_flops'] / 1e12, 6)} | "
            f"topk_reduce={_fmt_float(flops['reduce_flops'] / 1e12, 6)}"
        ),
        (
            "memory_breakdown_gb: "
            f"gate={_fmt_float(memory['gate_bytes'] / 1e9, 6)} | "
            f"dispatch_combine={_fmt_float(memory['dispatch_combine_bytes'] / 1e9, 6)} | "
            f"routed_ffn_act={_fmt_float(memory['routed_ffn_activation_bytes'] / 1e9, 6)} | "
            f"routed_ffn_w={_fmt_float(memory['routed_ffn_weight_bytes'] / 1e9, 6)} | "
            f"shared_ffn={_fmt_float((memory['shared_ffn_activation_bytes'] + memory['shared_ffn_weight_bytes']) / 1e9, 6)} | "
            f"reduce={_fmt_float(memory['reduce_bytes'] / 1e9, 6)}"
        ),
        "-" * 88,
        f"csv={_CSV_PATH}",
    ]
    if result["cache_path"]:
        lines.append(f"cache={result['cache_path']}")
    if result["expert_frequency_path"]:
        lines.append(f"expert_frequency_path={result['expert_frequency_path']}")
    if result.get("expert_plot_path"):
        lines.append(f"expert_plot={result['expert_plot_path']}")
    if result.get("expert_ffn_sweep_csv_path"):
        lines.append(f"expert_ffn_sweep_csv={result['expert_ffn_sweep_csv_path']}")
    if result.get("mcast2d_actual_core_csv_path"):
        lines.append(f"mcast2d_actual_core_csv={result['mcast2d_actual_core_csv_path']}")
    lines.append("=" * 88)
    print("\n".join(lines), flush=True)


@pytest.mark.timeout(_env_int("MOE_PREFILL_OPT_TIMEOUT", 0))
@pytest.mark.parametrize(
    "mesh_device",
    [pytest.param(CONFIG.mesh_shape, id=f"mesh-{CONFIG.mesh_shape[0]}x{CONFIG.mesh_shape[1]}")],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [_device_params_for_config(CONFIG)], indirect=True)
def test_moe_prefill_opt_perf(mesh_device, device_params):
    config = CONFIG
    _validate_config(config, mesh_device)
    mcast2d_actual_core_csv_path = _configure_mcast2d_actual_core_logging(config)

    mesh_config = extract_mesh_config(mesh_device)
    dispatch_group_size = mesh_config.dispatch_group_size
    num_dispatch_groups = mesh_config.num_dispatch_groups
    experts_per_chip, metadata_len, max_dispatch_buffer_token_size, max_dispatched_tokens_per_expert = (
        compute_constants(
            config.seq_len_per_chip,
            config.num_routed_experts,
            config.num_experts_per_tok,
            mesh_device.get_num_devices(),
            dispatch_group_size,
            config.dispatch_buffer_capacity_factor,
        )
    )

    gate_weights, routed_weights, shared_weights, cache_path = _prepare_weights(config, mesh_device, experts_per_chip)
    routing_frequency = _load_expert_frequency_profile(config)
    tt_x = _create_input(config, mesh_device, dispatch_group_size)

    tt_moe = TtMoe(
        mesh_device=mesh_device,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
        experts_per_chip=experts_per_chip,
        num_routed_experts=config.num_routed_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        max_dispatch_buffer_token_size=max_dispatch_buffer_token_size,
        seq_len_per_chip=config.seq_len_per_chip,
        gate_weights=gate_weights,
        emb_dim=config.emb_dim,
        hidden_dim=config.hidden_dim,
        num_links=1,
        topology=ttnn.Topology.Linear,
        routed_expert_weights=routed_weights,
        shared_expert_weights=shared_weights,
        routed_expert_activations_dtype=config.routed_activations_dtype,
        routed_expert_weights_dtype=config.routed_weights_dtype,
        shared_expert_activations_dtype=config.shared_activations_dtype,
        shared_expert_weights_dtype=config.shared_weights_dtype,
        gate_fallback_mode=config.gate_mode,
        weight_cache_path=cache_path,
        layer_idx=0,
        overlap_shared_expert_with_dispatch=config.overlap_shared_expert_with_dispatch,
        deallocate_forward_input=False,
        routing_frequency=routing_frequency,
        routing_seed=config.seed,
        use_shared_expert=config.use_shared_expert,
        dynamic_expert_capacity=config.dynamic_expert_capacity,
    )
    ttnn.synchronize_device(mesh_device)

    for warmup_idx in range(config.warmup_iterations):
        logger.info(f"MoE opt warmup {warmup_idx + 1}/{config.warmup_iterations}")
        _run_forward_once(tt_moe, tt_x, mesh_device)

    durations_s = []
    for iteration in range(config.measured_iterations):
        logger.info(f"MoE opt measured iter {iteration + 1}/{config.measured_iterations}")
        durations_s.append(_run_forward_once(tt_moe, tt_x, mesh_device))

    expert_ffn_breakdown_summary = ""
    expert_ffn_fused_op_summary = ""
    if config.expert_ffn_breakdown:
        expert_times_s, expert_timing_pass_s, expert_ffn_fused_op_summary = _measure_expert_ffn_fused_opt(
            config, tt_moe, mesh_device, dispatch_group_size, routing_frequency
        )
        _, _, expert_ffn_breakdown_summary = _measure_expert_ffn_direct_breakdown(
            config, tt_moe, mesh_device, dispatch_group_size, routing_frequency
        )
    else:
        expert_times_s, expert_timing_pass_s = _measure_expert_ffn_times(config, tt_moe, tt_x, mesh_device)

    expert_ffn_sweep_csv_path = _measure_expert_ffn_row_sweep(config, tt_moe, mesh_device)
    expert_subdevice_parallel_summary = _measure_expert_ffn_subdevice_parallel(
        config,
        tt_moe,
        mesh_device,
        dispatch_group_size,
        routing_frequency,
    )

    flops = _estimated_flops(config, dispatch_group_size, routing_frequency)
    memory = _estimated_memory_bytes(config, dispatch_group_size, routing_frequency)
    selected_top_summary, selected_bottom_summary = _selection_frequency_summaries(
        config,
        dispatch_group_size,
        routing_frequency,
    )
    avg_ms = statistics.mean(durations_s) * 1000
    median_ms = statistics.median(durations_s) * 1000
    min_ms = min(durations_s) * 1000
    max_ms = max(durations_s) * 1000
    tflops_per_s = flops["total_flops"] / (statistics.mean(durations_s) * 1e12)
    memory_bw_gb_s = memory["total_bytes"] / (statistics.mean(durations_s) * 1e9)

    result = {
        "name": config.name,
        "mesh_shape": f"{mesh_device.shape[0]}x{mesh_device.shape[1]}",
        "dispatch_group_size": dispatch_group_size,
        "num_dispatch_groups": num_dispatch_groups,
        "seq_len_per_chip": config.seq_len_per_chip,
        "emb_dim": config.emb_dim,
        "hidden_dim": config.hidden_dim,
        "num_routed_experts": config.num_routed_experts,
        "experts_per_chip": experts_per_chip,
        "num_experts_per_tok": config.num_experts_per_tok,
        "use_shared_expert": config.use_shared_expert,
        "routing_profile": config.routing_profile,
        "frequency_max_percent": config.frequency_max_percent,
        "frequency_min_percent": config.frequency_min_percent,
        "dynamic_expert_capacity": config.dynamic_expert_capacity,
        "measure_expert_times": config.measure_expert_times or config.expert_ffn_breakdown,
        "expert_ffn_breakdown": config.expert_ffn_breakdown,
        "expert_ffn_breakdown_iterations": config.expert_ffn_breakdown_iterations,
        "expert_ffn_breakdown_summary": expert_ffn_breakdown_summary,
        "expert_ffn_fused_op_summary": expert_ffn_fused_op_summary,
        "expert_ffn_sweep_csv_path": expert_ffn_sweep_csv_path,
        "expert_ffn_sweep_rows": ",".join(str(row) for row in config.expert_ffn_sweep_rows),
        "expert_subdevice_parallel": config.expert_subdevice_parallel,
        "expert_subdevice_parallel_summary": expert_subdevice_parallel_summary,
        "mcast2d_actual_core_csv_path": mcast2d_actual_core_csv_path,
        "expert_timing_pass_ms": expert_timing_pass_s * 1000,
        "expert_timing_source": "direct_fused_routed_expert_ffn_opt" if config.expert_ffn_breakdown else "",
        "expert_timing_active_experts": 0,
        "expert_timing_total_ffn_ms": 0.0,
        "max_expert_tflops_expert": "",
        "max_expert_tflops_per_s": 0.0,
        "min_expert_tflops_expert": "",
        "min_expert_tflops_per_s": 0.0,
        "expert_frequency_path": config.expert_frequency_path,
        "target_top_frequency_summary": _frequency_summary(routing_frequency, largest=True),
        "target_bottom_frequency_summary": _frequency_summary(routing_frequency, largest=False),
        "selected_top_frequency_summary": selected_top_summary,
        "selected_bottom_frequency_summary": selected_bottom_summary,
        "gate_mode": config.gate_mode.value,
        "warmup_iterations": config.warmup_iterations,
        "measured_iterations": config.measured_iterations,
        "avg_ms": avg_ms,
        "median_ms": median_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "estimated_tflops": flops["total_flops"] / 1e12,
        "estimated_tflops_per_s": tflops_per_s,
        "estimated_memory_gb": memory["total_bytes"] / 1e9,
        "estimated_memory_bw_gb_s": memory_bw_gb_s,
        "tokens": int(flops["tokens"]),
        "routed_assignments": int(flops["routed_assignments"]),
        "routed_ffn_rows": int(flops["routed_ffn_rows"]),
        "max_dispatch_buffer_token_size": max_dispatch_buffer_token_size,
        "actual_dispatch_buffer_token_size": int(
            getattr(tt_moe, "last_dispatch_buffer_token_size", max_dispatch_buffer_token_size)
        ),
        "max_dispatched_tokens_per_expert": max_dispatched_tokens_per_expert,
        "cache_path": str(cache_path) if cache_path else "",
        "config": str(asdict(config)),
    }
    result["expert_plot_path"] = _plot_expert_counts_and_roofline(
        config,
        dispatch_group_size,
        routing_frequency,
        durations_s,
        result,
        expert_times_s,
    )
    _RESULTS.append(result)
    _write_csv()
    _print_result_summary(result, durations_s, flops, memory)

    ttnn.deallocate(tt_x)
