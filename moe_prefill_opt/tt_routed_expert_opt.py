# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of Routed Expert module for processing dispatched tokens.

This module processes tokens that have been dispatched to local experts.
Unlike TtSharedExpert, this module:
- Does NOT use CCL (no all-gather, no reduce-scatter)
- Processes tokens that are already dispatched to each device
- Each device holds weights for `experts_per_chip` local experts
"""

from pathlib import Path
import os
import time
from typing import Optional

import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.lightweightmodule import LightweightModule
from moe_prefill_opt.init_helpers_opt import ExpertMapping

COMPUTE_KERNEL_CONFIG_LOFI = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, #LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)


class TtRoutedExpert(LightweightModule):
    @staticmethod
    def check_cache_complete(cache_path: Path, cache_name_prefix: str, experts_per_chip: int) -> bool:
        """Check if all routed expert weight cache files exist."""
        from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import pattern_exists

        for local_expert_idx in range(experts_per_chip):
            for proj in ["gate", "up", "down"]:
                pattern = f"{cache_name_prefix}.local_{local_expert_idx}_{proj}*.tensorbin"
                if not pattern_exists(pattern, "RoutedExpert"):
                    logger.debug(f"TTNN cache missing: {cache_name_prefix}.local_{local_expert_idx}_{proj}")
                    return False
        return True

    @staticmethod
    def _convert_and_cache_expert_weights(
        torch_weights: list[dict] | None,
        experts_per_chip: int,
        mesh_device: ttnn.MeshDevice,
        weights_dtype: ttnn.DataType,
        cache_path: Path | None,
        cache_name_prefix: str | None,
        device: ttnn.MeshDevice | None = None,
        *,
        emb_dim: int | None = None,
        hidden_dim: int | None = None,
    ):
        """
        Shared logic for converting expert weights to ttnn with caching.

        Args:
            torch_weights: List of expert weight dicts, or None for cache-only loading.
                When None, emb_dim and hidden_dim must be provided.
            experts_per_chip: Number of experts per chip (8 for 8x4 mesh)
            mesh_device: Mesh device reference
            weights_dtype: Weight data type
            cache_path: Cache directory
            cache_name_prefix: Prefix for cache files
            device: None for cache-only, mesh_device for cache+load
            emb_dim: Required when torch_weights is None
            hidden_dim: Required when torch_weights is None

        Returns:
            (gate_projs, up_projs, down_projs) if device is not None, else None
        """
        from tqdm import tqdm

        def _cache_name(name):
            if cache_path is None or cache_name_prefix is None:
                return None
            return str(cache_path / f"{cache_name_prefix}.{name}")

        mesh_rows, mesh_cols = mesh_device.shape
        gate_tensors, up_tensors, down_tensors = [], [], []

        mode = "build-cache" if device is None else ("load-cache" if torch_weights is None else "convert")
        for local_expert_idx in tqdm(range(experts_per_chip), desc=f"Expert weights ({mode})"):
            if torch_weights is not None:
                gate_weights, up_weights, down_weights = ExpertMapping.gather_weights_for_mesh_distribution(
                    torch_weights, local_expert_idx, mesh_rows, mesh_cols, experts_per_chip
                )

                stacked_gate = torch.stack([w.T.contiguous() for w in gate_weights], dim=0)
                in_f, out_f = stacked_gate.shape[1], stacked_gate.shape[2]
                stacked_gate = stacked_gate.reshape(mesh_rows, mesh_cols, in_f, out_f)

                stacked_up = torch.stack([w.T.contiguous() for w in up_weights], dim=0).reshape(
                    mesh_rows, mesh_cols, in_f, out_f
                )

                stacked_down = torch.stack([w.T.contiguous() for w in down_weights], dim=0)
                in_f_down, out_f_down = stacked_down.shape[1], stacked_down.shape[2]
                stacked_down = stacked_down.reshape(mesh_rows, mesh_cols, in_f_down, out_f_down)
            else:
                assert emb_dim is not None and hidden_dim is not None
                stacked_gate = torch.empty(mesh_rows, mesh_cols, emb_dim, hidden_dim)
                stacked_up = torch.empty(mesh_rows, mesh_cols, emb_dim, hidden_dim)
                stacked_down = torch.empty(mesh_rows, mesh_cols, hidden_dim, emb_dim)

            mem = ttnn.DRAM_MEMORY_CONFIG if device else None
            mapper = ExpertMapping.get_weights_mesh_mapper(mesh_device)

            gate_tt = ttnn.as_tensor(
                stacked_gate,
                mesh_mapper=mapper,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                dtype=weights_dtype,
                memory_config=mem,
                cache_file_name=_cache_name(f"local_{local_expert_idx}_gate"),
            )
            up_tt = ttnn.as_tensor(
                stacked_up,
                mesh_mapper=mapper,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                dtype=weights_dtype,
                memory_config=mem,
                cache_file_name=_cache_name(f"local_{local_expert_idx}_up"),
            )
            down_tt = ttnn.as_tensor(
                stacked_down,
                mesh_mapper=mapper,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                dtype=weights_dtype,
                memory_config=mem,
                cache_file_name=_cache_name(f"local_{local_expert_idx}_down"),
            )

            if device is None:
                del gate_tt, up_tt, down_tt
            else:
                gate_tt = ttnn.squeeze(ttnn.squeeze(gate_tt, dim=0), dim=0)
                up_tt = ttnn.squeeze(ttnn.squeeze(up_tt, dim=0), dim=0)
                down_tt = ttnn.squeeze(ttnn.squeeze(down_tt, dim=0), dim=0)
                gate_tensors.append(gate_tt)
                up_tensors.append(up_tt)
                down_tensors.append(down_tt)

        return (gate_tensors, up_tensors, down_tensors) if device else None

    @staticmethod
    def build_ttnn_cache(
        torch_weights: list[dict],
        experts_per_chip: int,
        mesh_device: ttnn.MeshDevice,
        weights_dtype: ttnn.DataType,
        cache_path: Path,
        cache_name_prefix: str,
    ):
        """Build TTNN cache for routed experts without device copy."""
        TtRoutedExpert._convert_and_cache_expert_weights(
            torch_weights, experts_per_chip, mesh_device, weights_dtype, cache_path, cache_name_prefix, device=None
        )

    """
    TTNN implementation of Routed Expert module.

    Processes dispatched tokens through local experts. Each device holds
    `experts_per_chip` experts and processes the tokens dispatched to them.

    Architecture (per expert):
        gate_out = x @ gate_proj
        up_out = x @ up_proj
        activated = silu(gate_out) * up_out
        output = activated @ down_proj

    Weight Layout:
        - Each expert has gate_proj, up_proj, down_proj
        - Weights are NOT sharded across devices (each device has full local expert weights)
        - gate_proj, up_proj: (emb_dim, hidden_dim)
        - down_proj: (hidden_dim, emb_dim)
    """

    def __init__(
        self,
        mesh_device,
        experts_per_chip: int,
        global_expert_idx_table: ttnn.Tensor,
        emb_dim: int = 7 * 1024,
        hidden_dim: int = 2 * 1024,
        max_tokens: int = 1600,
        torch_weights: list[dict] = None,
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
        compute_kernel_config: ttnn.WormholeComputeKernelConfig = COMPUTE_KERNEL_CONFIG_LOFI,
        weight_cache_path: Optional[Path] = None,
        cache_name_prefix: Optional[str] = None,
        dynamic_expert_capacity: bool = False,
    ):
        """
        Initialize TtRoutedExpert module.

        Args:
            mesh_device: TTNN mesh device
            experts_per_chip: Number of local experts per chip
            emb_dim: Embedding dimension (default: 7168)
            hidden_dim: Hidden/intermediate dimension (default: 2048)
            max_tokens: Maximum tokens per expert (default: 1600, used for program config)
            torch_weights: Optional list of dicts with keys 'gate_proj', 'up_proj', 'down_proj'
                          containing torch tensors. Length must be num_devices * experts_per_chip
                          (total routed experts), with weights ordered by global expert index.
                          Note: torch weights are in HuggingFace format (out_features, in_features)
                          so they need to be transposed for TTNN matmul.
            activations_dtype: Data type for activations (default: bfloat8_b)
            weights_dtype: Data type for weights (default: bfloat4_b)
            compute_kernel_config: Compute kernel configuration
            global_expert_idx_table: TTNN tensor mapping local expert slots to global expert ids.
                          Produced by sharding ExpertMapping.create_global_expert_idx_table via
                          get_ep_mesh_mapper, so each device holds (1, 1, experts_per_chip) of
                          global ids. Required.
        """
        super().__init__()
        self.mesh_device = mesh_device
        self.experts_per_chip = experts_per_chip
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.max_tokens = max_tokens
        self.num_devices = mesh_device.get_num_devices()
        self.activations_dtype = activations_dtype
        self.weights_dtype = weights_dtype
        self.compute_kernel_config = compute_kernel_config
        self.weight_cache_path = weight_cache_path
        self.cache_name_prefix = cache_name_prefix
        self.global_expert_idx_table = global_expert_idx_table
        self.dynamic_expert_capacity = dynamic_expert_capacity
        self.measure_expert_time = False
        self.breakdown_expert_ffn = False
        self.record_expert_ffn_breakdown = True
        self.last_expert_timings_s: list[float] = []
        self.last_expert_ffn_breakdowns: list[dict] = []
        self.host_local_expert_counts: torch.Tensor | list[int] | None = None
        self.host_local_expert_offsets: torch.Tensor | list[int] | None = None

        total_experts = self.num_devices * experts_per_chip
        logger.debug(f"Initializing TtRoutedExpert with experts_per_chip={experts_per_chip}")
        logger.debug(f"emb_dim={emb_dim}, hidden_dim={hidden_dim}")
        logger.debug(f"Mesh shape: {mesh_device.shape}, num_devices={self.num_devices}, total_experts={total_experts}")

        # Store weights for each local expert
        # Each expert has (gate_proj, up_proj, down_proj)
        self.gate_projs = []
        self.up_projs = []
        self.down_projs = []

        self.gate_projs_pc = None
        self.up_projs_pc = None
        self.down_projs_pc = None

        if torch_weights is not None:
            assert len(torch_weights) == total_experts, (
                f"Expected {total_experts} expert weights (num_devices={self.num_devices} * "
                f"experts_per_chip={experts_per_chip}), got {len(torch_weights)}"
            )
            logger.debug(f"Creating weights from provided torch tensors ({total_experts} experts)")
            result = self._convert_and_cache_expert_weights(
                torch_weights,
                experts_per_chip,
                self.mesh_device,
                self.weights_dtype,
                self.weight_cache_path,
                self.cache_name_prefix,
                device=self.mesh_device,
            )
        elif weight_cache_path is not None:
            logger.debug(f"Loading weights from cache ({experts_per_chip} local experts)")
            result = self._convert_and_cache_expert_weights(
                None,
                experts_per_chip,
                self.mesh_device,
                self.weights_dtype,
                self.weight_cache_path,
                self.cache_name_prefix,
                device=self.mesh_device,
                emb_dim=emb_dim,
                hidden_dim=hidden_dim,
            )
        else:
            logger.debug(f"Creating dummy tensors for testing ({total_experts} experts)")
            torch_weights = []
            for _ in range(total_experts):
                torch_weights.append(
                    {
                        "gate_proj": torch.empty(hidden_dim, emb_dim),
                        "up_proj": torch.empty(hidden_dim, emb_dim),
                        "down_proj": torch.empty(emb_dim, hidden_dim),
                    }
                )
            result = self._convert_and_cache_expert_weights(
                torch_weights,
                experts_per_chip,
                self.mesh_device,
                self.weights_dtype,
                None,
                None,
                device=self.mesh_device,
            )

        assert result is not None, "Expected weight tensors to be returned when device is provided"
        self.gate_projs, self.up_projs, self.down_projs = result

    @staticmethod
    def _ceil_to_tile(value: int) -> int:
        return ((value + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE

    @staticmethod
    def _div_up(value: int, divisor: int) -> int:
        return (int(value) + int(divisor) - 1) // int(divisor)

    def _opt_ffn_core_metadata(self, rows: int) -> dict[str, int | str]:
        """Return the core grids used by the routed expert FFN timing path."""
        grid_x_max = 11
        grid_y_max = 8
        tile = ttnn.TILE_SIZE

        m_tiles = max(1, self._div_up(rows, tile))
        n_gate_tiles = max(1, self._div_up(self.hidden_dim, tile))
        n_down_tiles = max(1, self._div_up(self.emb_dim, tile))

        try:
            compute_grid = self.mesh_device.compute_with_storage_grid_size()
            compute_x = int(compute_grid.x)
            compute_y = int(compute_grid.y)
            device_grid = f"{compute_x}x{compute_y}"
            device_cores = compute_x * compute_y
        except Exception:
            compute_x = grid_x_max
            compute_y = 10
            device_grid = "unknown"
            device_cores = 0

        if m_tiles > 64:
            # routed_expert_ffn_opt intentionally falls back to routed_expert_ffn_default
            # for large M. That path asks TTNN to auto-generate matmul configs; with
            # the sweep tensors/weights in DRAM interleaved, the auto path selects the
            # 2D multicast matmul factory on the full compute grid.
            return {
                "device_core_grid": device_grid,
                "device_compute_cores": device_cores,
                "ffn_path": "routed_expert_ffn_default_fallback",
                "matmul_program": "auto_generated_MatmulMultiCoreReuseMultiCastProgramConfig",
                "matmul_factory": "MatmulMultiCoreReuseMcast2DProgramFactory",
                "gate_up_core_grid": f"{compute_x}x{compute_y}",
                "gate_up_cores": compute_x * compute_y,
                "gate_up_per_core_m_tiles": self._div_up(m_tiles, compute_y),
                "gate_up_per_core_n_tiles": self._div_up(n_gate_tiles, compute_x),
                "mul_core_grid": "auto",
                "mul_cores": 0,
                "down_core_grid": f"{compute_x}x{compute_y}",
                "down_cores": compute_x * compute_y,
                "down_per_core_m_tiles": self._div_up(m_tiles, compute_y),
                "down_per_core_n_tiles": self._div_up(n_down_tiles, compute_x),
                "max_stage_cores": compute_x * compute_y,
            }

        gate_up_grid_y_upper = max(1, min(self._div_up(m_tiles, 4), grid_y_max))
        gate_up_per_core_m = self._div_up(m_tiles, gate_up_grid_y_upper)
        gate_up_grid_y = max(1, min(grid_y_max, self._div_up(m_tiles, gate_up_per_core_m)))
        gate_up_per_core_n = self._div_up(n_gate_tiles, grid_x_max)
        gate_up_grid_x = max(1, min(grid_x_max, self._div_up(n_gate_tiles, gate_up_per_core_n)))

        down_grid_y_upper = max(1, min(self._div_up(m_tiles, 4), grid_y_max))
        down_per_core_m = self._div_up(m_tiles, down_grid_y_upper)
        down_grid_y = max(1, min(grid_y_max, self._div_up(m_tiles, down_per_core_m)))
        down_per_core_n = self._div_up(n_down_tiles, grid_x_max)

        gate_up_cores = gate_up_grid_x * gate_up_grid_y
        down_cores = grid_x_max * down_grid_y
        return {
            "device_core_grid": device_grid,
            "device_compute_cores": device_cores,
            "ffn_path": "routed_expert_ffn_opt_bh_small_m",
            "matmul_program": "MatmulMultiCoreReuseMultiCastProgramConfig",
            "matmul_factory": "MatmulMultiCoreReuseMcast2DProgramFactory",
            "gate_up_core_grid": f"{gate_up_grid_x}x{gate_up_grid_y}",
            "gate_up_cores": gate_up_cores,
            "gate_up_per_core_m_tiles": gate_up_per_core_m,
            "gate_up_per_core_n_tiles": gate_up_per_core_n,
            "mul_core_grid": f"{gate_up_grid_x}x{gate_up_grid_y}",
            "mul_cores": gate_up_cores,
            "down_core_grid": f"{grid_x_max}x{down_grid_y}",
            "down_cores": down_cores,
            "down_per_core_m_tiles": down_per_core_m,
            "down_per_core_n_tiles": down_per_core_n,
            "max_stage_cores": max(gate_up_cores, down_cores),
        }

    def _local_expert_counts(self, expert_token_counts: ttnn.Tensor) -> list[int] | None:
        if not self.dynamic_expert_capacity:
            return None

        if self.num_devices != 1:
            logger.warning("dynamic_expert_capacity is currently enabled only for single-device opt runs")
            return None

        if self.host_local_expert_counts is not None:
            counts_source = self.host_local_expert_counts
            if isinstance(counts_source, torch.Tensor):
                counts = counts_source.reshape(-1).tolist()
            else:
                counts = list(counts_source)
        else:
            counts = ttnn.to_torch(expert_token_counts, dtype=torch.int32).reshape(-1).tolist()
        if len(counts) < self.experts_per_chip:
            raise RuntimeError(f"Expected at least {self.experts_per_chip} expert counts, got {len(counts)}")
        return [int(count) for count in counts[: self.experts_per_chip]]

    @staticmethod
    def _host_values_to_list(values: torch.Tensor | list[int] | None) -> list[int] | None:
        if values is None:
            return None
        if isinstance(values, torch.Tensor):
            return [int(value) for value in values.reshape(-1).tolist()]
        return [int(value) for value in values]

    def _host_counts_for_local_experts(self) -> list[int] | None:
        counts = self._host_values_to_list(self.host_local_expert_counts)
        if counts is None:
            return None
        if len(counts) < self.experts_per_chip:
            raise RuntimeError(f"Expected at least {self.experts_per_chip} expert counts, got {len(counts)}")
        return counts[: self.experts_per_chip]

    def _host_offsets_for_local_experts(self, counts: list[int] | None) -> list[int] | None:
        offsets = self._host_values_to_list(self.host_local_expert_offsets)
        if offsets is not None:
            if len(offsets) < self.experts_per_chip:
                raise RuntimeError(f"Expected at least {self.experts_per_chip} expert offsets, got {len(offsets)}")
            return offsets[: self.experts_per_chip]
        if counts is None:
            return [0] if self.experts_per_chip == 1 else None
        offsets = []
        prefix = 0
        for count in counts[: self.experts_per_chip]:
            offsets.append(prefix)
            prefix += self._ceil_to_tile(int(count))
        return offsets

    @staticmethod
    def shard_expert_token_counts(
        mesh_device: ttnn.MeshDevice,
        expert_token_counts: torch.Tensor,
    ) -> ttnn.Tensor:
        """
        Convert and shard the expert token counts tensor across mesh devices.

        Args:
            mesh_device: The mesh device to place the tensor on
            expert_token_counts: Total tokens per expert (sparse per group, replicated across dispatch_group_size)
                Shape: (num_dispatch_groups, dispatch_group_size, num_routed_experts) - from get_gate_outputs()

        Returns:
            TTNN tensor sharded across mesh devices.
            Per-device shape: (1, num_routed_experts)
        """
        logger.debug(f"[shard_expert_token_counts] INPUT: expert_token_counts.shape={expert_token_counts.shape}")
        mesh_mapper = ttnn.ShardTensor2dMesh(
            mesh_device,
            mesh_shape=mesh_device.shape,
            dims=(1, 0),
        )
        result = ttnn.from_torch(
            expert_token_counts,
            mesh_mapper=mesh_mapper,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.uint32,
        )
        result = ttnn.squeeze(result, 0)
        logger.debug(f"[shard_expert_token_counts] OUTPUT: result.shape={result.shape}")
        return result

    def _cache_name(self, name: str) -> Optional[str]:
        if self.weight_cache_path is None or self.cache_name_prefix is None:
            return None
        return str(self.weight_cache_path / f"{self.cache_name_prefix}.{name}")

    def _create_random_weight(self, shape: tuple, name: str) -> ttnn.Tensor:
        """
        Allocate uninitialized weight tensor on device DRAM (fast, no host transfer).

        Args:
            shape: Weight shape (in_features, out_features) for TTNN matmul
            name: Weight name for logging

        Returns:
            Uninitialized TTNN tensor on device DRAM
        """
        logger.debug(f"Allocating uninitialized weight {name} with shape {shape} on device DRAM")

        tt_weight = ttnn.allocate_tensor_on_device(
            ttnn.Shape(shape),
            self.weights_dtype,
            ttnn.TILE_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        return tt_weight

    def _expert_ffn(
        self,
        x: ttnn.Tensor,
        gate_proj: ttnn.Tensor,
        up_proj: ttnn.Tensor,
        down_proj: ttnn.Tensor,
        out: Optional[ttnn.Tensor] = None,
        expert_id: Optional[int] = None,
        sub_device_id: Optional[ttnn.SubDeviceId] = None,
    ) -> ttnn.Tensor:
        """
        Single expert FFN computation.

        Args:
            x: Input tensor. Shape is (1, tokens, emb_dim) for the Blackhole path
                (after ttnn.narrow) or (tokens, emb_dim) for the Wormhole path
                (after tensor indexing).
            gate_proj: Gate projection weight (emb_dim, hidden_dim)
            up_proj: Up projection weight (emb_dim, hidden_dim)
            down_proj: Down projection weight (hidden_dim, emb_dim)
            out: Optional pre-allocated output tensor for in-place matmul result.
                When provided, the final matmul writes directly into this buffer.
                When None, a new tensor is allocated for the output.

        Returns:
            Output tensor matching the shape of ``x``.
        """
        if self.breakdown_expert_ffn:
            label = f"E{expert_id}" if expert_id is not None else "E?"
            rows = int(x.shape[0])
            breakdown: dict[str, float | int | str] = {
                "expert_id": -1 if expert_id is None else int(expert_id),
                "rows": rows,
                "emb_dim": int(x.shape[-1]),
                "hidden_dim": int(gate_proj.shape[-1]),
                "input_dtype": str(x.dtype).replace("DataType.", ""),
                "weight_dtype": str(gate_proj.dtype).replace("DataType.", ""),
            }
            breakdown.update(self._opt_ffn_core_metadata(rows))

            def timed_step(name: str, fn):
                previous_stage_label = os.environ.get("TTNN_MCAST2D_STAGE_LABEL")
                os.environ["TTNN_MCAST2D_STAGE_LABEL"] = f"{label}:{name}"
                try:
                    ttnn.synchronize_device(self.mesh_device)
                    start_s = time.perf_counter()
                    value = fn()
                    ttnn.synchronize_device(self.mesh_device)
                    breakdown[f"{name}_ms"] = (time.perf_counter() - start_s) * 1000.0
                    return value
                finally:
                    if previous_stage_label is None:
                        os.environ.pop("TTNN_MCAST2D_STAGE_LABEL", None)
                    else:
                        os.environ["TTNN_MCAST2D_STAGE_LABEL"] = previous_stage_label

            stage_gate = getattr(ttnn.experimental.deepseek_prefill, "routed_expert_ffn_opt_gate", None)
            stage_up = getattr(ttnn.experimental.deepseek_prefill, "routed_expert_ffn_opt_up", None)
            stage_mul_reshard = getattr(ttnn.experimental.deepseek_prefill, "routed_expert_ffn_opt_mul_reshard", None)
            stage_down = getattr(ttnn.experimental.deepseek_prefill, "routed_expert_ffn_opt_down", None)
            use_cpp_stage_ops = all((stage_gate, stage_up, stage_mul_reshard, stage_down))
            breakdown["stage_source"] = (
                "routed_expert_ffn_opt_cpp_stage_ops" if use_cpp_stage_ops else "python_ttnn_matmul_fallback"
            )

            if use_cpp_stage_ops:
                gate_out = timed_step(
                    "gate_mm_silu",
                    lambda: stage_gate(
                        x,
                        gate_proj,
                        compute_kernel_config=self.compute_kernel_config,
                        sub_device_id=sub_device_id,
                    ),
                )
                up_out = timed_step(
                    "up_mm",
                    lambda: stage_up(
                        x,
                        up_proj,
                        compute_kernel_config=self.compute_kernel_config,
                        sub_device_id=sub_device_id,
                    ),
                )
                activated = timed_step(
                    "mul",
                    lambda: stage_mul_reshard(gate_out, up_out, sub_device_id=sub_device_id),
                )
                output = timed_step(
                    "down_mm",
                    lambda: stage_down(
                        activated,
                        down_proj,
                        compute_kernel_config=self.compute_kernel_config,
                        output=out,
                        sub_device_id=sub_device_id,
                    ),
                )
            else:
                gate_out = timed_step(
                    "gate_mm_silu",
                    lambda: ttnn.matmul(
                        x,
                        gate_proj,
                        activation="silu",
                        compute_kernel_config=self.compute_kernel_config,
                    ),
                )
                up_out = timed_step(
                    "up_mm",
                    lambda: ttnn.matmul(x, up_proj, compute_kernel_config=self.compute_kernel_config),
                )

                def multiply_gate_up():
                    ttnn.multiply_(gate_out, up_out)
                    return gate_out

                activated = timed_step("mul", multiply_gate_up)

                down_kwargs = {"compute_kernel_config": self.compute_kernel_config}
                if out is not None:
                    down_kwargs["optional_output_tensor"] = out
                output = timed_step("down_mm", lambda: ttnn.matmul(activated, down_proj, **down_kwargs))

            gate_ms = float(breakdown["gate_mm_silu_ms"])
            up_ms = float(breakdown["up_mm_ms"])
            mul_ms = float(breakdown["mul_ms"])
            down_ms = float(breakdown["down_mm_ms"])
            total_ms = gate_ms + up_ms + mul_ms + down_ms
            matmul_only_ms = gate_ms + up_ms + down_ms
            breakdown["total_ms"] = total_ms
            breakdown["matmul_only_ms"] = matmul_only_ms

            matmul_flops = 2.0 * rows * self.emb_dim * self.hidden_dim
            total_flops = 3.0 * matmul_flops
            breakdown["gate_mm_silu_flops"] = matmul_flops
            breakdown["up_mm_flops"] = matmul_flops
            breakdown["down_mm_flops"] = matmul_flops
            breakdown["total_matmul_flops"] = total_flops
            breakdown["gate_mm_silu_tflops_per_s"] = matmul_flops / max(gate_ms / 1000.0, 1e-12) / 1e12
            breakdown["up_mm_tflops_per_s"] = matmul_flops / max(up_ms / 1000.0, 1e-12) / 1e12
            breakdown["down_mm_tflops_per_s"] = matmul_flops / max(down_ms / 1000.0, 1e-12) / 1e12
            breakdown["matmul_only_tflops_per_s"] = total_flops / max(matmul_only_ms / 1000.0, 1e-12) / 1e12
            breakdown["tflops_per_s"] = total_flops / max(total_ms / 1000.0, 1e-12) / 1e12
            if self.record_expert_ffn_breakdown:
                self.last_expert_ffn_breakdowns.append(breakdown)
                logger.info(
                    "EXPERT_FFN_BREAKDOWN "
                    f"{label} rows={breakdown['rows']} dtype={breakdown['input_dtype']}/{breakdown['weight_dtype']} "
                    f"gate_mm_silu={breakdown['gate_mm_silu_ms']:.3f}ms "
                    f"up_mm={breakdown['up_mm_ms']:.3f}ms "
                    f"mul_reshard={breakdown['mul_ms']:.3f}ms "
                    f"down_mm={breakdown['down_mm_ms']:.3f}ms "
                    f"source={breakdown['stage_source']} "
                    f"path={breakdown['ffn_path']} "
                    f"program={breakdown['matmul_program']} "
                    f"factory={breakdown['matmul_factory']} "
                    f"cores=device={breakdown['device_core_grid']}({breakdown['device_compute_cores']}) "
                    f"gate/up={breakdown['gate_up_core_grid']}({breakdown['gate_up_cores']}) "
                    f"mul={breakdown['mul_core_grid']}({breakdown['mul_cores']}) "
                    f"down={breakdown['down_core_grid']}({breakdown['down_cores']}) "
                    f"matmul_tflops=gate={breakdown['gate_mm_silu_tflops_per_s']:.3f} "
                    f"up={breakdown['up_mm_tflops_per_s']:.3f} "
                    f"down={breakdown['down_mm_tflops_per_s']:.3f} "
                    f"matmul_only={breakdown['matmul_only_tflops_per_s']:.3f} "
                    f"total={breakdown['total_ms']:.3f}ms "
                    f"tflops={breakdown['tflops_per_s']:.3f}"
                )

            if not use_cpp_stage_ops:
                ttnn.deallocate(up_out)
            ttnn.deallocate(activated)
            return output

        routed_expert_ffn = getattr(
            ttnn.experimental.deepseek_prefill,
            "routed_expert_ffn_opt",
            ttnn.experimental.deepseek_prefill.routed_expert_ffn,
        )
        return routed_expert_ffn(
            x,
            gate_proj,
            up_proj,
            down_proj,
            compute_kernel_config=self.compute_kernel_config,
            output=out,
            sub_device_id=sub_device_id,
        )

    def forward(
        self,
        dispatched_buffer: ttnn.Tensor,
        expert_token_counts: ttnn.Tensor,
        expert_region_offsets: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        Blackhole forward implementation using narrow and in-place writes.

        Pre-allocates the output tensor with empty_like and uses narrow to extract
        per-expert slices and write FFN results directly into the output buffer,
        avoiding extra allocations from unsqueeze/concat.

        Args:
            dispatched_buffer: Dispatched tokens
                shape: (max_dispatch_buffer_token_size, emb_dim)
            expert_token_counts: Token counts per expert per chip
                If provided, only processes tokens up to the count (currently unused,
                all tokens are processed for simplicity)
            expert_region_offsets: Expert region start offsets per expert
                (shared across source devices in a dispatch group). Produced by
                offset_cumsum. Shape per device: (1, num_routed_experts).

        Returns:
            expert_outputs: Expert output tensor, same shape as dispatched_buffer
        """
        logger.debug(f"Forward pass: dispatched_buffer shape={dispatched_buffer.shape}")

        # Convert input to activations dtype if needed
        if dispatched_buffer.dtype != self.activations_dtype:
            logger.warning(f"{dispatched_buffer.dtype=} typecasting to {self.activations_dtype}")
            dispatched_buffer = ttnn.typecast(dispatched_buffer, self.activations_dtype)

        # Process each local expert
        # dispatched_buffer: (experts_per_chip, max_tokens, emb_dim)
        # We process expert by expert and reassemble

        local_expert_counts = self._local_expert_counts(expert_token_counts)
        if self.measure_expert_time:
            self.last_expert_timings_s = [0.0 for _ in range(self.experts_per_chip)]
        else:
            self.last_expert_timings_s = []

        if self.num_devices == 1 and dispatched_buffer.dtype != ttnn.bfloat8_b:
            host_counts = self._host_counts_for_local_experts()
            host_offsets = self._host_offsets_for_local_experts(host_counts)
            if host_offsets is None:
                raise RuntimeError(
                    "Non-BF8 multi-expert single-device routed path needs host expert offsets; "
                    "use host_all routing or routed BF8 activations for the DeepSeek extract/insert path."
                )

            expert_slices = []
            for local_expert in range(self.experts_per_chip):
                signpost(f"Expert {local_expert+1}/{self.experts_per_chip}")
                valid_token_count = host_counts[local_expert] if host_counts is not None else None
                if valid_token_count == 0:
                    logger.debug(f"Expert {local_expert}: skipped, token_count=0")
                    continue

                max_tokens_for_expert = self.max_tokens
                if valid_token_count is not None:
                    max_tokens_for_expert = min(self._ceil_to_tile(valid_token_count), self.max_tokens)
                start_offset = int(host_offsets[local_expert])
                tokens = ttnn.slice(
                    dispatched_buffer,
                    [start_offset, 0],
                    [start_offset + max_tokens_for_expert, dispatched_buffer.shape[1]],
                )
                if tokens.layout != ttnn.TILE_LAYOUT:
                    tokens = ttnn.to_layout(tokens, ttnn.TILE_LAYOUT)
                if tokens.dtype != self.activations_dtype:
                    tokens = ttnn.typecast(tokens, self.activations_dtype)

                if valid_token_count is None:
                    logger.debug(
                        f"Expert {local_expert}: non-BF8 bypass offset={start_offset}, input shape {tokens.shape}"
                    )
                else:
                    logger.debug(
                        f"Expert {local_expert}: non-BF8 bypass token_count={valid_token_count}, "
                        f"offset={start_offset}, capacity={max_tokens_for_expert}, input shape {tokens.shape}"
                    )

                if self.measure_expert_time:
                    ttnn.synchronize_device(self.mesh_device)
                    expert_start_s = time.perf_counter()
                output = self._expert_ffn(
                    tokens,
                    self.gate_projs[local_expert],
                    self.up_projs[local_expert],
                    self.down_projs[local_expert],
                    out=None,
                    expert_id=local_expert,
                )
                if self.measure_expert_time:
                    ttnn.synchronize_device(self.mesh_device)
                    self.last_expert_timings_s[local_expert] = time.perf_counter() - expert_start_s
                logger.debug(f"Expert {local_expert}: non-BF8 bypass output shape {output.shape}")
                expert_slices.append(
                    {
                        "expert_id": local_expert,
                        "start_offset": start_offset,
                        "capacity": max_tokens_for_expert,
                        "token_count": valid_token_count,
                        "output": output,
                    }
                )
            return expert_slices

        expert_outputs = dispatched_buffer
        for local_expert in range(self.experts_per_chip):
            signpost(f"Expert {local_expert+1}/{self.experts_per_chip}")

            valid_token_count = None
            max_tokens_for_expert = self.max_tokens
            if local_expert_counts is not None:
                valid_token_count = local_expert_counts[local_expert]
                if valid_token_count == 0:
                    logger.debug(f"Expert {local_expert}: skipped, token_count=0")
                    continue
                max_tokens_for_expert = min(self._ceil_to_tile(valid_token_count), self.max_tokens)

            # Extract tokens for this expert using the deepseek_prefill extract op,
            # which uses expert_region_offsets and expert_token_counts to slice out
            # this expert's valid rows
            tokens = ttnn.experimental.deepseek_prefill.extract(
                dispatched_buffer,
                expert_region_offsets,
                expert_token_counts,
                self.global_expert_idx_table,
                local_expert_id=local_expert,
                max_dispatched_tokens_per_expert=max_tokens_for_expert,
            )
            if valid_token_count is None:
                logger.debug(f"Expert {local_expert}: input shape {tokens.shape}")
            else:
                logger.debug(
                    f"Expert {local_expert}: token_count={valid_token_count}, "
                    f"capacity={max_tokens_for_expert}, input shape {tokens.shape}"
                )

            # Run FFN. Optional timing synchronizes around the expert FFN only;
            # it is used by the perf harness in a separate instrumentation pass.
            if self.measure_expert_time:
                ttnn.synchronize_device(self.mesh_device)
                expert_start_s = time.perf_counter()
            output = self._expert_ffn(
                tokens,
                self.gate_projs[local_expert],
                self.up_projs[local_expert],
                self.down_projs[local_expert],
                out=None,
                expert_id=local_expert,
            )
            if self.measure_expert_time:
                ttnn.synchronize_device(self.mesh_device)
                self.last_expert_timings_s[local_expert] = time.perf_counter() - expert_start_s
            logger.debug(f"Expert {local_expert}: output shape {output.shape}")

            # Insert this expert's output back into the flat expert_outputs buffer at
            # the expert's region (determined by expert_region_offsets and expert_token_counts).
            expert_outputs = ttnn.experimental.deepseek_prefill.insert(
                expert_outputs,
                output,
                expert_region_offsets,
                expert_token_counts,
                self.global_expert_idx_table,
                local_expert_id=local_expert,
            )

        # Shape: (experts_per_chip, max_tokens, emb_dim)
        logger.debug(f"Final expert_outputs shape: {expert_outputs.shape}")

        return expert_outputs
