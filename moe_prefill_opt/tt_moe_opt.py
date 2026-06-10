# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of MoE module connecting all MoE components.

This module assembles the full MoE pipeline:
1. Dispatch: Route tokens to expert buffers
2. Routed Experts: Process tokens in expert-specific buffers
3. Shared Expert: Process original input (in parallel with routed path)
4. Combine: Reconstruct outputs to original token positions
5. Split Connection: Apply gate weights and sum expert contributions
6. Final: Add routed output + shared output
"""

import os
from pathlib import Path
from typing import Optional, Union

import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from moe_prefill_opt.init_helpers_opt import ExpertMapping, get_ep_mesh_mapper
from moe_prefill_opt.tt_combine_opt import TtCombineModule
from moe_prefill_opt.tt_dispatch_opt import TtDispatchModule
from moe_prefill_opt.tt_moe_gate_prefill_opt import GateComputeMode, TtMoEGateConfig, TtMoEGatePrefill
from moe_prefill_opt.tt_moe_intermediates_opt import TtMoEIntermediates
from moe_prefill_opt.tt_reduce_opt import TtReduceModule
from moe_prefill_opt.tt_routed_expert_opt import TtRoutedExpert
from moe_prefill_opt.tt_shared_expert_opt import TtSharedExpert


class TtMoe(LightweightModule):
    """
    TTNN implementation of complete MoE pipeline.

    Architecture:
        x → [Dispatch] → dispatched_buffer → [Routed Experts] → expert_outputs
                                                                      ↓
                                                               [Combine] → combined_output
                                                                      ↓
        x → [Shared Expert] → shared_output           [Split Connection] → routed_output
                                      ↓                        ↓
                                final = routed_output + shared_output

    Layout Flow:
        - Dispatch: ROW_MAJOR → ROW_MAJOR
        - Routed Expert: TILE_LAYOUT → TILE_LAYOUT (convert before/after)
        - Combine: ROW_MAJOR → ROW_MAJOR
        - Shared Expert: TILE_LAYOUT → TILE_LAYOUT
        - Split Connection: ROW_MAJOR (elementwise ops)
        - Final Add: ROW_MAJOR
    """

    @staticmethod
    def _configure_opt_gate_groups(
        gate_config: TtMoEGateConfig,
        num_routed_experts: int,
        num_experts_per_tok: int,
        gate_fallback_mode: GateComputeMode,
    ):
        """Keep DeepSeek gate defaults when valid, but allow small host-gate opt runs.

        The DeepSeek defaults use 8 groups, 4 selected groups, and therefore select
        the top-2 experts within each group. With only 8 routed experts this leaves
        one expert per group, so the host reference gate fails trying to take top-2.
        Device grouped-gate kernels are specialized for the DeepSeek defaults, so
        only host grouped-gate paths are adjusted here.
        """
        host_grouped_gate_modes = (GateComputeMode.HOST_ALL, GateComputeMode.HOST_GROUPED_GATE)
        if gate_fallback_mode not in host_grouped_gate_modes:
            return

        def is_valid(n_groups: int, topk_groups: int) -> bool:
            if n_groups <= 0 or topk_groups <= 0:
                return False
            if num_routed_experts % n_groups != 0 or n_groups % topk_groups != 0:
                return False
            experts_per_group = num_routed_experts // n_groups
            summed_experts_per_group = n_groups // topk_groups
            if summed_experts_per_group > experts_per_group:
                return False
            return num_experts_per_tok <= topk_groups * experts_per_group

        if is_valid(gate_config.n_expert_groups, gate_config.n_limited_groups):
            return

        candidate_groups = [
            groups
            for groups in range(min(DeepSeekV3Config.NUM_EXPERT_GROUPS, num_routed_experts), 0, -1)
            if num_routed_experts % groups == 0
        ]
        for groups in candidate_groups:
            # Selecting all groups makes summed_experts_per_group=1, which is the
            # most permissive host-gate policy for tiny synthetic expert counts.
            if is_valid(groups, groups):
                gate_config.n_expert_groups = groups
                gate_config.n_limited_groups = groups
                logger.info(
                    "Adjusted opt host gate groups for small MoE config: "
                    f"n_expert_groups={groups}, n_limited_groups={groups}"
                )
                return

        raise ValueError(
            "Unable to derive valid host gate grouping for "
            f"num_routed_experts={num_routed_experts}, num_experts_per_tok={num_experts_per_tok}"
        )

    @staticmethod
    def check_cache_complete(
        cache_path: Path,
        layer_idx: int,
        experts_per_chip: int,
        include_shared_expert: bool = True,
    ) -> bool:
        """Check if MoE cache is complete (gate + routed experts + shared expert)."""
        prefix = f"layer_{layer_idx}"
        if not TtMoEGatePrefill.check_cache_complete(cache_path, f"{prefix}.gate"):
            return False
        if not TtRoutedExpert.check_cache_complete(cache_path, f"{prefix}.routed_expert", experts_per_chip):
            return False
        if include_shared_expert and not TtSharedExpert.check_cache_complete(cache_path, f"{prefix}.shared_expert"):
            return False
        return True

    @staticmethod
    def build_ttnn_cache(
        gate_weights: dict | None,
        routed_expert_weights: list[dict] | None,
        shared_expert_weights: dict | None,
        experts_per_chip: int,
        emb_dim: int,
        hidden_dim: int,
        mesh_device: ttnn.MeshDevice,
        routed_expert_weights_dtype: ttnn.DataType,
        shared_expert_weights_dtype: ttnn.DataType,
        cache_path: Path,
        layer_idx: int,
    ):
        """Build TTNN cache for MoE (gate + routed experts + shared expert) without device copy."""
        # Build gate cache (delegate to TtMoEGatePrefill)
        if gate_weights:
            from moe_prefill_opt.tt_moe_gate_prefill_opt import TtMoEGateConfig, TtMoEGatePrefill

            # Create minimal config for caching
            gate_config = TtMoEGateConfig()
            gate_config.dim = emb_dim
            gate_config.n_routed_experts = gate_weights["weight"].shape[0]

            TtMoEGatePrefill.build_ttnn_cache(
                torch_weight=gate_weights["weight"],
                torch_bias=gate_weights["e_score_correction_bias"],
                config=gate_config,
                mesh_device=mesh_device,
                cache_path=cache_path,
                cache_name_prefix=f"layer_{layer_idx}.gate",
            )

        # Build routed expert cache
        if routed_expert_weights:
            TtRoutedExpert.build_ttnn_cache(
                routed_expert_weights,
                experts_per_chip,
                mesh_device,
                routed_expert_weights_dtype,
                cache_path,
                f"layer_{layer_idx}.routed_expert",
            )

        # Build shared expert cache
        if shared_expert_weights:
            TtSharedExpert.build_ttnn_cache(
                shared_expert_weights,
                emb_dim,
                hidden_dim,
                mesh_device,
                shared_expert_weights_dtype,
                cache_path,
                f"layer_{layer_idx}.shared_expert",
            )

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        dispatch_group_size: int,
        num_dispatch_groups: int,
        experts_per_chip: int,
        num_routed_experts: int,
        num_experts_per_tok: int,
        metadata_len: int,
        max_dispatched_tokens_per_expert: int,
        max_dispatch_buffer_token_size: int,
        seq_len_per_chip: int,
        gate_weights: dict,
        emb_dim: int = DeepSeekV3Config.EMB_SIZE,
        hidden_dim: int = DeepSeekV3Config.MOE_INTERMEDIATE_SIZE,
        num_links: Union[int, tuple[int, int]] = 1,
        topology: Union[ttnn.Topology, tuple[ttnn.Topology, ttnn.Topology]] = ttnn.Topology.Linear,
        routed_expert_weights: list[dict] = None,
        shared_expert_weights: dict = None,
        routed_expert_activations_dtype=ttnn.bfloat8_b,
        routed_expert_weights_dtype=ttnn.bfloat4_b,
        shared_expert_activations_dtype=ttnn.bfloat16,
        shared_expert_weights_dtype=ttnn.bfloat8_b,
        gate_fallback_mode: GateComputeMode = GateComputeMode.HOST_ALL,
        weight_cache_path: Optional[Path] = None,
        layer_idx: int = 0,
        overlap_shared_expert_with_dispatch: bool = True,
        deallocate_forward_input: bool = True,
        routing_frequency: Optional[torch.Tensor] = None,
        routing_seed: int = 0,
        use_shared_expert: bool = True,
        dynamic_expert_capacity: bool = False,
    ):
        """
        Initialize TtMoe module.

        Args:
            mesh_device: TTNN mesh device
            dispatch_group_size: Number of chips in each dispatch group
            num_dispatch_groups: Number of parallel dispatch groups
            experts_per_chip: Number of experts per chip
            num_routed_experts: Total number of routed experts
            num_experts_per_tok: Number of experts each token routes to
            metadata_len: Length of metadata per token
            max_dispatched_tokens_per_expert: Per-expert theoretical upper bound on the
                number of tokens any single expert may receive (full sequence length).
            max_dispatch_buffer_token_size: Total token capacity of the flat dispatch
                buffer per chip (shared across all local experts).
            seq_len_per_chip: Sequence length per chip
            emb_dim: Embedding dimension (default: 7168)
            hidden_dim: Hidden/intermediate dimension (default: 2048)
            num_links: Number of ethernet links for CCL. Int applies to both axes;
                       tuple (row, col) allows separate config per axis.
            topology: CCL topology. Scalar applies to both axes;
                      tuple (row, col) allows separate config per axis.
            routed_expert_weights: Optional list of dicts with gate_proj, up_proj, down_proj
                                   per expert. Length must be experts_per_chip.
            shared_expert_weights: Optional dict with gate_proj, up_proj, down_proj
                                   for shared expert.
            routed_expert_activations_dtype: Data type for routed expert activations
            routed_expert_weights_dtype: Data type for routed expert weights
            shared_expert_activations_dtype: Data type for shared expert activations
            shared_expert_weights_dtype: Data type for shared expert weights
            gate_weights: Dict with "weight" and "e_score_correction_bias" keys for gate
            gate_fallback_mode: Fallback mode for gate (default: HOST_ALL)
            overlap_shared_expert_with_dispatch: If True, run the shared expert and dispatch
                on disjoint sub-devices so they overlap on-chip. If False, skip sub-device
                setup and run them sequentially on the full Tensix grid.
            deallocate_forward_input: If True, preserve the reference implementation's memory
                behavior by deallocating ``x`` after dispatch. Perf harnesses can set this to
                False to reuse a preloaded input tensor across warmup/measurement iterations.
            routing_frequency: Optional expert frequency profile for host-gate opt
                experiments. When provided, host gate modes sample top-k routing from
                this profile instead of deriving it from random synthetic gate weights.
            use_shared_expert: If False, skip the DeepSeek shared expert path. This is
                useful for Qwen3-style MoE layers, which have routed experts but no
                shared expert.
            dynamic_expert_capacity: If True, single-device opt runs process each
                expert with a tile-rounded capacity based on that expert's actual
                token count instead of the full sequence-length capacity.
        """
        super().__init__()
        self.mesh_device = mesh_device
        self.dispatch_group_size = dispatch_group_size
        self.num_dispatch_groups = num_dispatch_groups
        self.experts_per_chip = experts_per_chip
        self.num_routed_experts = num_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.seq_len_per_chip = seq_len_per_chip
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.use_shared_expert = use_shared_expert
        self.overlap_shared_expert_with_dispatch = overlap_shared_expert_with_dispatch and use_shared_expert
        self.deallocate_forward_input = deallocate_forward_input
        self.dynamic_expert_capacity = dynamic_expert_capacity
        self.configured_dispatch_buffer_token_size = max_dispatch_buffer_token_size
        self.last_dispatch_buffer_token_size = max_dispatch_buffer_token_size

        # Unpack row/col CCL config
        if isinstance(num_links, tuple):
            self.row_num_links, self.col_num_links = num_links
        else:
            self.row_num_links = self.col_num_links = num_links

        if isinstance(topology, tuple):
            self.row_topology, self.col_topology = topology
        else:
            self.row_topology = self.col_topology = topology

        # Always create dispatch table at init (static tensor) - needed by gate and dispatch module
        expert_dispatch_table = ExpertMapping.create_dispatch_table(
            num_routed_experts, dispatch_group_size, num_dispatch_groups
        )

        # Build gate internally
        gate_config = TtMoEGateConfig()
        gate_config.dim = emb_dim
        gate_config.sp_dim = seq_len_per_chip
        gate_config.n_routed_experts = num_routed_experts
        gate_config.n_activated_experts = num_experts_per_tok
        gate_config.ccl_config["NUM_LINKS"] = self.col_num_links if isinstance(num_links, tuple) else num_links
        self._configure_opt_gate_groups(gate_config, num_routed_experts, num_experts_per_tok, gate_fallback_mode)

        # Handle cache-only case (gate_weights=None)
        if gate_weights is not None:
            gate_weight = gate_weights["weight"]
            gate_bias = gate_weights["e_score_correction_bias"]
        else:
            # Dummy tensors for cache load (ignored when cache exists)
            gate_weight = torch.empty(num_routed_experts, emb_dim)
            gate_bias = torch.empty(num_routed_experts)

        self.gate = TtMoEGatePrefill(
            gate_config,
            mesh_device,
            dispatch_table=expert_dispatch_table,
            experts_per_chip=experts_per_chip,
            weight=gate_weight,
            bias=gate_bias,
            fallback_mode=gate_fallback_mode,
            weight_cache_path=weight_cache_path,
            cache_name_prefix=f"layer_{layer_idx}.gate",
            routing_frequency=routing_frequency,
            routing_seed=routing_seed,
        )
        logger.debug(f"Initializing TtMoe")
        logger.debug(f"  mesh_device.shape={mesh_device.shape}")
        logger.debug(f"  dispatch_group_size={dispatch_group_size}, num_dispatch_groups={num_dispatch_groups}")
        logger.debug(f"  experts_per_chip={experts_per_chip}, num_routed_experts={num_routed_experts}")
        logger.debug(f"  num_experts_per_tok={num_experts_per_tok}")
        logger.debug(f"  seq_len_per_chip={seq_len_per_chip}, emb_dim={emb_dim}, hidden_dim={hidden_dim}")

        self.tt_expert_dispatch_table = TtDispatchModule.shard_expert_dispatch_table(
            mesh_device, expert_dispatch_table, dispatch_axis=0
        )

        # ========================================
        # Sub-devices: when overlap is enabled, split the Tensix grid into a "dispatch"
        # strip and a "shared expert" strip so the two ops run on disjoint cores and the
        # Fast-Dispatch per-sub-device counters let them overlap on-chip.
        #   sub-device 0 (dispatch_sd):     rows [0, dispatch_sd_rows)
        #   sub-device 1 (shared_sd):       rows [dispatch_sd_rows, grid_y)
        # When overlap is disabled, both ops run sequentially on the full grid and no
        # sub-device manager is created.
        # ========================================
        if self.overlap_shared_expert_with_dispatch:
            dispatch_sd_rows = 1
            grid = mesh_device.compute_with_storage_grid_size()
            grid_x, grid_y = grid.x, grid.y
            assert 0 < dispatch_sd_rows < grid_y, f"dispatch_sd_rows={dispatch_sd_rows} must be in (0, grid_y={grid_y})"
            dispatch_cores = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, dispatch_sd_rows - 1))}
            )
            shared_cores = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, dispatch_sd_rows), ttnn.CoreCoord(grid_x - 1, grid_y - 1))}
            )
            dispatch_sd = ttnn.SubDevice([dispatch_cores])
            shared_sd = ttnn.SubDevice([shared_cores])
            self.sd_manager_id = mesh_device.create_sub_device_manager([dispatch_sd, shared_sd], 0)
            self.dispatch_sd_id = ttnn.SubDeviceId(0)
            self.shared_sd_id = ttnn.SubDeviceId(1)
            # Stash the CoreRangeSet of the shared sub-device so TtSharedExpert can build
            # sub-device-confined shard_specs in Python without a C++ worker_cores binding.
            self.shared_sd_cores = shared_cores
            logger.debug(
                f"Sub-devices: grid={grid_x}x{grid_y}, dispatch=rows[0,{dispatch_sd_rows}), "
                f"shared=rows[{dispatch_sd_rows},{grid_y})"
            )
        else:
            self.sd_manager_id = None
            self.dispatch_sd_id = None
            self.shared_sd_id = None
            self.shared_sd_cores = None
            logger.debug("Sub-devices disabled: shared expert and dispatch will run sequentially")

        # Initialize dispatch module (row axis: axis 0)
        self.dispatch_module = TtDispatchModule(
            mesh_device=mesh_device,
            dispatch_group_size=dispatch_group_size,
            experts_per_chip=experts_per_chip,
            num_routed_experts=num_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            metadata_len=metadata_len,
            max_dispatch_buffer_token_size=max_dispatch_buffer_token_size,
            seq_len_per_chip=seq_len_per_chip,
            emb_dim=emb_dim,
            cluster_axis=0,
            num_links=self.row_num_links,
            topology=self.row_topology,
            subdevice_id=self.dispatch_sd_id,
        )

        # Initialize combine module (row axis: axis 0)
        self.combine_module = TtCombineModule(
            mesh_device=mesh_device,
            dispatch_group_size=dispatch_group_size,
            num_dispatch_groups=num_dispatch_groups,
            experts_per_chip=experts_per_chip,
            num_experts_per_tok=num_experts_per_tok,
            seq_len_per_chip=seq_len_per_chip,
            cluster_axis=0,
            num_links=self.row_num_links,
            topology=self.row_topology,
            init_zeros=False,
        )

        # Build (group, chip, local_expert) -> global expert id table, sharded
        # across the EP mesh so each device holds (1, 1, experts_per_chip).
        # Then squeeze the two leading singleton dims so each device has a 1D
        # (experts_per_chip,) lookup vector (required by extract/insert validators).
        global_expert_idx_tt = ttnn.from_torch(
            ExpertMapping.create_global_expert_idx_table(
                experts_per_chip=experts_per_chip,
                dispatch_group_size=dispatch_group_size,
                num_dispatch_groups=num_dispatch_groups,
            ),
            mesh_mapper=get_ep_mesh_mapper(mesh_device),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.uint32,
        )
        global_expert_idx_tt = ttnn.squeeze(global_expert_idx_tt, 0)
        global_expert_idx_tt = ttnn.squeeze(global_expert_idx_tt, 0)

        # Initialize routed expert
        self.routed_expert = TtRoutedExpert(
            mesh_device=mesh_device,
            experts_per_chip=experts_per_chip,
            global_expert_idx_table=global_expert_idx_tt,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            max_tokens=max_dispatched_tokens_per_expert,
            torch_weights=routed_expert_weights,
            activations_dtype=routed_expert_activations_dtype,
            weights_dtype=routed_expert_weights_dtype,
            weight_cache_path=weight_cache_path,
            cache_name_prefix=f"layer_{layer_idx}.routed_expert",
            dynamic_expert_capacity=dynamic_expert_capacity,
        )

        # Initialize shared expert (col axis: axis 1). Qwen3 MoE has no shared
        # expert, so the opt harness can skip this path for Qwen-style runs.
        if self.use_shared_expert:
            self.shared_expert = TtSharedExpert(
                mesh_device=mesh_device,
                emb_dim=emb_dim,
                hidden_dim=hidden_dim,
                torch_weights=shared_expert_weights,
                num_links=self.col_num_links,
                topology=self.col_topology,
                activations_dtype=shared_expert_activations_dtype,
                weights_dtype=shared_expert_weights_dtype,
                weight_cache_path=weight_cache_path,
                cache_name_prefix=f"layer_{layer_idx}.shared_expert",
                subdevice_id=self.shared_sd_id,
                subdevice_cores=self.shared_sd_cores,
            )
        else:
            self.shared_expert = None

        # Initialize reduce module for post-combine reduction (col axis: axis 1)
        # topk_dim=3 because combine output is (1, dispatch_group_size, seq_len, topk, emb_dim)
        # cluster_axis=1 to reduce-scatter across TP axis (same as shared expert)
        self.reduce_module = TtReduceModule(
            mesh_device=mesh_device,
            topk_dim=3,  # topk is at dim 3 in 5D tensor from combine
            cluster_axis=1,  # TP axis for reduce-scatter
            num_links=self.col_num_links,
            topology=self.col_topology,
        )

        # Load debug flags from environment
        self.debug_token_count = os.getenv("TT_DS_PREFILL_DEBUG_TOKEN_COUNT", "0").lower() in ("1", "true", "yes")

        logger.debug("TtMoe initialization complete")

    def _single_device_local_dispatch(
        self,
        x: ttnn.Tensor,
        scores: ttnn.Tensor,
        indices: ttnn.Tensor,
        tt_expert_offsets: ttnn.Tensor,
        host_expert_offsets: torch.Tensor | None = None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, torch.Tensor]:
        """Host-pack an all-local dispatch buffer for 1x1 opt runs.

        The DeepSeek dispatch kernel is fabric-oriented and currently expects a
        mesh neighbor even when all routed experts live on the same device. This
        opt-only path keeps the same flat expert-region layout but avoids the
        fabric dispatch op for single-device experiments.
        """
        host_x = ttnn.to_torch(x).to(torch.bfloat16)
        host_indices = ttnn.to_torch(indices, dtype=torch.int32).reshape(
            self.dispatch_group_size, self.seq_len_per_chip, self.num_experts_per_tok
        )
        host_scores = ttnn.to_torch(scores).reshape(
            self.dispatch_group_size, self.seq_len_per_chip, self.num_experts_per_tok
        )
        if host_expert_offsets is None:
            host_offsets = ttnn.to_torch(tt_expert_offsets, dtype=torch.int32).reshape(-1)
        else:
            host_offsets = host_expert_offsets.reshape(-1).to(torch.int32)
        next_offsets = host_offsets.clone()

        max_tokens_limit = self.dispatch_module.max_dispatch_buffer_token_size
        hidden_dim = host_x.shape[-1]
        host_counts = torch.bincount(
            host_indices.reshape(-1).to(torch.int64),
            minlength=self.num_routed_experts,
        ).to(torch.int64)
        padded_counts = ((host_counts + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        required_tokens = int((host_offsets.to(torch.int64) + padded_counts).max().item()) if host_counts.numel() else 0
        max_tokens = max(ttnn.TILE_SIZE, required_tokens)
        if required_tokens > max_tokens_limit:
            logger.debug(
                f"[TtMoe._single_device_local_dispatch] compact dispatch requires {required_tokens} rows, "
                f"which exceeds configured max_dispatch_buffer_token_size={max_tokens_limit}; "
                "using the larger tile-padded compact buffer for the single-device host dispatch path"
            )
        elif max_tokens < max_tokens_limit:
            logger.debug(
                f"[TtMoe._single_device_local_dispatch] compact dispatch buffer rows={max_tokens} "
                f"(configured capacity={max_tokens_limit})"
            )

        self.last_dispatch_buffer_token_size = int(max_tokens)
        dispatch_buffer = torch.zeros((1, 1, max_tokens, hidden_dim), dtype=torch.bfloat16)
        metadata = torch.full((1, 1, max_tokens, self.dispatch_module.metadata_len), -1, dtype=torch.int32)

        for source_idx in range(host_indices.shape[0]):
            for token_idx in range(host_indices.shape[1]):
                for topk_idx in range(host_indices.shape[2]):
                    routed_expert = int(host_indices[source_idx, token_idx, topk_idx].item())
                    if routed_expert < 0 or routed_expert >= self.num_routed_experts:
                        continue

                    page_idx = int(next_offsets[routed_expert].item())
                    next_offsets[routed_expert] += 1
                    if page_idx >= max_tokens:
                        continue

                    dispatch_buffer[0, 0, page_idx, :] = host_x[source_idx, token_idx, :]
                    metadata[0, 0, page_idx, 0] = 0
                    metadata[0, 0, page_idx, 1] = token_idx
                    metadata[0, 0, page_idx, 2] = topk_idx
                    metadata[0, 0, page_idx, 3] = routed_expert
                    metadata[0, 0, page_idx, 4] = int(float(host_scores[source_idx, token_idx, topk_idx]))

        mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
        tt_dispatch_buffer = ttnn.from_torch(
            dispatch_buffer,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mesh_mapper,
        )
        tt_metadata = ttnn.from_torch(
            metadata,
            device=self.mesh_device,
            dtype=ttnn.int32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mesh_mapper,
        )
        return tt_dispatch_buffer, tt_metadata, metadata

    def _single_device_local_combine(self, expert_outputs: ttnn.Tensor, metadata: torch.Tensor) -> ttnn.Tensor:
        """Host-unpack expert outputs into [1, 1, seq, topk, hidden] for 1x1 opt runs."""
        host_expert_outputs = ttnn.to_torch(expert_outputs).reshape(-1, self.emb_dim).to(torch.bfloat16)
        flat_metadata = metadata.reshape(-1, self.dispatch_module.metadata_len)
        combined = self._empty_local_combined_tensor()

        valid_rows = torch.nonzero(flat_metadata[:, 1] >= 0, as_tuple=False).flatten()
        for row_idx in valid_rows.tolist():
            token_idx = int(flat_metadata[row_idx, 1].item())
            topk_idx = int(flat_metadata[row_idx, 2].item())
            if token_idx < self.seq_len_per_chip and topk_idx < self.num_experts_per_tok:
                combined[0, 0, token_idx, topk_idx, :] = host_expert_outputs[row_idx, :]

        return self._combined_to_device(combined)

    def _empty_local_combined_tensor(self) -> torch.Tensor:
        return torch.zeros(
            (1, self.dispatch_group_size, self.seq_len_per_chip, self.num_experts_per_tok, self.emb_dim),
            dtype=torch.bfloat16,
        )

    def _combined_to_device(self, combined: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            combined,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _single_device_local_combine_expert_slices(self, expert_slices: list[dict], metadata: torch.Tensor) -> ttnn.Tensor:
        """Host-combine per-expert output slices from the non-BF8 routed path."""
        flat_metadata = metadata.reshape(-1, self.dispatch_module.metadata_len)
        combined = self._empty_local_combined_tensor()

        for expert_slice in expert_slices:
            expert_id = int(expert_slice["expert_id"])
            start_offset = int(expert_slice["start_offset"])
            host_output = ttnn.to_torch(expert_slice["output"]).reshape(-1, self.emb_dim).to(torch.bfloat16)
            valid_rows = torch.nonzero(flat_metadata[:, 3] == expert_id, as_tuple=False).flatten()
            for row_idx in valid_rows.tolist():
                token_idx = int(flat_metadata[row_idx, 1].item())
                topk_idx = int(flat_metadata[row_idx, 2].item())
                local_row = int(row_idx - start_offset)
                if (
                    token_idx < self.seq_len_per_chip
                    and topk_idx < self.num_experts_per_tok
                    and 0 <= local_row < host_output.shape[0]
                ):
                    combined[0, 0, token_idx, topk_idx, :] = host_output[local_row, :]

        return self._combined_to_device(combined)

    def forward(
        self,
        x: ttnn.Tensor,
        return_intermediates: bool = False,
    ) -> tuple[ttnn.Tensor, Optional[TtMoEIntermediates]]:
        """
        Forward pass through the full MoE pipeline.

        Args:
            x: Input tensor - ROW_MAJOR, sharded:
               - For 2D mesh: sharded dims=(0, -1) - dim 0 across axis 0, dim -1 across axis 1
               - Shape per device: (dispatch_group_size/axis0, seq_len_per_chip, emb_dim/axis1)
            return_intermediates: If True, return intermediate tensors for debugging

        Returns:
            Tuple of (final_output, intermediates):
            - final_output: MoE output with same sharding as input
            - intermediates: TtMoEIntermediates if return_intermediates=True, else None
        """
        signpost(header="MoE_START")
        logger.debug(f"[TtMoe.forward] INPUT SHAPES:")
        logger.debug(f"  x.shape={x.shape}")

        # ========================================
        # Gate: compute weights/indices/offsets/counts from x
        # ========================================
        # Reshape 3D -> 2D for gate: (batch, seq, emb) -> (batch*seq, emb)

        scores, indices, gate_logits, tt_expert_offsets, tt_expert_token_counts, tt_expert_region_offsets = self.gate(
            ttnn.view(x, (x.shape[0] * x.shape[1], x.shape[2]))
        )
        gate_logits = (
            ttnn.to_memory_config(gate_logits, ttnn.DRAM_MEMORY_CONFIG)
            if return_intermediates
            else ttnn.deallocate(gate_logits)
        )  # gate_logits is only used for debugging/intermediates, move to DRAM or deallocate immediately

        if self.debug_token_count:
            # DEBUG: Print full token counts per expert for monitoring (controlled by env var)
            _counts_4d = ttnn.unsqueeze_to_4D(tt_expert_token_counts)
            _ep_composer = ttnn.create_mesh_composer(self.mesh_device, ttnn.MeshComposerConfig(dims=[1, 0]))
            _counts_host = ttnn.to_torch(_counts_4d, mesh_composer=_ep_composer).squeeze(2)
            logger.info(f"[TtMoe.forward] expert_token_counts: {_counts_host.flatten().tolist()}")

            # DEBUG: Print full region offsets per expert for monitoring
            _offsets_4d = ttnn.unsqueeze_to_4D(tt_expert_region_offsets)
            _offsets_host = ttnn.to_torch(_offsets_4d, mesh_composer=_ep_composer).squeeze(2)
            logger.info(f"[TtMoe.forward] expert_region_offsets: {_offsets_host.flatten().tolist()}")

        # Gate outputs uint16 indices; dispatch requires int32.
        # this should be aligned in the further PR.
        # Typecast in TILE_LAYOUT to avoid alignment issues, then convert to ROW_MAJOR.
        if indices.dtype != ttnn.int32:
            indices = ttnn.to_layout(indices, ttnn.TILE_LAYOUT)
            indices = ttnn.typecast(indices, ttnn.int32)
            indices = ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT)
        else:
            indices = ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT)
        #
        # Ensure ROW_MAJOR layout for dispatch compatibility
        scores = ttnn.to_layout(scores, ttnn.ROW_MAJOR_LAYOUT)

        # Reshape back to 3D: (batch*seq, topk) -> (batch, seq, topk)
        seq_dim = x.shape[1]
        batch_dim = x.shape[0]
        scores = ttnn.reshape(scores, (batch_dim, seq_dim, scores.shape[-1]))
        indices = ttnn.reshape(indices, (batch_dim, seq_dim, indices.shape[-1]))

        logger.debug(f"  {scores.shape=} {scores.memory_config()=}")
        logger.debug(f"  {indices.shape=} {indices.memory_config()=}")

        # ========================================
        # Step 0: All-gather x to get full emb_dim (replicated across TP axis)
        # ========================================
        # Input x is sharded: (dispatch_group_size/axis0, seq_len_per_chip, emb_dim/axis1)
        # Both shared_expert and dispatch need full emb_dim, so all-gather first
        # Only needed if there are multiple devices in TP axis (axis 1)
        owns_dispatch_input = self.deallocate_forward_input
        if self.mesh_device.shape[1] > 1:
            x = ttnn.all_gather(
                x,
                dim=-1,  # Gather along emb_dim
                cluster_axis=1,  # Gather across axis 1 (TP axis)
                num_links=self.col_num_links,
                topology=self.col_topology,
            )
            owns_dispatch_input = True
        logger.debug(f"[TtMoe.forward] x (after all_gather) shape: {x.shape}")

        signpost("shared_expert_and_dispatch_start")
        if self.overlap_shared_expert_with_dispatch:
            self.mesh_device.load_sub_device_manager(self.sd_manager_id)

        # ========================================
        # Step 1: Shared expert (enabled)
        # ========================================
        # Shared expert expects replicated input (full emb_dim)
        # Convert x to TILE_LAYOUT for shared expert
        logger.debug(f"[TtMoe.forward] {x.shape=} {x.memory_config()=}")

        if self.use_shared_expert:
            shared_output = self.shared_expert(x)
            logger.debug(f"[TtMoe.forward] Shared expert output shape: {shared_output.shape}")
        else:
            shared_output = None
            logger.debug("[TtMoe.forward] Shared expert disabled")

        # ========================================
        # Step 2: Dispatch (enabled)
        # ========================================
        # Dispatch expects full emb_dim on each device (x already has this)
        logger.debug(f"[TtMoe.forward] {x.shape=} {x.memory_config()=}")
        use_single_device_local_moe = self.mesh_device.get_num_devices() == 1
        local_dispatch_metadata = None
        if use_single_device_local_moe:
            dispatched_buffer, metadata, local_dispatch_metadata = self._single_device_local_dispatch(
                x, scores, indices, tt_expert_offsets, getattr(self.gate, "last_host_expert_offsets", None)
            )
        else:
            dispatched_buffer, metadata = self.dispatch_module(
                x,
                scores,
                indices,
                tt_expert_offsets,
                self.tt_expert_dispatch_table,
            )
        if self.overlap_shared_expert_with_dispatch:
            self.mesh_device.clear_loaded_sub_device_manager()
        if owns_dispatch_input:
            x = ttnn.deallocate(x)
        scores = ttnn.to_memory_config(scores, ttnn.DRAM_MEMORY_CONFIG)
        indices = ttnn.to_memory_config(indices, ttnn.DRAM_MEMORY_CONFIG)
        self.last_dispatch_buffer_token_size = int(dispatched_buffer.shape[-2])
        logger.debug(f"[TtMoe.forward] Dispatch output: buffer={dispatched_buffer.shape}, metadata={metadata.shape}")

        signpost("shared_expert_and_dispatch_end")

        # ========================================
        # Step 3: Routed experts (enabled)
        # ========================================
        # Dispatch output is (1, dispatch_group_size_per_device, experts_per_chip, max_tokens, emb_dim)
        # Routed expert expects (experts_per_chip, max_tokens, emb_dim)
        # Squeeze the first two dimensions

        # Convert dispatched_buffer to TILE_LAYOUT for routed experts. The DeepSeek
        # extract kernel requires a BFLOAT8_B TILE global tensor. For the opt-only
        # single-device non-BF8 path, however, tilizing the full sparse dispatch
        # capacity can be enormous (for example 2M x 2048 rows) and may crash in
        # TTNN. Keep the flat ROW_MAJOR buffer and tilize only per-expert slices in
        # TtRoutedExpert.forward.
        use_single_device_non_bf8_slice_path = (
            use_single_device_local_moe and self.routed_expert.activations_dtype != ttnn.bfloat8_b
        )
        dispatched_buffer_tiled = ttnn.squeeze(ttnn.squeeze(dispatched_buffer, dim=0), dim=0)
        if use_single_device_non_bf8_slice_path:
            logger.debug(
                "[TtMoe.forward] Skipping whole-dispatch to_layout for single-device non-BF8 routed path; "
                "per-expert slices will be tiled after slicing"
            )
        else:
            dispatched_buffer_tiled = ttnn.to_layout(dispatched_buffer_tiled, ttnn.TILE_LAYOUT)
            if dispatched_buffer_tiled.dtype != self.routed_expert.activations_dtype:
                dispatched_buffer_tiled = ttnn.typecast(dispatched_buffer_tiled, self.routed_expert.activations_dtype)

        # Free the original ROW_MAJOR DRAM buffer before entering routed_expert for clear state.
        # When return_intermediates=True, keep it so the PCC check can compare against the
        # bfloat16 torch reference (the tiled buffer may be bfloat8_b). If we kept a squeezed
        # view of the original ROW_MAJOR buffer for slice-on-demand, the original storage must
        # remain allocated until routed experts finish.
        if not return_intermediates and not use_single_device_non_bf8_slice_path:
            dispatched_buffer = ttnn.deallocate(dispatched_buffer)

        logger.debug(
            f"[TtMoe.forward] dispatched_buffer_tiled shape: {dispatched_buffer_tiled.shape} "
            f"dtype={dispatched_buffer_tiled.dtype}"
        )
        self.routed_expert.host_local_expert_counts = (
            getattr(self.gate, "last_host_expert_counts", None) if use_single_device_local_moe else None
        )
        self.routed_expert.host_local_expert_offsets = (
            getattr(self.gate, "last_host_expert_offsets", None) if use_single_device_local_moe else None
        )

        # NOTE: expert_outputs aliases dispatched_buffer_tiled — TtRoutedExpert.forward sets
        # expert_outputs = dispatched_buffer and then writes per-expert FFN results back
        # in-place via deepseek_prefill.insert. The two names point at the same device buffer.
        # Therefore we must NOT call ttnn.deallocate(dispatched_buffer_tiled) here; doing so
        # would free the storage that expert_outputs still depends on, and the subsequent
        # ttnn.unsqueeze / combine_module calls would raise "Tensor is not allocated".
        expert_outputs = self.routed_expert(dispatched_buffer_tiled, tt_expert_token_counts, tt_expert_region_offsets)
        if use_single_device_non_bf8_slice_path and not return_intermediates:
            dispatched_buffer = ttnn.deallocate(dispatched_buffer)
        expert_slice_outputs = expert_outputs if isinstance(expert_outputs, list) else None
        if expert_slice_outputs is None:
            logger.debug(f"[TtMoe.forward] expert_outputs shape: {expert_outputs.shape}")

            # Add back the batch dimensions for combine
            # (experts_per_chip, max_tokens, emb_dim) -> (1, 1, experts_per_chip, max_tokens, emb_dim)
            expert_outputs = ttnn.unsqueeze(expert_outputs, dim=0)
            expert_outputs = ttnn.unsqueeze(expert_outputs, dim=0)
            logger.debug(f"[TtMoe.forward] expert_outputs (unsqueezed) shape: {expert_outputs.shape}")
        else:
            logger.debug(f"[TtMoe.forward] expert slice outputs={len(expert_slice_outputs)}")

        # ========================================
        # Step 4: Combine (enabled)
        # ========================================
        # Combine expects TILE_LAYOUT input
        if expert_slice_outputs is None:
            logger.debug(f"[TtMoe.forward] expert_outputs shape: {expert_outputs.shape} {expert_outputs.dtype=}")

        if use_single_device_local_moe:
            if expert_slice_outputs is None:
                combined_output = self._single_device_local_combine(expert_outputs, local_dispatch_metadata)
            else:
                combined_output = self._single_device_local_combine_expert_slices(
                    expert_slice_outputs, local_dispatch_metadata
                )
        else:
            if expert_slice_outputs is not None:
                raise RuntimeError("Non-BF8 routed expert slice outputs are only supported for single-device opt runs")
            combined_output = self.combine_module(
                expert_outputs,
                metadata,
                tt_expert_token_counts,
                tt_expert_region_offsets,
            )
        logger.debug(f"[TtMoe.forward] combined_output shape: {combined_output.shape} {combined_output.dtype=}")

        # ========================================
        # Step 5: Reduce (fused weighted sum over topk + reduce-scatter for TP sharding)
        # ========================================
        # combined_output: (1, dispatch_group_size, seq_len_per_chip, num_experts_per_tok, emb_dim)
        #                  (1, 1, 256, 4, 2048) per device - 5D tensor, ROW_MAJOR
        #
        # TtReduceModule uses fused post_combine_reduce kernel:
        # 1. Fused weighted sum over topk (dim=3): reads ROW_MAJOR, outputs TILE_LAYOUT
        # 2. Reduce-scatter across TP axis: (1, 1, 256, 2048) -> (1, 1, 256, 512) per device
        routed_output = self.reduce_module(
            combined_output,
            weights=scores,
            indices=indices,
            expert_dispatch_table=self.tt_expert_dispatch_table,
        )
        logger.debug(f"[TtMoe.forward] routed_output (after reduce) shape: {routed_output.shape}")

        # Remove extra batch dimensions to match shared_output shape
        # (1, 1, 256, 512) -> (1, 256, 512)
        routed_output = ttnn.squeeze(routed_output, dim=0)
        logger.debug(f"[TtMoe.forward] routed_output (squeezed) shape: {routed_output.shape}")

        # ========================================
        # Step 6: Final output
        # ========================================
        # final_output = routed_output + shared_output
        # Both should be in TILE_LAYOUT with shape (dispatch_group_size, seq_len_per_chip, emb_dim)
        if shared_output is None:
            final_output = routed_output
        else:
            final_output = ttnn.add(routed_output, shared_output)
        logger.debug(f"[TtMoe.forward] final_output (tiled) shape: {final_output.shape}")

        # Build intermediates if requested
        intermediates = None
        if return_intermediates:
            # Check for buffer overflow (dispatch kernel silently drops overflow tokens).
            # The kernel bounds-check is against max_dispatch_buffer_token_size (total per-chip
            # buffer capacity). Group-sparse counts mean each chip's experts_per_chip-sized
            # chunk of _counts_host holds that chip's nonzero counts; the sum of each chunk is
            # the chip's total dispatched tokens and must fit in the dispatch buffer.
            _counts_4d = ttnn.unsqueeze_to_4D(tt_expert_token_counts)
            _ep_composer = ttnn.create_mesh_composer(self.mesh_device, ttnn.MeshComposerConfig(dims=[1, 0]))
            _counts_host = ttnn.to_torch(_counts_4d, mesh_composer=_ep_composer).squeeze(2)
            _per_chip_sums = _counts_host.to(torch.int64).flatten().view(-1, self.experts_per_chip).sum(dim=1)
            max_per_chip_sum = int(_per_chip_sums.max().item())
            max_capacity = self.dispatch_module.max_dispatch_buffer_token_size
            logger.info(
                f"[TtMoe.forward] max per-chip dispatched token sum: {max_per_chip_sum} "
                f"(max_dispatch_buffer_token_size={max_capacity})"
            )
            if max_per_chip_sum > max_capacity:
                logger.error(
                    f"[TtMoe.forward] per-chip dispatched token sum ({max_per_chip_sum}) exceeds "
                    f"max_dispatch_buffer_token_size ({max_capacity}). "
                    f"Overflow tokens were dropped - output data is corrupted. "
                    f"Reduce sequence length."
                )
                logger.debug(f"[TtMoe.forward] expert_token_counts: {_counts_host.flatten().tolist()}")
                logger.debug(f"[TtMoe.forward] per_chip_sums: {_per_chip_sums.tolist()}")

            # Every per-expert region offset must address a row inside the dispatch buffer
            # (i.e. < max_dispatch_buffer_token_size). An offset >= capacity means the
            # expert's region starts past the end of the buffer and its tokens are dropped.
            _offsets_4d = ttnn.unsqueeze_to_4D(tt_expert_region_offsets)
            _offsets_host = ttnn.to_torch(_offsets_4d, mesh_composer=_ep_composer).squeeze(2)
            _offsets_flat = _offsets_host.to(torch.int64).flatten()
            _argmax_offset = int(_offsets_flat.argmax().item())
            max_region_offset = int(_offsets_flat[_argmax_offset].item())
            max_offset_token_count = int(_counts_host.to(torch.int64).flatten()[_argmax_offset].item())
            logger.info(
                f"[TtMoe.forward] max expert region offset: {max_region_offset} "
                f"(token_count for that expert: {max_offset_token_count}, "
                f"max_dispatch_buffer_token_size={max_capacity})"
            )
            if max_region_offset >= max_capacity:
                logger.error(
                    f"[TtMoe.forward] expert region offset ({max_region_offset}) is not below "
                    f"max_dispatch_buffer_token_size ({max_capacity}). "
                    f"Overflow tokens were dropped - output data is corrupted. "
                    f"Reduce sequence length."
                )
                logger.debug(f"[TtMoe.forward] expert_region_offsets: {_offsets_host.flatten().tolist()}")

            intermediates = TtMoEIntermediates(
                gate_scores=scores,
                gate_indices=indices,
                gate_logits=gate_logits,
                dispatched_buffer=dispatched_buffer,
                metadata=metadata,
                expert_outputs=expert_outputs,
                shared_output=shared_output,
                combined_output=combined_output,
                routed_output=routed_output,
                expert_token_counts=tt_expert_token_counts,
            )

        signpost(header="MoE_END")
        return final_output, intermediates
