# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, List

import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.common import (
    Mode,
    PagedAttentionConfig,
    copy_host_to_device,
    create_tt_model,
    get_padded_prefill_len,
    preprocess_inputs_prefill,
    sample_host,
)
from models.tt_transformers.tt.generator import Generator, SamplingParams, create_submeshes
from models.tt_transformers.tt.model_config import DecodersPrecision, determine_device_name


def performance_optimizations(model_args):
    return DecodersPrecision.performance(model_args.n_layers, model_args.model_name)


# Edit these values directly for quick experiments. Keep this hardcoded so a
# stale shell HF_MODEL does not silently switch the model config.
HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PROMPTS = [
    "Who is Donald Trump?",
#Explain what Tenstorrent P100A is in simple terms.
]

# The pytest text demo uses instruct formatting for the normal question/answer
# paths. Set this to False only if HF_MODEL points at base, non-instruct weights.
INSTRUCT = True

# On a single P100A card, this is one physical device. Do not use the 11x10
# chip core grid as the mesh shape.
MESH_SHAPE = (1, 1)

BATCH_SIZE = 1
DATA_PARALLEL = 1
MAX_SEQ_LEN = 1024
MAX_GENERATED_TOKENS = 200
STOP_AT_EOS = True
ENABLE_TRACE = True
NUM_COMMAND_QUEUES = 2
USE_PREFETCHER = False
USE_HF_ROPE = False
NUM_LAYERS = None
FORCE_HOST_SAMPLING = True
RUN_EXTRA_PREFILL_WARMUP = False
DEBUG_PROMPT = True
RUN_TEXT_DEMO = False
RUN_SUBDEVICE_STAGE_BENCHMARK = True

SUBDEVICE_STAGE_NUM_SUBDEVICES = 2
SUBDEVICE_STAGE_WARMUP_ITERATIONS = 2
SUBDEVICE_STAGE_MEASURED_ITERATIONS = 10
SUBDEVICE_STAGE_DECODE_TOKENS = 1
SUBDEVICE_STAGE_PREFILL_QUEUE_ID = 0
SUBDEVICE_STAGE_DECODE_QUEUE_ID = 1
SUBDEVICE_STAGE_PREFILL_ROWS = 8

PAGED_ATTENTION = True
PAGE_PARAMS = {
    "page_block_size": 32,
    "page_max_num_blocks_per_dp": 1024,
}

SAMPLING_PARAMS = {
    "temperature": 0.0,
    "top_p": 0.08,
    "top_k": 32,
}

TRACE_REGION_SIZE = 50_000_000
#이거를 켤 경우 기존 cache 삭제하고 시작
CLEAR_WEIGHT_CACHE_ON_START = False


def create_tt_page_table(global_batch_size, data_parallel, paged_attention_config):
    if paged_attention_config is None:
        return None

    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    reverse_permutation = torch.argsort(permutation).repeat(data_parallel)
    return reverse_permutation.reshape(
        global_batch_size,
        paged_attention_config.max_num_blocks // (global_batch_size // data_parallel),
    )


def open_mesh_device():
    rows, cols = MESH_SHAPE
    return ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(rows, cols),
        trace_region_size=TRACE_REGION_SIZE,
        num_command_queues=NUM_COMMAND_QUEUES,
    )


def weight_cache_root(device_name):
    tt_cache_path = os.getenv("TT_CACHE_PATH")
    if tt_cache_path:
        cache_path = Path(tt_cache_path) / device_name
    else:
        cache_path = Path("model_cache") / HF_MODEL / device_name
    if USE_HF_ROPE:
        cache_path = cache_path / "hf_rope"
    return cache_path


def clear_weight_cache(device_name):
    if not CLEAR_WEIGHT_CACHE_ON_START:
        return

    cache_path = weight_cache_root(device_name)
    if cache_path.exists():
        logger.warning(f"Deleting existing TT weight cache: {cache_path}")
        shutil.rmtree(cache_path)
    else:
        logger.info(f"No existing TT weight cache to delete: {cache_path}")


def prepare_model(mesh_device):
    global_batch_size = BATCH_SIZE * DATA_PARALLEL
    submesh_devices = create_submeshes(mesh_device, DATA_PARALLEL)

    paged_attention_config = (
        PagedAttentionConfig(
            block_size=PAGE_PARAMS["page_block_size"],
            max_num_blocks=PAGE_PARAMS["page_max_num_blocks_per_dp"],
        )
        if PAGED_ATTENTION
        else None
    )

    models = []
    model_args = []
    tt_kv_cache = []
    state_dict = None
    max_batch_size_per_dp_group = global_batch_size // DATA_PARALLEL

    for submesh in submesh_devices:
        model_args_i, model_i, tt_kv_cache_i, state_dict = create_tt_model(
            submesh,
            instruct=INSTRUCT,
            max_batch_size=max_batch_size_per_dp_group,
            optimizations=performance_optimizations,
            max_seq_len=MAX_SEQ_LEN,
            paged_attention_config=paged_attention_config,
            dtype=ttnn.bfloat8_b,
            state_dict=state_dict,
            num_layers=NUM_LAYERS,
            use_prefetcher=USE_PREFETCHER,
            use_hf_rope=USE_HF_ROPE,
        )
        model_args.append(model_args_i)
        models.append(model_i)
        tt_kv_cache.append(tt_kv_cache_i)

    page_table = create_tt_page_table(global_batch_size, DATA_PARALLEL, paged_attention_config)
    tokenizer = model_args[0].tokenizer
    processor = model_args[0].processor
    generator = Generator(models, model_args, mesh_device, processor=processor, tokenizer=tokenizer)

    return generator, model_args, page_table, tt_kv_cache, tokenizer


def make_prompt_batch():
    global_batch_size = BATCH_SIZE * DATA_PARALLEL
    if len(PROMPTS) >= global_batch_size:
        return PROMPTS[:global_batch_size]

    logger.warning("PROMPTS has fewer entries than the global batch size; repeating prompts to fill the batch.")
    repeats = (global_batch_size + len(PROMPTS) - 1) // len(PROMPTS)
    return (PROMPTS * repeats)[:global_batch_size]


def decode_text(generator, model_args, page_table, tt_kv_cache, tokenizer, input_prompts):
    global_batch_size = BATCH_SIZE * DATA_PARALLEL
    (
        input_tokens_prefill_pt,
        encoded_prompts,
        decoding_pos,
        prefill_lens,
    ) = preprocess_inputs_prefill(
        input_prompts,
        tokenizer,
        model_args,
        INSTRUCT,
        MAX_GENERATED_TOKENS,
        max_prefill_len=MAX_SEQ_LEN,
    )

    max_encoded_prompt_len = max(len(prompt) for prompt in encoded_prompts)
    assert (
        MAX_GENERATED_TOKENS + max_encoded_prompt_len <= MAX_SEQ_LEN
    ), f"prompt tokens ({max_encoded_prompt_len}) + generated tokens ({MAX_GENERATED_TOKENS}) exceed MAX_SEQ_LEN ({MAX_SEQ_LEN})"

    if PAGED_ATTENTION:
        paged_cache_max_seq_len = PAGE_PARAMS["page_block_size"] * PAGE_PARAMS["page_max_num_blocks_per_dp"] / BATCH_SIZE
        assert (
            MAX_GENERATED_TOKENS + max_encoded_prompt_len <= paged_cache_max_seq_len
        ), f"prompt + generated tokens exceed paged cache limit ({paged_cache_max_seq_len})"

    input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(global_batch_size, -1)

    if DEBUG_PROMPT:
        for user, prompt in enumerate(input_prompts):
            decoded_prompt = tokenizer.decode(encoded_prompts[user])
            logger.info(f"[User {user}] Raw prompt: {prompt!r}")
            logger.info(f"[User {user}] Encoded prompt length: {len(encoded_prompts[user])}")
            logger.info(f"[User {user}] Encoded prompt ids head: {encoded_prompts[user][:16]}")
            logger.info(f"[User {user}] Decoded encoded prompt:\n{decoded_prompt}")

    device_sampling_params = (
        SamplingParams(
            temperature=SAMPLING_PARAMS["temperature"],
            top_k=SAMPLING_PARAMS["top_k"],
            top_p=SAMPLING_PARAMS["top_p"],
            seed=SAMPLING_PARAMS.get("seed"),
            frequency_penalty=SAMPLING_PARAMS.get("frequency_penalty", 0.0),
            presence_penalty=SAMPLING_PARAMS.get("presence_penalty", 0.0),
            repetition_penalty=SAMPLING_PARAMS.get("repetition_penalty", 1.0),
            enable_log_probs=SAMPLING_PARAMS.get("enable_log_probs", False),
        )
        if generator.model[0]._supports_on_device_sampling and not FORCE_HOST_SAMPLING
        else None
    )
    logger.info(
        f"Sampling path: {'device' if device_sampling_params is not None else 'host'} "
        f"(FORCE_HOST_SAMPLING={FORCE_HOST_SAMPLING})"
    )

    if RUN_EXTRA_PREFILL_WARMUP:
        logger.info("Running extra prefill compile/warmup pass...")
        generator.prefill_forward_text(
            input_tokens_prefill_pt,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            prompt_lens=decoding_pos,
            sampling_params=device_sampling_params,
            warmup_prefill=True,
            enable_trace=ENABLE_TRACE,
        )

    logger.info("Running prefill...")
    prefill_start = time.perf_counter()
    prefill_out = generator.prefill_forward_text(
        input_tokens_prefill_pt,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=decoding_pos,
        sampling_params=device_sampling_params,
        warmup_prefill=True,
        enable_trace=ENABLE_TRACE,
    )
    prefill_time_s = time.perf_counter() - prefill_start

    if device_sampling_params is not None and isinstance(prefill_out, tuple):
        prefilled_token, _ = prefill_out
    else:
        prefilled_token = torch.argmax(prefill_out, dim=-1)

    if DEBUG_PROMPT:
        for user in range(global_batch_size):
            token = int(prefilled_token[user].item())
            logger.info(f"[User {user}] First token from prefill: id={token}, text={tokenizer.decode([token])!r}")

    all_outputs = [encoded_prompts[b][: decoding_pos[b]] for b in range(global_batch_size)]
    for user in range(global_batch_size):
        all_outputs[user].append(int(prefilled_token[user].item()))

    out_tok = prefilled_token
    current_pos = torch.tensor(decoding_pos)
    user_done = [False] * global_batch_size
    decode_times_s: List[float] = []

    logger.info("Starting decode loop...")
    for iteration in range(MAX_GENERATED_TOKENS):
        iter_start = time.perf_counter()
        logits, _ = generator.decode_forward(
            out_tok,
            current_pos,
            enable_trace=ENABLE_TRACE,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            reset_batch=(iteration == 0),
            sampling_params=device_sampling_params,
            prompt_tokens=input_tokens_prefill_pt,
            output_tokens=out_tok,
        )

        if device_sampling_params is not None:
            out_tok = logits.unsqueeze(1)
        else:
            _, out_tok = sample_host(
                logits,
                temperature=SAMPLING_PARAMS["temperature"],
                top_p=SAMPLING_PARAMS["top_p"],
                on_host=True,
            )

        iter_time_s = time.perf_counter() - iter_start
        decode_times_s.append(iter_time_s)

        current_pos += 1
        for user in range(global_batch_size):
            user_tok = int(out_tok[user].item())
            if user_tok not in tokenizer.stop_tokens and not user_done[user]:
                all_outputs[user].append(user_tok)
            elif STOP_AT_EOS:
                user_done[user] = True
            else:
                all_outputs[user].append(user_tok)

        if STOP_AT_EOS and all(user_done):
            logger.info(f"All users reached EOS at decode iteration {iteration}.")
            break

    outputs = []
    for output_tokens, prompt in zip(all_outputs, input_prompts):
        full_text = tokenizer.decode(output_tokens)
        prompt_with_tags = tokenizer.decode(model_args[0].encode_prompt(prompt, instruct=INSTRUCT))
        outputs.append(full_text.replace(prompt_with_tags, "", 1).strip())

    return outputs, prefill_time_s, decode_times_s


def _core_range(x0: int, y0: int, x1: int, y1: int) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))})


def _core_range_set_for_subdevice_cores(subdevice_row_range, num_cols: int, num_cores: int) -> ttnn.CoreRangeSet:
    y0, y1 = subdevice_row_range
    capacity = num_cols * (y1 - y0 + 1)
    if num_cores > capacity:
        raise ValueError(f"Requested {num_cores} cores from subdevice rows {y0}-{y1}, but capacity is {capacity}")

    remaining = num_cores
    core_ranges = set()
    for y in range(y0, y1 + 1):
        if remaining == 0:
            break
        cores_in_row = min(num_cols, remaining)
        core_ranges.add(ttnn.CoreRange(ttnn.CoreCoord(0, y), ttnn.CoreCoord(cores_in_row - 1, y)))
        remaining -= cores_in_row

    return ttnn.CoreRangeSet(core_ranges)


def _split_stage_ranges(cols: int, rows: int, min_decode_worker_cores: int = 1) -> list[tuple[int, int, ttnn.CoreRangeSet]]:
    if SUBDEVICE_STAGE_NUM_SUBDEVICES != 2:
        raise ValueError("This benchmark expects exactly two subdevices: prefill and decode.")
    min_decode_rows = max(1, (min_decode_worker_cores + cols - 1) // cols)
    if min_decode_rows >= rows:
        raise ValueError(
            f"Decode subdevice needs at least {min_decode_worker_cores} worker cores, but grid={cols}x{rows} "
            f"can leave at most {cols * (rows - 1)} cores while keeping a prefill subdevice."
        )

    prefill_rows = min(SUBDEVICE_STAGE_PREFILL_ROWS, rows - min_decode_rows)
    if not 0 < prefill_rows < rows:
        raise ValueError(
            f"SUBDEVICE_STAGE_PREFILL_ROWS={SUBDEVICE_STAGE_PREFILL_ROWS} must leave at least "
            f"{min_decode_rows} decode rows for device grid={cols}x{rows}"
        )

    split_rows = [(0, prefill_rows - 1), (prefill_rows, rows - 1)]
    return [(y0, y1, _core_range(0, y0, cols - 1, y1)) for y0, y1 in split_rows]


@contextmanager
def stage_subdevice_manager(mesh_device, min_decode_worker_cores: int = 1):
    grid = mesh_device.compute_with_storage_grid_size()
    cols, rows = int(grid.x), int(grid.y)
    sub_ranges = _split_stage_ranges(cols, rows, min_decode_worker_cores)
    sub_devices = [ttnn.SubDevice([core_range]) for _, _, core_range in sub_ranges]
    manager = mesh_device.create_sub_device_manager(sub_devices, 0)
    mesh_device.load_sub_device_manager(manager)

    sub_device_ids = [ttnn.SubDeviceId(index) for index in range(SUBDEVICE_STAGE_NUM_SUBDEVICES)]
    sub_device_grid_sizes = [(cols, y1 - y0 + 1) for y0, y1, _ in sub_ranges]
    sub_device_row_ranges = [(y0, y1) for y0, y1, _ in sub_ranges]
    split_desc = ", ".join(f"id={idx}: grid={cols}x{y1 - y0 + 1} rows={y0}-{y1}" for idx, (y0, y1, _) in enumerate(sub_ranges))
    try:
        yield sub_device_ids, sub_device_grid_sizes, sub_device_row_ranges, split_desc
    finally:
        mesh_device.reset_sub_device_stall_group()
        mesh_device.clear_loaded_sub_device_manager()
        mesh_device.remove_sub_device_manager(manager)


def _required_decode_worker_cores(generator, model_args) -> int:
    prefetcher = getattr(generator.model[0], "prefetcher", None)
    mem_config = model_args[0].get_residual_mem_config(Mode.DECODE, prefetcher)
    shard_spec = mem_config.shard_spec
    if callable(shard_spec):
        shard_spec = shard_spec()
    if shard_spec is None:
        return 1
    return int(shard_spec.grid.num_cores())


def _fit_core_range_to_subdevice(subdevice_row_range, num_cols: int, num_rows: int) -> ttnn.CoreRangeSet:
    y0, _ = subdevice_row_range
    return _core_range(0, y0, num_cols - 1, y0 + num_rows - 1)


def _fit_memory_config_to_subdevice(memory_config, subdevice_grid_size, subdevice_row_range):
    if memory_config is None or not hasattr(memory_config, "is_sharded") or not memory_config.is_sharded():
        return memory_config

    shard_spec = memory_config.shard_spec
    if callable(shard_spec):
        shard_spec = shard_spec()
    if shard_spec is None:
        return memory_config

    num_cores = int(shard_spec.num_cores())
    new_grid = _core_range_set_for_subdevice_cores(subdevice_row_range, subdevice_grid_size[0], num_cores)
    if new_grid == shard_spec.grid:
        return memory_config

    return ttnn.MemoryConfig(
        memory_config.memory_layout,
        memory_config.buffer_type,
        ttnn.ShardSpec(new_grid, shard_spec.shape, shard_spec.orientation),
    )


def _fit_memory_configs_in_value_to_subdevice(value, subdevice_grid_size, subdevice_row_range):
    if hasattr(value, "is_sharded"):
        return _fit_memory_config_to_subdevice(value, subdevice_grid_size, subdevice_row_range)
    if isinstance(value, dict):
        return {
            key: _fit_memory_configs_in_value_to_subdevice(item, subdevice_grid_size, subdevice_row_range)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_fit_memory_configs_in_value_to_subdevice(item, subdevice_grid_size, subdevice_row_range) for item in value]
    if isinstance(value, tuple):
        return tuple(_fit_memory_configs_in_value_to_subdevice(item, subdevice_grid_size, subdevice_row_range) for item in value)
    return value


def _fit_sdpa_program_config_to_subdevice(program_config, subdevice_grid_size, subdevice_row_range=None):
    if program_config is None:
        if subdevice_row_range is None:
            return None
        sub_cols, sub_rows = subdevice_grid_size
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(sub_cols, sub_rows),
            sub_core_grids=_fit_core_range_to_subdevice(subdevice_row_range, sub_cols, sub_rows),
            q_chunk_size=0,
            k_chunk_size=0,
            exp_approx_mode=False,
        )

    if isinstance(program_config.compute_with_storage_grid_size, tuple):
        grid_x, grid_y = program_config.compute_with_storage_grid_size
    else:
        grid_x, grid_y = program_config.compute_with_storage_grid_size.x, program_config.compute_with_storage_grid_size.y

    sub_cols, sub_rows = subdevice_grid_size
    fit_cols = min(int(grid_x), sub_cols)
    fit_rows = min(int(grid_y), sub_rows)
    sub_core_grids = program_config.sub_core_grids
    if subdevice_row_range is not None and sub_core_grids is None:
        sub_core_grids = _fit_core_range_to_subdevice(subdevice_row_range, fit_cols, fit_rows)

    if fit_cols == int(grid_x) and fit_rows == int(grid_y) and sub_core_grids == program_config.sub_core_grids:
        return program_config

    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(fit_cols, fit_rows),
        sub_core_grids=sub_core_grids,
        q_chunk_size=program_config.q_chunk_size,
        k_chunk_size=program_config.k_chunk_size,
        exp_approx_mode=program_config.exp_approx_mode,
        max_cores_per_head_batch=program_config.max_cores_per_head_batch,
    )


def _fit_matmul_program_config_to_subdevice(program_config, subdevice_grid_size):
    if program_config is None:
        return None

    if not hasattr(program_config, "compute_with_storage_grid_size"):
        return program_config

    sub_cols, sub_rows = subdevice_grid_size
    grid = program_config.compute_with_storage_grid_size
    fit_cols = min(int(grid.x), sub_cols)
    fit_rows = min(int(grid.y), sub_rows)
    allowed_worker_cores = _core_range(0, 0, fit_cols - 1, fit_rows - 1)

    if isinstance(program_config, ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig):
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(fit_cols, fit_rows),
            in0_block_w=program_config.in0_block_w,
            out_subblock_h=program_config.out_subblock_h,
            out_subblock_w=program_config.out_subblock_w,
            out_block_h=program_config.out_block_h,
            out_block_w=program_config.out_block_w,
            per_core_M=program_config.per_core_M,
            per_core_N=program_config.per_core_N,
            fuse_batch=program_config.fuse_batch,
            fused_activation=program_config.fused_activation,
            mcast_in0=program_config.mcast_in0,
            gather_in0=program_config.gather_in0,
            hop_cores=program_config.hop_cores,
            num_global_cb_receivers=program_config.num_global_cb_receivers,
            untilize_out=program_config.untilize_out,
            allowed_worker_cores=allowed_worker_cores,
        )

    if isinstance(program_config, ttnn.MatmulMultiCoreReuseMultiCastProgramConfig):
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(fit_cols, fit_rows),
            in0_block_w=program_config.in0_block_w,
            out_subblock_h=program_config.out_subblock_h,
            out_subblock_w=program_config.out_subblock_w,
            out_block_h=program_config.out_block_h,
            out_block_w=program_config.out_block_w,
            per_core_M=program_config.per_core_M,
            per_core_N=program_config.per_core_N,
            transpose_mcast=program_config.transpose_mcast,
            fused_activation=program_config.fused_activation,
            fuse_batch=program_config.fuse_batch,
            allowed_worker_cores=allowed_worker_cores,
        )

    if isinstance(program_config, ttnn.MatmulMultiCoreReuseProgramConfig):
        return ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(fit_cols, fit_rows),
            in0_block_w=program_config.in0_block_w,
            out_subblock_h=program_config.out_subblock_h,
            out_subblock_w=program_config.out_subblock_w,
            per_core_M=program_config.per_core_M,
            per_core_N=program_config.per_core_N,
            allowed_worker_cores=allowed_worker_cores,
        )

    return program_config


@contextmanager
def stage_subdevice_scope(
    mesh_device,
    sub_device_id: ttnn.SubDeviceId,
    queue_id: int,
    subdevice_grid_size,
    subdevice_row_range,
):
    subdevice_ops = {
        "ttnn.add",
        "ttnn.concat",
        "ttnn.embedding",
        "ttnn.interleaved_to_sharded",
        "ttnn.layer_norm",
        "ttnn.linear",
        "ttnn.matmul",
        "ttnn.matmul_batched_weights",
        "ttnn.mul",
        "ttnn.multiply",
        "ttnn.pad",
        "ttnn.rms_norm",
        "ttnn.reshape",
        "ttnn.slice",
        "ttnn.to_memory_config",
        "ttnn.transpose",
        "ttnn.typecast",
        "ttnn.untilize",
        "ttnn.experimental.rotary_embedding_llama",
        "ttnn.transformer.scaled_dot_product_attention",
    }
    sdpa_decode_ops = {
        "ttnn.transformer.paged_scaled_dot_product_attention_decode",
        "ttnn.transformer.scaled_dot_product_attention_decode",
    }
    sub_core_grid_ops = {
        "ttnn.experimental.nlp_concat_heads_decode",
    }
    subdevice_core_range = _fit_core_range_to_subdevice(
        subdevice_row_range, subdevice_grid_size[0], subdevice_grid_size[1]
    )

    def inject_subdevice_id(operation, _args, kwargs):
        op_name = getattr(operation, "python_fully_qualified_name", "")
        for key, value in list(kwargs.items()):
            kwargs[key] = _fit_memory_configs_in_value_to_subdevice(value, subdevice_grid_size, subdevice_row_range)

        has_explicit_core_scope = kwargs.get("sub_core_grids") is not None
        if op_name in subdevice_ops and kwargs.get("sub_device_id") is None and not has_explicit_core_scope:
            kwargs["sub_device_id"] = sub_device_id
        if op_name in {"ttnn.linear", "ttnn.matmul", "ttnn.matmul_batched_weights"}:
            kwargs["program_config"] = _fit_matmul_program_config_to_subdevice(
                kwargs.get("program_config"), subdevice_grid_size
            )
        if op_name == "ttnn.transformer.scaled_dot_product_attention":
            kwargs["program_config"] = _fit_sdpa_program_config_to_subdevice(
                kwargs.get("program_config"), subdevice_grid_size
            )
        if op_name in sdpa_decode_ops:
            kwargs["program_config"] = _fit_sdpa_program_config_to_subdevice(
                kwargs.get("program_config"), subdevice_grid_size, subdevice_row_range
            )
        if op_name in sub_core_grid_ops and kwargs.get("sub_core_grids") is None:
            kwargs["sub_core_grids"] = subdevice_core_range

    mesh_device.set_sub_device_stall_group([sub_device_id])
    with ttnn.command_queue(queue_id), ttnn.register_pre_operation_hook(inject_subdevice_id):
        yield


def _flatten_ttnn_tensors(value) -> Iterable[ttnn.Tensor]:
    if isinstance(value, ttnn.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_ttnn_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _flatten_ttnn_tensors(item)


def _deallocate_ttnn_outputs(value) -> None:
    for tensor in _flatten_ttnn_tensors(value):
        try:
            ttnn.deallocate(tensor)
        except Exception:
            try:
                tensor.deallocate()
            except Exception:
                pass


def _prepare_stage_benchmark_inputs(model_args, page_table, tt_kv_cache, tokenizer, input_prompts):
    global_batch_size = BATCH_SIZE * DATA_PARALLEL
    if global_batch_size != 1 or DATA_PARALLEL != 1:
        raise ValueError("The subdevice stage benchmark currently expects BATCH_SIZE=1 and DATA_PARALLEL=1.")
    if not PAGED_ATTENTION or page_table is None or tt_kv_cache is None:
        raise ValueError("The subdevice stage benchmark requires PAGED_ATTENTION=True.")

    input_tokens_prefill_pt, encoded_prompts, decoding_pos, _ = preprocess_inputs_prefill(
        input_prompts,
        tokenizer,
        model_args,
        INSTRUCT,
        MAX_GENERATED_TOKENS,
        max_prefill_len=MAX_SEQ_LEN,
    )
    input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(global_batch_size, -1)

    prompt_len = int(decoding_pos[0])
    prefill_seq_len = get_padded_prefill_len(prompt_len)
    prefill_ids = torch.cat(
        [
            input_tokens_prefill_pt[0:1, :prompt_len],
            torch.zeros(1, prefill_seq_len - prompt_len, dtype=torch.long),
        ],
        dim=-1,
    )

    decode_page_shift = max(1, page_table.shape[1] // 2)
    decode_page_table = torch.roll(page_table.clone(), shifts=decode_page_shift, dims=1)
    decode_in_tok = input_tokens_prefill_pt[:, prompt_len - 1 : prompt_len]
    current_pos = torch.tensor(decoding_pos)

    return {
        "encoded_prompts": encoded_prompts,
        "prompt_len": prompt_len,
        "prefill_ids": prefill_ids,
        "prefill_page_table": page_table,
        "decode_page_table": decode_page_table,
        "decode_in_tok": decode_in_tok,
        "current_pos": current_pos,
    }


def _set_generator_mode(generator: Generator, mode: Mode) -> None:
    generator.mode = mode
    for model in generator.model:
        model.switch_mode(mode)


def _run_prefill_stage_no_read(generator: Generator, page_table, tt_kv_cache, bench_inputs):
    model_id = 0
    prompt_len = bench_inputs["prompt_len"]
    prefill_seq_len = bench_inputs["prefill_ids"].shape[-1]
    _set_generator_mode(generator, Mode.PREFILL)
    page_table_user = generator._get_prefill_user_page_table(
        page_table[0:1],
        tt_kv_cache[model_id],
        prompt_len,
        trace_enabled=False,
        prefill_seq_len=prefill_seq_len,
        use_batched_prefill=False,
        user_id=0,
        padded_batch_size=None,
    )
    return generator.prefill_forward_single_user_text(
        bench_inputs["prefill_ids"],
        page_table=page_table_user,
        user_id=0,
        last_token_idx=prompt_len - 1,
        kv_cache=tt_kv_cache[model_id],
        model_id=model_id,
        batch_size=1,
    )


def _run_decode_stage_no_read(generator: Generator, page_table, tt_kv_cache, bench_inputs):
    outputs = None
    for iteration in range(SUBDEVICE_STAGE_DECODE_TOKENS):
        current_pos = bench_inputs["current_pos"] + iteration
        outputs = generator.decode_forward(
            bench_inputs["decode_in_tok"],
            current_pos,
            enable_trace=False,
            read_from_device=False,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            reset_batch=(iteration == 0),
            sampling_params=None,
        )
    return outputs


def _prepare_prefill_stage_device_inputs(generator: Generator, page_table, tt_kv_cache, bench_inputs):
    model_id = 0
    prompt_len = bench_inputs["prompt_len"]
    prefill_seq_len = bench_inputs["prefill_ids"].shape[-1]
    _set_generator_mode(generator, Mode.PREFILL)
    page_table_user = generator._get_prefill_user_page_table(
        page_table[0:1],
        tt_kv_cache[model_id],
        prompt_len,
        trace_enabled=False,
        prefill_seq_len=prefill_seq_len,
        use_batched_prefill=False,
        user_id=0,
        padded_batch_size=None,
    )
    host_inputs = generator.model[model_id].prepare_prefill_inputs_trace(
        bench_inputs["prefill_ids"],
        page_table=page_table_user,
        user_id=0,
    )
    device_inputs = copy_host_to_device(
        (host_inputs[0], host_inputs[3], host_inputs[4], host_inputs[5]),
        mesh_device=generator.model_args[model_id].mesh_device,
    )
    return {
        "model_id": model_id,
        "prompt_len": prompt_len,
        "device_inputs": device_inputs,
        "rot_mats_global": host_inputs[1],
        "rot_mats_local": host_inputs[2],
    }


def _run_prefill_stage_prepared(generator: Generator, tt_kv_cache, prepared_inputs):
    model_id = prepared_inputs["model_id"]
    _set_generator_mode(generator, Mode.PREFILL)
    transformed_inputs = generator.model[model_id].transform_and_embed_prefill_inputs_device(
        *prepared_inputs["device_inputs"]
    )
    return generator.model[model_id].ttnn_prefill_forward(
        x=transformed_inputs[0],
        rot_mats_global=prepared_inputs["rot_mats_global"],
        rot_mats_local=prepared_inputs["rot_mats_local"],
        page_table=transformed_inputs[1],
        chunk_page_table=transformed_inputs[2],
        # The benchmark prefill starts at 0, so the full RoPE mats are already the right slice.
        # Skipping the traced dynamic slice avoids a tiny unaligned rank-1 concat.
        chunk_start_idx=None,
        get_last_token=(prepared_inputs["prompt_len"] - 1) // 32 * 32,
        kv_cache=tt_kv_cache[model_id],
    )


def _prepare_decode_stage_device_inputs(generator: Generator, page_table, bench_inputs):
    prepared_inputs = []
    for iteration in range(SUBDEVICE_STAGE_DECODE_TOKENS):
        current_pos = bench_inputs["current_pos"] + iteration
        tokens = torch.chunk(bench_inputs["decode_in_tok"], generator.data_parallel, 0)
        current_pos_chunks = torch.chunk(current_pos, generator.data_parallel, 0)
        page_table_chunks = torch.chunk(page_table, generator.data_parallel, 0) if page_table is not None else None
        iteration_inputs = []
        _set_generator_mode(generator, Mode.DECODE)
        for model_id in range(generator.data_parallel):
            user_page_table = page_table_chunks[model_id] if page_table_chunks is not None else None
            iteration_inputs.append(
                generator.model[model_id].prepare_inputs_decode(
                    tokens[model_id],
                    current_pos_chunks[model_id],
                    user_page_table,
                )
            )
        prepared_inputs.append(iteration_inputs)
    return prepared_inputs


def _run_decode_stage_prepared(generator: Generator, tt_kv_cache, prepared_inputs):
    outputs = None
    _set_generator_mode(generator, Mode.DECODE)
    for iteration_inputs in prepared_inputs:
        outputs = []
        for model_id, device_inputs in enumerate(iteration_inputs):
            user_kv_cache = tt_kv_cache[model_id] if tt_kv_cache is not None else None
            tt_logits_i, tt_log_probs_i = generator.model[model_id].ttnn_decode_forward(
                device_inputs[0],
                device_inputs[1],
                rot_mat_idxs=device_inputs[2],
                page_table=device_inputs[3],
                kv_cache=user_kv_cache,
                sampling_on_device=False,
            )
            outputs.append((tt_logits_i, tt_log_probs_i))
    return outputs


def _time_full_grid_sequential_once(mesh_device, generator, tt_kv_cache, bench_inputs):
    ttnn.synchronize_device(mesh_device)
    start_s = time.perf_counter()
    prefill_out = _run_prefill_stage_no_read(
        generator,
        bench_inputs["prefill_page_table"],
        tt_kv_cache,
        bench_inputs,
    )
    decode_out = _run_decode_stage_no_read(
        generator,
        bench_inputs["decode_page_table"],
        tt_kv_cache,
        bench_inputs,
    )
    ttnn.synchronize_device(mesh_device)
    elapsed_s = time.perf_counter() - start_s
    _deallocate_ttnn_outputs((prefill_out, decode_out))
    return elapsed_s


def _time_subdevice_parallel_once(
    mesh_device,
    generator,
    tt_kv_cache,
    bench_inputs,
    sub_device_ids,
    sub_device_grid_sizes,
    sub_device_row_ranges,
):
    ttnn.synchronize_device(mesh_device)
    prefill_inputs = _prepare_prefill_stage_device_inputs(
        generator,
        bench_inputs["prefill_page_table"],
        tt_kv_cache,
        bench_inputs,
    )
    decode_inputs = _prepare_decode_stage_device_inputs(
        generator,
        bench_inputs["decode_page_table"],
        bench_inputs,
    )
    ttnn.synchronize_device(mesh_device)
    start_s = time.perf_counter()
    with stage_subdevice_scope(
        mesh_device,
        sub_device_ids[0],
        SUBDEVICE_STAGE_PREFILL_QUEUE_ID,
        sub_device_grid_sizes[0],
        sub_device_row_ranges[0],
    ):
        prefill_out = _run_prefill_stage_prepared(generator, tt_kv_cache, prefill_inputs)
    with stage_subdevice_scope(
        mesh_device,
        sub_device_ids[1],
        SUBDEVICE_STAGE_DECODE_QUEUE_ID,
        sub_device_grid_sizes[1],
        sub_device_row_ranges[1],
    ):
        decode_out = _run_decode_stage_prepared(
            generator,
            tt_kv_cache,
            decode_inputs,
        )
    mesh_device.reset_sub_device_stall_group()
    ttnn.synchronize_device(mesh_device)
    elapsed_s = time.perf_counter() - start_s
    _deallocate_ttnn_outputs((prefill_inputs["device_inputs"], decode_inputs, prefill_out, decode_out))
    return elapsed_s


def _avg_ms(values_s: list[float]) -> float:
    return statistics.mean(values_s) * 1000.0


def _print_stage_benchmark_summary(full_seq_s, sub_parallel_s, split_desc, bench_inputs):
    full_avg_s = statistics.mean(full_seq_s)
    sub_avg_s = statistics.mean(sub_parallel_s)
    logger.info("")
    logger.info("=" * 96)
    logger.info("SUBDEVICE STAGE BENCHMARK RESULT")
    logger.info("-" * 96)
    logger.info(f"prompt_len={bench_inputs['prompt_len']} | decode_tokens={SUBDEVICE_STAGE_DECODE_TOKENS}")
    logger.info(f"split={split_desc}")
    logger.info(
        f"queues: prefill={SUBDEVICE_STAGE_PREFILL_QUEUE_ID}, decode={SUBDEVICE_STAGE_DECODE_QUEUE_ID} | "
        f"warmup={SUBDEVICE_STAGE_WARMUP_ITERATIONS} | iters={SUBDEVICE_STAGE_MEASURED_ITERATIONS}"
    )
    logger.info(
        f"full_grid_sequential: avg_ms={_avg_ms(full_seq_s):.3f} | "
        f"min_ms={min(full_seq_s) * 1000.0:.3f} | max_ms={max(full_seq_s) * 1000.0:.3f}"
    )
    logger.info(
        f"subdevice_parallel:  avg_ms={_avg_ms(sub_parallel_s):.3f} | "
        f"min_ms={min(sub_parallel_s) * 1000.0:.3f} | max_ms={max(sub_parallel_s) * 1000.0:.3f}"
    )
    logger.info(
        f"speedup(full_seq/subdevice_parallel)={full_avg_s / max(sub_avg_s, 1e-12):.3f} | "
        f"parallel_time/full_seq_time={sub_avg_s / max(full_avg_s, 1e-12):.3f}"
    )
    logger.info("=" * 96)


def run_subdevice_stage_benchmark(mesh_device, generator, model_args, page_table, tt_kv_cache, tokenizer, input_prompts):
    if SUBDEVICE_STAGE_NUM_SUBDEVICES != 2:
        raise ValueError("This benchmark compares exactly two stages, so SUBDEVICE_STAGE_NUM_SUBDEVICES must be 2.")
    if NUM_COMMAND_QUEUES <= max(SUBDEVICE_STAGE_PREFILL_QUEUE_ID, SUBDEVICE_STAGE_DECODE_QUEUE_ID):
        raise ValueError("NUM_COMMAND_QUEUES must cover the configured prefill/decode queue ids.")

    min_decode_worker_cores = _required_decode_worker_cores(generator, model_args)
    bench_inputs = _prepare_stage_benchmark_inputs(model_args, page_table, tt_kv_cache, tokenizer, input_prompts)

    logger.info("Preparing independent decode KV cache before timing...")
    decode_kv_out = _run_prefill_stage_no_read(
        generator,
        bench_inputs["decode_page_table"],
        tt_kv_cache,
        bench_inputs,
    )
    ttnn.synchronize_device(mesh_device)
    _deallocate_ttnn_outputs(decode_kv_out)

    logger.info("Warming up full-grid sequential prefill+decode...")
    for _ in range(SUBDEVICE_STAGE_WARMUP_ITERATIONS):
        _time_full_grid_sequential_once(mesh_device, generator, tt_kv_cache, bench_inputs)

    full_seq_s = []
    for iteration in range(SUBDEVICE_STAGE_MEASURED_ITERATIONS):
        elapsed = _time_full_grid_sequential_once(mesh_device, generator, tt_kv_cache, bench_inputs)
        full_seq_s.append(elapsed)
        logger.info(f"full_grid_sequential[{iteration}]: {elapsed * 1000.0:.3f} ms")

    sub_parallel_s = []
    with stage_subdevice_manager(mesh_device, min_decode_worker_cores) as (
        sub_device_ids,
        sub_device_grid_sizes,
        sub_device_row_ranges,
        split_desc,
    ):
        logger.info(f"Loaded stage subdevice manager: {split_desc}")
        logger.info("Warming up subdevice-parallel prefill+decode...")
        for _ in range(SUBDEVICE_STAGE_WARMUP_ITERATIONS):
            _time_subdevice_parallel_once(
                mesh_device,
                generator,
                tt_kv_cache,
                bench_inputs,
                sub_device_ids,
                sub_device_grid_sizes,
                sub_device_row_ranges,
            )

        for iteration in range(SUBDEVICE_STAGE_MEASURED_ITERATIONS):
            elapsed = _time_subdevice_parallel_once(
                mesh_device,
                generator,
                tt_kv_cache,
                bench_inputs,
                sub_device_ids,
                sub_device_grid_sizes,
                sub_device_row_ranges,
            )
            sub_parallel_s.append(elapsed)
            logger.info(f"subdevice_parallel[{iteration}]: {elapsed * 1000.0:.3f} ms")

    _print_stage_benchmark_summary(full_seq_s, sub_parallel_s, split_desc, bench_inputs)


def print_perf(prefill_time_s, decode_times_s):
    global_batch_size = BATCH_SIZE * DATA_PARALLEL
    generated_decode_tokens = len(decode_times_s)
    steady_decode_times = decode_times_s[1:] if len(decode_times_s) > 1 else decode_times_s
    avg_decode_s = sum(steady_decode_times) / len(steady_decode_times) if steady_decode_times else 0.0
    tok_s_user = 1.0 / avg_decode_s if avg_decode_s else 0.0
    tok_s = tok_s_user * global_batch_size

    logger.info(f"Prefill latency: {prefill_time_s * 1000:.2f} ms")
    if decode_times_s:
        logger.info(f"First decode iteration latency: {decode_times_s[0] * 1000:.2f} ms")
    logger.info(f"Generated decode tokens/user: {generated_decode_tokens}")
    logger.info(f"Average steady decode latency: {avg_decode_s * 1000:.2f} ms/token")
    logger.info(f"Decode throughput: {tok_s_user:.2f} tok/s/user, {tok_s:.2f} tok/s total")


def main():
    os.environ["HF_MODEL"] = HF_MODEL
    os.environ.pop("TT_CACHE_PATH", None)
    logger.info(f"HF_MODEL={HF_MODEL}")

    mesh_device = None
    try:
        mesh_device = open_mesh_device()
        device_name = determine_device_name(mesh_device)
        logger.info(f"Opened mesh with {mesh_device.get_num_devices()} device(s); detected {device_name}")
        clear_weight_cache(device_name)

        generator, model_args, page_table, tt_kv_cache, tokenizer = prepare_model(mesh_device)
        input_prompts = make_prompt_batch()

        if RUN_TEXT_DEMO:
            outputs, prefill_time_s, decode_times_s = decode_text(
                generator, model_args, page_table, tt_kv_cache, tokenizer, input_prompts
            )

            for idx, (prompt, output) in enumerate(zip(input_prompts, outputs)):
                print(f"\n== USER {idx} PROMPT ==\n{prompt}")
                print(f"\n== USER {idx} OUTPUT ==\n{output}\n")

            print_perf(prefill_time_s, decode_times_s)

        if RUN_SUBDEVICE_STAGE_BENCHMARK:
            run_subdevice_stage_benchmark(
                mesh_device,
                generator,
                model_args,
                page_table,
                tt_kv_cache,
                tokenizer,
                input_prompts,
            )
    finally:
        if mesh_device is not None:
            ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
