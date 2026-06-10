# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import time
from pathlib import Path

import torch
from loguru import logger
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

import ttnn
from models.common.sampling import SamplingParams
from models.common.utility_functions import is_blackhole
from models.demos.qwen25_vl.tt.common import (
    PagedAttentionConfig,
    merge_vision_tokens,
    multimodal_rope_from_hf,
    preprocess_inputs_prefill,
    sample_host,
)
from models.demos.qwen25_vl.tt.generator import Generator
from models.demos.qwen25_vl.tt.model import DropInVisionTransformer, Transformer
from models.demos.qwen25_vl.tt.model_config import VisionModelArgs
from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs, determine_device_name


HF_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

# Current P100A setup exposes one physical device. Keep this as the physical
# mesh, not the 11x10 chip core grid.
MESH_SHAPE = (1, 1)

IMAGE_PATH = Path("models/tt_transformers/demo/sample_prompts/llama_models/dog.jpg")
TEXT_PROMPT = "Describe this image in detail."

MAX_BATCH_SIZE = 1
MAX_SEQ_LEN = 4096
MAX_GENERATED_TOKENS = 200
STOP_AT_EOS = True
ENABLE_TRACE = True
USE_TT_VISION = True
FORCE_HOST_SAMPLING = True

PAGE_PARAMS = {"page_block_size": 32, "page_max_num_blocks": 1024}
SAMPLING_PARAMS = {"temperature": 0.0, "top_p": 0.08}

TRACE_REGION_SIZE = 36_000_000 if is_blackhole() else 28_467_200
NUM_COMMAND_QUEUES = 1
CLEAR_WEIGHT_CACHE_ON_START = False


def performance_optimizations(model_args):
    return DecodersPrecision.performance(model_args.n_layers, model_args.model_name)


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
        return Path(tt_cache_path) / device_name
    return Path("model_cache") / HF_MODEL / device_name


def clear_weight_cache(device_name):
    if not CLEAR_WEIGHT_CACHE_ON_START:
        return

    cache_path = weight_cache_root(device_name)
    if cache_path.exists():
        logger.warning(f"Deleting existing TT weight cache: {cache_path}")
        shutil.rmtree(cache_path)
    else:
        logger.info(f"No existing TT weight cache to delete: {cache_path}")


def create_tt_page_table(paged_attention_config, model_args):
    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    return reverse_permutation.reshape(
        model_args.max_batch_size,
        paged_attention_config.max_num_blocks // model_args.max_batch_size,
    )


def create_tt_model(mesh_device):
    model_args = ModelArgs(
        mesh_device,
        instruct=True,
        max_batch_size=MAX_BATCH_SIZE,
        optimizations=performance_optimizations,
        max_seq_len=MAX_SEQ_LEN,
    )
    model_args.use_qk_fused = False

    state_dict = model_args.load_state_dict()
    paged_attention_config = PagedAttentionConfig(
        block_size=PAGE_PARAMS["page_block_size"],
        max_num_blocks=PAGE_PARAMS["page_max_num_blocks"],
    )

    model = Transformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=ttnn.bfloat8_b,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b),
        paged_attention_config=paged_attention_config,
    )
    tt_kv_cache = [layer.attention.layer_past for layer in model.layers]
    page_table = create_tt_page_table(paged_attention_config, model_args)
    return model_args, model, page_table, tt_kv_cache


def make_messages():
    image_uri = IMAGE_PATH.resolve().as_posix()
    if not image_uri.startswith(("file://", "http://", "https://")):
        image_uri = "file://" + image_uri

    return [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_uri},
                    {"type": "text", "text": TEXT_PROMPT},
                ],
            }
        ]
    ]


def prepare_inputs(processor, messages):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )


def run_generation(mesh_device):
    model_args, model, page_table, tt_kv_cache = create_tt_model(mesh_device)
    processor = model_args.processor or AutoProcessor.from_pretrained(model_args.CKPT_DIR)
    tokenizer = model_args.tokenizer
    generator = Generator(model, model_args, mesh_device, processor=processor, tokenizer=tokenizer)

    ref_model_name = model_args.CKPT_DIR
    reference_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        ref_model_name,
        torch_dtype="auto",
        device_map="auto",
    )

    if USE_TT_VISION:
        vision_model_args = VisionModelArgs(
            mesh_device,
            max_batch_size=MAX_BATCH_SIZE,
            max_seq_len=MAX_SEQ_LEN,
            optimizations=DecodersPrecision.accuracy(reference_model.config.vision_config.depth, ref_model_name),
        )
        vision_model_args.hf_config.vision_config.depth = reference_model.config.vision_config.depth
        visual_model = DropInVisionTransformer(reference_model.visual, vision_model_args, debug=False)
    else:
        visual_model = reference_model.visual

    messages = make_messages()
    inputs = prepare_inputs(processor, messages)
    pad_token_id = tokenizer.pad_token_id

    logger.info("Running vision model prefill...")
    vision_start = time.perf_counter()
    image_embeds = (
        visual_model(inputs.pixel_values, grid_thw=inputs.image_grid_thw)
        if "pixel_values" in inputs
        else torch.tensor([], dtype=torch.bfloat16)
    )
    vision_time_s = time.perf_counter() - vision_start

    logger.info("Preparing merged text + vision embeddings...")
    text_embeds = reference_model.model.language_model.embed_tokens(inputs.input_ids)
    input_embeds = merge_vision_tokens(inputs.input_ids, text_embeds, image_embeds, reference_model.config)
    assert MAX_SEQ_LEN >= max(len(x) for x in input_embeds) + MAX_GENERATED_TOKENS

    input_prefill_pt, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        input_embeds,
        model_args,
        inputs.attention_mask,
        pad_embedding=reference_model.model.language_model.embed_tokens(torch.tensor(pad_token_id)),
    )
    cos, sin, rope_deltas = multimodal_rope_from_hf(
        inputs,
        input_embeds,
        reference_model,
        model_args,
        pad_token_id=pad_token_id,
    )

    logger.info("Compiling/warming text prefill...")
    generator.prefill_forward_text(
        input_prefill_pt[0].unsqueeze(0),
        rot_mats=(cos, sin),
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=decoding_pos,
    )

    logger.info("Running text prefill...")
    prefill_start = time.perf_counter()
    logits = generator.prefill_forward_text(
        input_prefill_pt,
        rot_mats=(cos, sin),
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=decoding_pos,
    )
    prefill_time_s = time.perf_counter() - prefill_start
    generator.update_rope_deltas([rope_delta.item() for rope_delta in rope_deltas])

    out_tok = torch.argmax(logits, dim=-1)
    current_pos = torch.tensor(decoding_pos)
    all_outputs = [[int(out_tok[user].item())] for user in range(MAX_BATCH_SIZE)]
    user_done = [False] * MAX_BATCH_SIZE
    decode_times_s = []

    argmax_on_device = model._supports_on_device_sampling and not FORCE_HOST_SAMPLING
    device_sampling_params = SamplingParams(temperature=0.0, top_k=-1, top_p=1.0) if argmax_on_device else None
    logger.info(f"Sampling path: {'device' if argmax_on_device else 'host'}")

    logger.info("Starting decode loop...")
    for iteration in range(MAX_GENERATED_TOKENS):
        iter_start = time.perf_counter()
        logits, _ = generator.decode_forward(
            out_tok,
            current_pos,
            enable_trace=ENABLE_TRACE,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            sampling_params=device_sampling_params,
        )

        if argmax_on_device:
            out_tok = logits.unsqueeze(1)
        else:
            _, out_tok = sample_host(
                logits,
                None,
                temperature=SAMPLING_PARAMS["temperature"],
                top_p=SAMPLING_PARAMS["top_p"],
                on_host=True,
            )

        decode_times_s.append(time.perf_counter() - iter_start)
        current_pos += 1

        for user in range(MAX_BATCH_SIZE):
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

    outputs = [tokenizer.decode(output).strip() for output in all_outputs]
    return outputs, vision_time_s, prefill_time_s, decode_times_s, prefill_lens


def print_perf(vision_time_s, prefill_time_s, decode_times_s, prefill_lens):
    steady_decode_times = decode_times_s[1:] if len(decode_times_s) > 1 else decode_times_s
    avg_decode_s = sum(steady_decode_times) / len(steady_decode_times) if steady_decode_times else 0.0
    tok_s_user = 1.0 / avg_decode_s if avg_decode_s else 0.0

    logger.info(f"Vision prefill latency: {vision_time_s * 1000:.2f} ms")
    logger.info(f"Text prefill latency: {prefill_time_s * 1000:.2f} ms")
    logger.info(f"Prefill padded length: {prefill_lens[0]}")
    if decode_times_s:
        logger.info(f"First decode iteration latency: {decode_times_s[0] * 1000:.2f} ms")
    logger.info(f"Generated decode tokens/user: {len(decode_times_s)}")
    logger.info(f"Average steady decode latency: {avg_decode_s * 1000:.2f} ms/token")
    logger.info(f"Decode throughput: {tok_s_user:.2f} tok/s/user")


def main():
    os.environ["HF_MODEL"] = HF_MODEL
    os.environ.pop("TT_CACHE_PATH", None)

    logger.info(f"HF_MODEL={HF_MODEL}")
    logger.info(f"IMAGE_PATH={IMAGE_PATH}")
    logger.info(f"TEXT_PROMPT={TEXT_PROMPT!r}")

    mesh_device = None
    try:
        mesh_device = open_mesh_device()
        device_name = determine_device_name(mesh_device)
        logger.info(f"Opened mesh with {mesh_device.get_num_devices()} device(s); detected {device_name}")
        clear_weight_cache(device_name)

        outputs, vision_time_s, prefill_time_s, decode_times_s, prefill_lens = run_generation(mesh_device)

        print("\n== IMAGE ==")
        print(str(IMAGE_PATH))
        print("\n== TEXT PROMPT ==")
        print(TEXT_PROMPT)
        print("\n== OUTPUT ==")
        print(outputs[0])
        print_perf(vision_time_s, prefill_time_s, decode_times_s, prefill_lens)
    finally:
        if mesh_device is not None:
            ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
