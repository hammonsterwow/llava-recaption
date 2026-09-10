#!/usr/bin/env python3
import argparse
import gc
import json
import os
import time
import traceback
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info


DEFAULT_PROMPT = """Generate a detailed and visually grounded description of the image.

Describe all visually salient information, including:
- the overall scene and setting,
- salient objects and people,
- important visual attributes such as color, shape, material, appearance, count, and state,
- actions and interactions,
- spatial relationships between objects,
- relevant foreground and background elements,
- and clearly legible text when applicable.

Pay particular attention to individual objects, their attributes, and pairwise relationships.
Only describe information that is visually supported by the image.
Do not infer hidden intentions, identities, causes, exact locations, brands, or events unless they are clearly visible.
Do not add generic world knowledge that is not supported by the image.

Write one coherent, detailed natural-language paragraph.
Avoid unnecessary repetition.
The description length should adapt to the visual complexity of the image.
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-json", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--shard-id", type=int, required=True)
    p.add_argument("--num-shards", type=int, default=4)

    p.add_argument("--min-visual-tokens", type=int, default=256)
    p.add_argument("--max-visual-tokens", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=256)

    # TITAN Xp: fp16 saves memory but may be slow on Pascal.
    # Benchmark fp16 vs fp32 on a small subset if desired.
    p.add_argument(
        "--compute-dtype",
        choices=["fp16", "fp32"],
        default="fp16",
        help="bitsandbytes 4-bit compute dtype",
    )

    p.add_argument(
        "--model-dtype",
        choices=["fp16", "fp32"],
        default="fp16",
        help="dtype for non-quantized modules; fp16 is recommended for 12 GB VRAM",
    )

    p.add_argument("--limit", type=int, default=None,
                   help="Process only N samples from this shard; useful for smoke tests.")
    p.add_argument("--prompt-file", type=str, default=None)
    p.add_argument("--flush-every", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def dtype_from_name(name):
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(name)


def load_done_ids(path: Path):
    done = set()
    if not path.exists():
        return done

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if row.get("id") is not None and str(row.get("caption", "")).strip():
                    done.add(str(row["id"]))
            except json.JSONDecodeError:
                # A final partially-written line can occur if a job is killed mid-write.
                # Ignore it; the sample will be regenerated.
                print(f"[WARN] malformed JSONL at {path}:{line_no}; ignoring")
    return done


def append_jsonl(f, obj, flush=False):
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    if flush:
        f.flush()
        os.fsync(f.fileno())


def make_messages(image_path: Path, prompt: str, min_vtokens: int, max_vtokens: int):
    # Qwen docs explicitly support local files via file:// URI.
    uri = image_path.resolve().as_uri()
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": uri,
                    "min_pixels": int(min_vtokens * 28 * 28),
                    "max_pixels": int(max_vtokens * 28 * 28),
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


@torch.inference_mode()
def generate_caption(
    model,
    processor,
    image_path: Path,
    prompt: str,
    min_vtokens: int,
    max_vtokens: int,
    max_new_tokens: int,
):
    messages = make_messages(
        image_path=image_path,
        prompt=prompt,
        min_vtokens=min_vtokens,
        max_vtokens=max_vtokens,
    )

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        use_cache=True,
    )

    # Remove prompt tokens.
    trimmed = generated_ids[:, inputs.input_ids.shape[1]:]

    caption = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    del generated_ids, trimmed, inputs, image_inputs, video_inputs
    return caption


def is_cuda_oom(exc):
    s = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in s


def main():
    args = parse_args()

    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("shard-id must satisfy 0 <= shard-id < num-shards")
    if args.min_visual_tokens > args.max_visual_tokens:
        raise ValueError("min-visual-tokens must be <= max-visual-tokens")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("CUDA device:", torch.cuda.get_device_name(0))
    print("CUDA capability:", torch.cuda.get_device_capability(0))
    print("Model:", args.model)

    input_json = Path(args.input_json)
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"part_{args.shard_id:02d}.jsonl"
    err_path = output_dir / f"errors_{args.shard_id:02d}.jsonl"

    with input_json.open("r", encoding="utf-8") as f:
        all_samples = json.load(f)

    # Deterministic interleaved sharding: 0,4,8... / 1,5,9... etc.
    indexed = list(enumerate(all_samples))
    shard = indexed[args.shard_id::args.num_shards]

    done_ids = load_done_ids(out_path)

    # Exclude completed images before selecting this run's batch.
    remaining = [
        item for item in shard
        if str(item[1]["id"]) not in done_ids
    ]

    if args.limit is not None:
        shard = remaining[:args.limit]
    else:
        shard = remaining
    print(f"Total samples: {len(all_samples):,}")
    print(f"Shard {args.shard_id}/{args.num_shards}: {len(shard):,}")
    print(f"Already completed in shard: {len(done_ids):,}")

    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    else:
        prompt = DEFAULT_PROMPT

    compute_dtype = dtype_from_name(args.compute_dtype)
    model_dtype = dtype_from_name(args.model_dtype)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    print(
        f"Loading model in NF4: compute_dtype={compute_dtype}, "
        f"model_dtype={model_dtype}"
    )

    # Each process should see exactly one GPU through CUDA_VISIBLE_DEVICES.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        quantization_config=quant_config,
        torch_dtype=model_dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=args.min_visual_tokens * 28 * 28,
        max_pixels=args.max_visual_tokens * 28 * 28,
    )

    success = 0
    skipped = 0
    errors = 0
    start_time = time.time()

    # Line-buffered append. fsync cadence is configurable.
    with out_path.open("a", encoding="utf-8", buffering=1) as fout, \
         err_path.open("a", encoding="utf-8", buffering=1) as ferr:

        pbar = tqdm(shard, desc=f"shard {args.shard_id}", dynamic_ncols=True)

        for local_step, (source_index, sample) in enumerate(pbar, start=1):
            sample_id = str(sample["id"])

            if sample_id in done_ids:
                skipped += 1
                continue

            rel_image = sample["image"]
            image_path = image_root / rel_image

            if not image_path.exists():
                errors += 1
                append_jsonl(
                    ferr,
                    {
                        "id": sample_id,
                        "image": rel_image,
                        "source_index": source_index,
                        "error": "missing_image",
                    },
                    flush=True,
                )
                continue

            used_max_vtokens = args.max_visual_tokens

            try:
                try:
                    caption = generate_caption(
                        model=model,
                        processor=processor,
                        image_path=image_path,
                        prompt=prompt,
                        min_vtokens=args.min_visual_tokens,
                        max_vtokens=args.max_visual_tokens,
                        max_new_tokens=args.max_new_tokens,
                    )
                except Exception as e:
                    # If only this image OOMs, retry at the minimum token budget.
                    if is_cuda_oom(e) and args.max_visual_tokens > args.min_visual_tokens:
                        print(
                            f"\n[OOM] id={sample_id}; retrying with "
                            f"{args.min_visual_tokens} visual tokens"
                        )
                        torch.cuda.empty_cache()
                        gc.collect()
                        used_max_vtokens = args.min_visual_tokens
                        caption = generate_caption(
                            model=model,
                            processor=processor,
                            image_path=image_path,
                            prompt=prompt,
                            min_vtokens=args.min_visual_tokens,
                            max_vtokens=args.min_visual_tokens,
                            max_new_tokens=args.max_new_tokens,
                        )
                    else:
                        raise

                if not caption:
                    raise RuntimeError("empty_caption")

                row = {
                    "id": sample_id,
                    "image": rel_image,
                    "source_index": source_index,
                    "caption": caption,
                    "max_visual_tokens_used": used_max_vtokens,
                }

                do_flush = (success + 1) % args.flush_every == 0
                append_jsonl(fout, row, flush=do_flush)

                done_ids.add(sample_id)
                success += 1

            except Exception as e:
                errors += 1
                if is_cuda_oom(e):
                    torch.cuda.empty_cache()
                    gc.collect()

                append_jsonl(
                    ferr,
                    {
                        "id": sample_id,
                        "image": rel_image,
                        "source_index": source_index,
                        "error": repr(e),
                        "traceback": traceback.format_exc(limit=5),
                    },
                    flush=True,
                )

            elapsed = max(time.time() - start_time, 1e-6)
            rate = success / elapsed
            pbar.set_postfix(
                ok=success,
                err=errors,
                skip=skipped,
                img_s=f"{rate:.3f}",
            )

    elapsed = time.time() - start_time
    print(
        f"Finished shard {args.shard_id}: "
        f"success={success}, errors={errors}, skipped={skipped}, "
        f"elapsed={elapsed/3600:.2f} h"
    )


if __name__ == "__main__":
    main()
