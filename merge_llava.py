#!/usr/bin/env python3
import argparse
import copy
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--original-json", required=True)
    p.add_argument("--parts-dir", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--num-shards", type=int, default=4)
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any original sample is missing a recaption.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    original_json = Path(args.original_json)
    parts_dir = Path(args.parts_dir)
    output_json = Path(args.output_json)

    with original_json.open("r", encoding="utf-8") as f:
        original = json.load(f)

    caption_map = {}
    duplicate_ids = []

    for shard_id in range(args.num_shards):
        part = parts_dir / f"part_{shard_id:02d}.jsonl"
        if not part.exists():
            print(f"[WARN] missing part file: {part}")
            continue

        with part.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[WARN] malformed line ignored: {part}:{line_no}")
                    continue

                sid = str(row["id"])
                caption = str(row.get("caption", "")).strip()
                if not caption:
                    continue

                if sid in caption_map:
                    duplicate_ids.append(sid)
                caption_map[sid] = caption

    original_ids = [str(x["id"]) for x in original]
    original_id_set = set(original_ids)

    missing = [sid for sid in original_ids if sid not in caption_map]
    extras = [sid for sid in caption_map if sid not in original_id_set]

    print(f"Original samples : {len(original):,}")
    print(f"Recaptions       : {len(caption_map):,}")
    print(f"Missing          : {len(missing):,}")
    print(f"Extra IDs        : {len(extras):,}")
    print(f"Duplicate IDs    : {len(duplicate_ids):,}")

    if duplicate_ids:
        raise RuntimeError(
            f"Duplicate recaption IDs found; first few: {duplicate_ids[:10]}"
        )

    if args.strict and missing:
        raise RuntimeError(
            f"{len(missing)} samples are missing recaptions; "
            f"first few: {missing[:10]}"
        )

    merged = []
    for sample in original:
        sid = str(sample["id"])

        # Deep-copy to preserve the original file.
        new_sample = copy.deepcopy(sample)

        if sid in caption_map:
            conversations = new_sample.get("conversations", [])
            gpt_indices = [
                i for i, turn in enumerate(conversations)
                if turn.get("from") == "gpt"
            ]

            if not gpt_indices:
                raise RuntimeError(f"No GPT turn for sample id={sid}")

            # Pretraining data normally has a single GPT answer.
            conversations[gpt_indices[0]]["value"] = caption_map[sid]

        merged.append(new_sample)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False)

    print(f"Saved: {output_json}")


if __name__ == "__main__":
    main()
