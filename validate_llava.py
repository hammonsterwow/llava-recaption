#!/usr/bin/env python3
import argparse
import json
import os
import statistics
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True)
    p.add_argument("--image-root", required=True)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    ids = [str(x["id"]) for x in data]
    dup_count = len(ids) - len(set(ids))

    missing_images = []
    empty_captions = []
    word_counts = []

    for x in data:
        image_path = os.path.join(args.image_root, x["image"])
        if not os.path.exists(image_path):
            missing_images.append(x["id"])

        gpt_turns = [
            t for t in x.get("conversations", [])
            if t.get("from") == "gpt"
        ]

        if not gpt_turns or not str(gpt_turns[0].get("value", "")).strip():
            empty_captions.append(x["id"])
        else:
            cap = str(gpt_turns[0]["value"]).strip()
            word_counts.append(len(cap.split()))

    print(f"samples        : {len(data):,}")
    print(f"duplicate ids  : {dup_count:,}")
    print(f"missing images : {len(missing_images):,}")
    print(f"empty captions : {len(empty_captions):,}")

    if word_counts:
        print(f"caption words mean   : {statistics.mean(word_counts):.1f}")
        print(f"caption words median : {statistics.median(word_counts):.1f}")
        print(f"caption words min/max: {min(word_counts)} / {max(word_counts)}")

    if missing_images:
        print("first missing image ids:", missing_images[:10])
    if empty_captions:
        print("first empty caption ids:", empty_captions[:10])


if __name__ == "__main__":
    main()
