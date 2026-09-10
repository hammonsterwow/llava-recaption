#!/usr/bin/env python3
import argparse
import base64
import html
import json
import mimetypes
import random
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parts-dir", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--output-html", required=True)
    p.add_argument("--sample-size", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--meta-json",
        default=None,
        help="Optional LLaVA meta JSON; shows the original blip_caption when available.",
    )
    return p.parse_args()


def read_rows(parts_dir: Path):
    rows = []
    for part in sorted(parts_dir.glob("part_*.jsonl")):
        with part.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("id") is not None and str(row.get("caption", "")).strip():
                    rows.append(row)
    return rows


def load_original_captions(meta_json: Path, wanted_ids):
    if meta_json is None:
        return {}
    with meta_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    wanted = set(wanted_ids)
    return {
        str(row.get("id")): row.get("blip_caption")
        for row in data
        if str(row.get("id")) in wanted
    }


def image_data_uri(path: Path):
    mime, _ = mimetypes.guess_type(path.name)
    mime = mime or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def main():
    args = parse_args()
    parts_dir = Path(args.parts_dir)
    image_root = Path(args.image_root)
    output_html = Path(args.output_html)

    rows = read_rows(parts_dir)
    if not rows:
        raise RuntimeError(f"No valid captions found in {parts_dir}/part_*.jsonl")

    random.seed(args.seed)
    selected = random.sample(rows, min(args.sample_size, len(rows)))
    original = load_original_captions(
        Path(args.meta_json) if args.meta_json else None,
        [str(row["id"]) for row in selected],
    )

    cards = []
    for row in selected:
        sample_id = str(row["id"])
        rel_image = str(row["image"])
        image_path = image_root / rel_image

        if image_path.exists():
            image_src = image_data_uri(image_path)
            image_block = f'<img src="{image_src}" alt="{html.escape(rel_image)}">'
        else:
            image_block = '<div class="missing">Image file is missing.</div>'

        old_caption = original.get(sample_id)
        old_html = ""
        if old_caption is not None:
            old_html = f"""
            <div class="label">Original BLIP caption</div>
            <div class="caption old">{html.escape(str(old_caption))}</div>
            """

        cards.append(
            f"""
            <section class="card">
              <h2>ID {html.escape(sample_id)}</h2>
              <div class="path">{html.escape(rel_image)}</div>
              {image_block}
              <div class="label">Generated Qwen caption</div>
              <div class="caption">{html.escape(str(row['caption']))}</div>
              {old_html}
            </section>
            """
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLaVA Recaption Review</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f5f5f5; color: #111; }}
main {{ max-width: 1000px; margin: 0 auto; padding: 28px; }}
.card {{ background: white; border: 1px solid #ddd; border-radius: 12px; padding: 20px; margin: 0 0 24px; }}
.card img {{ display: block; max-width: 100%; max-height: 700px; object-fit: contain; margin: 16px auto; border-radius: 8px; }}
h1 {{ margin-top: 0; }}
h2 {{ margin-bottom: 4px; }}
.path {{ color: #666; font-family: monospace; overflow-wrap: anywhere; }}
.label {{ font-weight: 700; margin-top: 16px; }}
.caption {{ margin-top: 6px; line-height: 1.55; white-space: pre-wrap; }}
.old {{ color: #555; }}
.missing {{ padding: 24px; border: 1px dashed #c00; color: #c00; margin: 16px 0; }}
</style>
</head>
<body>
<main>
<h1>LLaVA Recaption Review</h1>
<p>Random sample: {len(selected)} / {len(rows)} generated captions</p>
{''.join(cards)}
</main>
</body>
</html>
"""

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(document, encoding="utf-8")
    print(f"Saved: {output_html}")
    print("The images are embedded in the HTML, so the file can be copied to another computer and opened directly.")


if __name__ == "__main__":
    main()
