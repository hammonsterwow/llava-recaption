# LLaVA-558K recaptioning on TITAN Xp x4

## 1. Environment
Recommended: Linux, Python 3.10+, CUDA/PyTorch setup compatible with TITAN Xp (Pascal).
For bitsandbytes Pascal support, prefer a CUDA 11.8-12.6 compatible environment.

```bash
pip install -r requirements.txt
```

Check:

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
PY
```

## 2. One-GPU smoke test

```bash
CUDA_VISIBLE_DEVICES=0 python recaption.py \
  --input-json /DATA/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
  --image-root /DATA/LLaVA-Pretrain/images \
  --output-dir /DATA/recaption_test \
  --shard-id 0 \
  --num-shards 1 \
  --limit 20 \
  --min-visual-tokens 256 \
  --max-visual-tokens 512 \
  --max-new-tokens 256 \
  --compute-dtype fp16 \
  --model-dtype fp16
```

Monitor:

```bash
watch -n 1 nvidia-smi
```

## 3. Four-GPU smoke test

```bash
LIMIT=20 bash launch_4gpu.sh \
  /DATA/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
  /DATA/LLaVA-Pretrain/images \
  /DATA/qwen_dense_558k_test
```

`LIMIT=20` means 20 images per shard, so 80 total.

## 4. Full run

Run inside tmux/screen:

```bash
bash launch_4gpu.sh \
  /DATA/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
  /DATA/LLaVA-Pretrain/images \
  /DATA/qwen_dense_558k
```

Outputs:
- `part_00.jsonl` ... `part_03.jsonl`
- `errors_00.jsonl` ... `errors_03.jsonl`
- `logs/gpu_0.log` ... `logs/gpu_3.log`

If interrupted, run the same command again. Existing successful IDs are skipped.

## 5. Merge back to LLaVA format

Only use `--strict` after all workers are complete and error files are resolved.

```bash
python merge_llava.py \
  --original-json /DATA/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
  --parts-dir /DATA/qwen_dense_558k \
  --output-json /DATA/qwen_dense_558k.json \
  --strict
```

## 6. Validate

```bash
python validate_llava.py \
  --json /DATA/qwen_dense_558k.json \
  --image-root /DATA/LLaVA-Pretrain/images
```

## TITAN Xp notes

Start with:
- batch size: 1 (hard-coded)
- 4-bit NF4
- min visual tokens: 256
- max visual tokens: 512
- max new tokens: 256
- compute dtype: fp16
- model dtype: fp16

If OOM persists for many images:
- set `MAX_VTOK=384`
- then `MAX_VTOK=256`

If throughput is extremely low, benchmark a small subset with:
- `COMPUTE_DTYPE=fp32`
- keep `MODEL_DTYPE=fp16` first

Example:
```bash
LIMIT=20 COMPUTE_DTYPE=fp32 bash launch_4gpu.sh ...
```

Do not set `MODEL_DTYPE=fp32` first on a 12 GB GPU; it can materially increase VRAM usage.
