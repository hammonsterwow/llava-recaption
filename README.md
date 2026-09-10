# LLaVA-558K recaptioning on TITAN Xp x4

This repository generates dense image captions for the LLaVA-558K pretraining images with `Qwen/Qwen2.5-VL-7B-Instruct` using four independent GPU workers.

The production prompt is stored in [`prompt_kakao.txt`](prompt_kakao.txt) and is also the exact built-in default in `recaption.py`. Do not change the prompt after production generation has started unless you intentionally restart the dataset from scratch.

## Current server paths

```text
work dir   : /mnt3/jisung/recaption
model      : /mnt3/jisung/models/Qwen2.5-VL-7B-Instruct
input meta : /mnt3/jisung/recaption/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k_meta.json
original   : /mnt3/jisung/recaption/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json
image root : /mnt3/jisung/recaption/data/LLaVA-Pretrain
output dir : /mnt3/jisung/qwen_dense_558k
```

The image paths in the meta JSON are relative paths such as `00453/004539375.jpg`, so the image root is `.../LLaVA-Pretrain`, not an `images/` subdirectory.

## Environment

For the currently configured server:

```bash
source /mnt3/jisung/miniforge3/etc/profile.d/conda.sh
conda activate recaption
export PYTHONNOUSERSITE=1
cd /mnt3/jisung/recaption
```

Check CUDA if needed:

```bash
nvidia-smi
python3.10 - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
PY
```

## Important: discard results generated with the previous prompt

Before starting the new production run, make sure no recaption worker is running:

```bash
pgrep -af '[r]ecaption.py'
```

If the command prints nothing, it is safe to remove the old test/production outputs that were generated with the previous prompt:

```bash
rm -rf /mnt3/jisung/qwen_dense_558k
rm -rf /mnt3/jisung/recaption_test_01
rm -rf /mnt3/jisung/recaption_test_4gpu_01
```

Do not delete the input data or downloaded Qwen model.

## 1. Smoke test: one GPU, 20 images

```bash
CUDA_VISIBLE_DEVICES=0 python3.10 recaption.py \
  --model /mnt3/jisung/models/Qwen2.5-VL-7B-Instruct \
  --input-json /mnt3/jisung/recaption/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k_meta.json \
  --image-root /mnt3/jisung/recaption/data/LLaVA-Pretrain \
  --output-dir /mnt3/jisung/recaption_test_01 \
  --shard-id 0 \
  --num-shards 1 \
  --limit 20 \
  --min-visual-tokens 256 \
  --max-visual-tokens 512 \
  --max-new-tokens 256 \
  --compute-dtype fp16 \
  --model-dtype fp16 \
  --prompt-file /mnt3/jisung/recaption/prompt_kakao.txt
```

Expected: 20 successes and 0 errors.

## 2. Review image and caption together

`review.py` creates a self-contained HTML file with the images embedded, so the HTML can be copied to another computer and opened directly.

```bash
python3.10 review.py \
  --parts-dir /mnt3/jisung/recaption_test_01 \
  --image-root /mnt3/jisung/recaption/data/LLaVA-Pretrain \
  --meta-json /mnt3/jisung/recaption/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k_meta.json \
  --output-html /mnt3/jisung/recaption_test_01/review.html \
  --sample-size 20
```

The page shows the image, the generated Qwen caption, and the original BLIP caption when available.

## 3. Use tmux for long runs

Yes: production runs should continue to use `tmux`. `tmux` keeps the server process alive when VS Code SSH or the laptop disconnects.

Create a session:

```bash
tmux new -s llava_recaption
```

Inside tmux, activate the environment:

```bash
source /mnt3/jisung/miniforge3/etc/profile.d/conda.sh
conda activate recaption
export PYTHONNOUSERSITE=1
cd /mnt3/jisung/recaption
```

Detach without stopping the job:

```text
Ctrl+B, then D
```

Reconnect later:

```bash
tmux attach -t llava_recaption
```

An SSH disconnect does not stop a job running inside tmux. A server shutdown/reboot does stop it, but the next run can resume from the saved JSONL results.

## 4. Four-GPU production run

### Full run

Use this when all four GPUs may stay allocated for a long period:

```bash
MODEL=/mnt3/jisung/models/Qwen2.5-VL-7B-Instruct \
PROMPT_FILE=/mnt3/jisung/recaption/prompt_kakao.txt \
bash launch_4gpu.sh \
  /mnt3/jisung/recaption/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k_meta.json \
  /mnt3/jisung/recaption/data/LLaVA-Pretrain \
  /mnt3/jisung/qwen_dense_558k
```

### Time-limited session

`LIMIT` is the number of remaining samples processed by each shard in that invocation. For example, `LIMIT=120` processes at most 480 images total across four GPUs:

```bash
MODEL=/mnt3/jisung/models/Qwen2.5-VL-7B-Instruct \
PROMPT_FILE=/mnt3/jisung/recaption/prompt_kakao.txt \
LIMIT=120 \
bash launch_4gpu.sh \
  /mnt3/jisung/recaption/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k_meta.json \
  /mnt3/jisung/recaption/data/LLaVA-Pretrain \
  /mnt3/jisung/qwen_dense_558k
```

Re-running the same command with the same output directory skips successful IDs first and then selects the next remaining batch.

## 5. Resume and crash safety

Results are written as:

```text
part_00.jsonl ... part_03.jsonl
errors_00.jsonl ... errors_03.jsonl
logs/gpu_0.log ... logs/gpu_3.log
run_config.json
```

Successful captions are appended immediately. If the server stops while the last JSONL line is being written, `recaption.py` repairs only that trailing partial record on the next run and regenerates the affected sample.

`run_config.json` records the model, prompt hash/text, sharding, visual token settings, dtype settings, input JSON, and image root. A later run using the same output directory but different generation settings is rejected so that captions produced under different conditions are not silently mixed.

`launch_4gpu.sh` also places a lock in the output directory. A second launcher cannot write to the same result files at the same time. A stale same-host lock left by a crashed launcher is removed automatically after its PID is no longer alive.

## 6. Monitor progress

Worker processes:

```bash
pgrep -af '[r]ecaption.py'
```

GPU status:

```bash
nvidia-smi
```

One worker log:

```bash
tail -f /mnt3/jisung/qwen_dense_558k/logs/gpu_0.log
```

Total successful lines:

```bash
cat /mnt3/jisung/qwen_dense_558k/part_*.jsonl | wc -l
```

Errors:

```bash
wc -l /mnt3/jisung/qwen_dense_558k/errors_*.jsonl
```

## 7. Stop intentionally

When a launcher is running in the foreground inside tmux, prefer `Ctrl+C` in that tmux window. The launcher traps the signal and terminates its four worker processes before releasing the output lock.

If the server itself shuts down, simply restart later with the same output directory. Completed IDs are skipped.

## 8. Merge after all recaptions are complete

Only use `--strict` after all workers are complete and remaining errors have been resolved.

```bash
python3.10 merge_llava.py \
  --original-json /mnt3/jisung/recaption/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
  --parts-dir /mnt3/jisung/qwen_dense_558k \
  --output-json /mnt3/jisung/qwen_dense_558k.json \
  --num-shards 4 \
  --strict
```

Validate:

```bash
python3.10 validate_llava.py \
  --json /mnt3/jisung/qwen_dense_558k.json \
  --image-root /mnt3/jisung/recaption/data/LLaVA-Pretrain
```

## TITAN Xp defaults

The current baseline is:

```text
batch size        : 1
quantization      : 4-bit NF4
min visual tokens : 256
max visual tokens : 512
max new tokens    : 256
compute dtype     : fp16
model dtype       : fp16
```

If individual images OOM at 512 visual tokens, `recaption.py` retries that image at the minimum visual-token budget. If many images OOM, reduce `MAX_VTOK` only in a fresh output directory unless you intentionally want to restart the dataset with a different configuration.
