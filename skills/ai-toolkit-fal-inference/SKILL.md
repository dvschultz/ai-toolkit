---
name: ai-toolkit-fal-inference
description: Run real-prompt inference on trained ai-toolkit Klein 9B base LoRAs via fal.ai's hosted Klein endpoint. Use whenever the user wants to test a trained LoRA against custom prompts WITHOUT provisioning a custom GPU pod — A/B comparing checkpoints, testing an artist's prompt against multiple versions, generating presentation samples, or validating a checkpoint against prompts that weren't in the training sample matrix. Triggers on phrases like "test this LoRA via fal", "run inference on fal", "generate with my v3 / v4 LoRA against this prompt", "A/B these checkpoints on fal", "send this LoRA to fal", "compare these LoRAs on the same prompt", or any request to make images from a local .safetensors against custom prompts. Distinct from `ai-toolkit-remote-launch` (which spins up a training pod) and from picking-from-existing-samples (which uses the training-config sample matrix). This skill is specifically for ad-hoc real-prompt inference on the fal hosted endpoint.
---

# AI Toolkit fal Inference

Driver for `scripts/fal/inference.py` — runs LoRA inference on
fal.ai's `fal-ai/flux-2/klein/9b/base/lora` endpoint so you can test trained
checkpoints against real prompts without spinning up your own GPU pod.

## When you've been triggered

The user wants one of:
1. **A/B test multiple checkpoints** against the same prompts to pick a winner
   (e.g. "test v4 step 1000 and 1250 against these two prompts")
2. **Validate a checkpoint** against a prompt set that wasn't in the training
   sample matrix (e.g. artist-supplied prompts after they've seen the first
   round of samples)
3. **Generate presentation images** — final-quality samples at fal's hosted
   inference for a model the user is about to ship

The script handles the rest: LoRA upload (with hash-keyed cache so re-runs
skip the upload), submission, parallel polling, image download, per-LoRA
output organization, manifest JSON.

## Inputs you need

Before invoking, confirm:

1. **LoRA file paths** — local `.safetensors` files. The user usually has 1–3
   they want to compare. Each gets a short label (defaults to filename stem
   if not provided).
2. **Prompts** — exact text the user wants to test. Include the trigger phrase
   (e.g. `silex relief, 4m0nsx`) if applicable — the script does not add it.
3. **Seeds** — explicit list (for reproducible A/B) or `--num-seeds N` for N
   random. **For A/B comparisons, ALWAYS use explicit seeds** so the same
   seed produces directly comparable images across LoRAs.
4. **Optional overrides** — LoRA `--scale` (default 1.5), `--guidance`,
   `--steps`, `--image-size`. Default fal settings are usually fine.

If anything is missing, ask once. Don't proceed on guesses — wrong prompt
means wrong test.

## Required env

`FAL_KEY` must be in `.env` at the repo root (or exported). The script
auto-loads `.env` via python-dotenv. If `FAL_KEY` is missing, the script
errors out immediately.

## How to invoke

The canonical A/B pattern:

```bash
.venv-captioning/bin/python scripts/fal/inference.py \
    --run <descriptive_run_name> \
    --lora <label_A>:output/<run>/<lora_A>.safetensors \
    --lora <label_B>:output/<run>/<lora_B>.safetensors \
    --lora <label_C>:output/<run>/<lora_C>.safetensors \
    --prompt "<exact prompt 1, trigger included>" \
    --prompt "<exact prompt 2, trigger included>" \
    --seed 42 --seed 123 --seed 7
```

Output lands in `output/inference/<run>/<lora_label>/p<idx>_s<seed>.png` plus a
`manifest.json` covering every request + response.

### Why the captioning venv

`fal-client` is installed in `.venv-captioning/` (alongside `google-genai`).
The main venv doesn't have it. Always invoke with
`.venv-captioning/bin/python`, not the system `python`.

## Defaults that matter

| Field | Default | Why |
|---|---|---|
| `--scale` | **1.5** | Calibrated higher than ai-toolkit's 1.0 per the [fal LoRA strength calibration] memory. fal interprets LoRA strength differently than the training-time inference path; under-strength produces weak style hits. |
| `--guidance` | 5.0 | fal endpoint default |
| `--steps` | 28 | fal endpoint default |
| `--image-size` | `landscape_4_3` | fal endpoint default. Override with WxH like `1024x1024` for square. |
| `--acceleration` | `regular` | fal default; `high` for faster but possibly lower quality |
| `--negative-prompt` | `""` (empty) | Add one if gibberish-text leak is a problem at inference — see [Suppressing LoRA text hallucination at inference] memory |
| `--workers` | 4 | Concurrent inference requests; fal handles this fine |

## LoRA upload + caching

The first time a `.safetensors` is referenced, the script uploads it to fal
storage (~30-60 sec per 700MB file on a normal connection) and saves the
returned CDN URL keyed by the file's sha256 in
`scripts/fal/.fal_lora_cache.json`. Subsequent runs that reference the same
file skip the upload.

**Privacy model (read once, internalize):** fal storage URLs are random
30-char tokens. They're not publicly discoverable, but anyone who has the
URL can download. The URL never leaves your machine + fal's worker. For
testing, this is fine. For shipping a model to production with stricter
guarantees, switch to a signed S3/R2 URL — the `--lora` argument accepts
any URL, not just local paths (paths get auto-uploaded; full URLs are
passed through).

Per the [private LoRA hosting on fal: URL secrecy is the only model] memory,
HuggingFace private repos are NOT a working option for this endpoint — the
fal schema has no per-LoRA auth field. fal storage is the right default.

## Output layout

```
output/inference/<run_name>/
├── manifest.json                # all params + per-image results + errors
├── <lora_label_A>/
│   ├── p0_s42.png
│   ├── p0_s123.png
│   ├── p0_s7.png
│   ├── p1_s42.png
│   ├── p1_s123.png
│   └── p1_s7.png
├── <lora_label_B>/
│   └── ...
└── <lora_label_C>/
    └── ...
```

Naming: `p{prompt_index}_s{seed}.png` — keep prompt indices stable across
LoRAs so the same `p0_s42.png` in different folders compares the same prompt
at the same seed across checkpoints.

## Cost

fal Klein 9B is **$0.02 per megapixel**. landscape_4_3 at default size is
~0.78 MP, so ~$0.016 per image. An 18-image A/B (3 LoRAs × 2 prompts × 3
seeds) costs **~$0.30**. Effectively free for testing.

## Common patterns

### A/B comparing checkpoints (the main use case)

```bash
.venv-captioning/bin/python scripts/fal/inference.py \
    --run amon_v3v4_ab \
    --lora v3_winner:output/amon_silex_klein_9b_v3/amon_silex_klein_9b_v3_000004250.safetensors \
    --lora v4_1000:output/amon_silex_klein_9b_v4/amon_silex_klein_9b_v4_000001000.safetensors \
    --lora v4_1250:output/amon_silex_klein_9b_v4/amon_silex_klein_9b_v4_000001250.safetensors \
    --prompt "suburban houses below a smoking industrial complex under a blue sky, silex relief, 4m0nsx" \
    --prompt "a herd of horses standing in a grassy field under a blue sky, silex relief, 4m0nsx" \
    --seed 42 --seed 123 --seed 7
```

### Validating one checkpoint against artist-supplied prompts

```bash
.venv-captioning/bin/python scripts/fal/inference.py \
    --run pre_ship_validation \
    --lora final:output/<run>/<run>_<step>.safetensors \
    --prompt "<artist prompt 1>" --prompt "<artist prompt 2>" \
    --num-seeds 3
```

### Tuning LoRA strength

Sweep `--scale` if outputs are too weak or too strong:

```bash
# Same prompt + seed at three different scales
for s in 1.0 1.5 2.0; do
    .venv-captioning/bin/python scripts/fal/inference.py \
        --run scale_sweep_$s \
        --lora my_lora:path/to.safetensors \
        --prompt "<prompt>" \
        --seed 42 \
        --scale $s
done
```

## Things to watch for

- **Trigger phrase IS required in the prompt**, just like in training. The
  script doesn't auto-append. If the user's prompt looks suspiciously vanilla
  (no `silex relief, 4m0nsx`-style suffix), ask whether they meant to include
  it.
- **A/B without explicit seeds is uninformative.** Random seeds across LoRAs
  means you're comparing different latent noise, not different LoRAs. Always
  pass `--seed X --seed Y --seed Z` for comparison runs.
- **The fal endpoint is `flux-2/klein/9b/base/lora` specifically** — for LoRAs
  trained against `black-forest-labs/FLUX.2-klein-base-9B`. If a LoRA was
  trained against a different Klein variant (dev, distilled, etc.) this
  endpoint won't load it correctly. Check the training config's
  `model.name_or_path` before using.
- **Errors are logged per-image, not fatal.** A flaky inference job won't
  abort the whole batch; the failed job appears in `manifest.json.errors`
  and the rest of the batch completes.

## When NOT to use this skill

- The user wants to PICK a checkpoint from existing training samples → use
  `ai-toolkit-sample-reviewer` instead (works from the local sample images
  the trainer already produced).
- The user wants to TRAIN a new LoRA → `ai-toolkit-remote-launch`.
- The model is not a Klein 9B base LoRA (e.g. Wan22 video, Flux dev, Z-Image
  Turbo). This endpoint is Klein-specific. Other models need their own fal
  endpoints — the script can be adapted, but the current default endpoint
  is hard-coded.

## Related skills

- `ai-toolkit-remote-launch` / `monitor` / `teardown` — training pipeline
- `ai-toolkit-sample-reviewer` — pick a checkpoint from training samples
- `flux2-klein-prompter` — write prompts that play well with Klein
