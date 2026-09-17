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
1. **Pinned-seed A/B test multiple checkpoints** against the same prompts at the
   same noise — for "did checkpoint X change behavior Y" controlled comparisons
   (e.g. "test v4 step 1000 vs 1250 at the same seed")
2. **Capability-sample a model** to assess what it's actually capable of producing —
   for "find the most accurate / best checkpoint" decisions (e.g. "which v4
   checkpoint is most dataset-accurate?")
3. **Validate a checkpoint** against artist-supplied prompts after they've seen
   the first round of samples
4. **Generate presentation images** — final-quality samples for shipping

## The TWO modes — pick the right one

This is the most important methodology choice in the skill. Pick wrong and the
conclusions you draw won't generalize.

### Mode A: Pinned-seed A/B (the `--seed` flag, repeated)

**When**: you want to know "given identical conditions, which LoRA produces a
better output." Latent noise is controlled across LoRAs — the only variable is
the LoRA itself. Use for:
- Direct visual A/B between checkpoints
- Comparing scale-sweep outputs (same noise across scales)
- Quick "does checkpoint X behave differently from Y on the same prompt"

**Sample size**: 3–6 seeds × 2–3 prompts is enough. Not designed for capability
characterization.

**Invocation**: `--seed 42 --seed 123 --seed 7` (explicit, shared across LoRAs).

### Mode B: Capability sampling (the `--per-lora-num-seeds` flag) — DEFAULT for "find best checkpoint"

**When**: you want to know "what is this model CAPABLE of producing across its
output distribution." Each image gets a fresh seed; seeds are NOT shared across
LoRAs. Use for:
- Picking the most-accurate checkpoint from many candidates
- Pre-ship validation of a single model
- Showing an artist the model's range
- Any judgment that depends on the *distribution* of outputs, not a specific noise

**Sample size**: **10+ images per model minimum**, across **5+ prompts of varied
complexity**. A single seed never tells you what a model is capable of.

**Invocation**: `--per-lora-num-seeds 2` with 5+ prompts → 10+ images per LoRA,
each with a unique fresh seed.

### Prompts for capability sampling: cover a range of complexities

Don't just use 2 prompts of similar shape. Mix:
- **Training-format match**: shortest noun-glyph e.g. `"a dolphin glyph, silex relief, 4m0nsx"`
- **Novel single subject**: noun the dataset never had e.g. `"a key glyph, ..."`
- **Multi-subject group**: e.g. `"a horse herd glyph, ..."`
- **Architectural / complex scene**: e.g. `"an industrial landscape glyph, ..."`
- **Organic / soft subject**: e.g. `"a butterfly glyph, ..."` or `"a flower glyph, ..."`
- **Edge case**: e.g. `"a text glyph"`, an abstract concept, a deliberately ambiguous prompt

5–6 such prompts × 2 seeds each = 10–12 images per LoRA. Costs about $0.20 per
LoRA. Pay it — single-seed conclusions are worse than no conclusions.

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
4. **Optional overrides** — LoRA `--scale` (default 1.4, calibrated via 0.1-step strength sweep — see [fal strength sweep methodology] memory), `--guidance`,
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
| `--scale` | **1.4** | Calibrated via 0.1-step strength sweep on Klein 9B base (see [fal strength sweep methodology] memory). Below ~1.3 the LoRA fires without arrow/callout grammar; at 1.4 full V1-style annotation grammar (arrows + multi-element labels + iridescent variant) lands without compositional crowding. Above ~1.7 labels start clipping the frame. Override per-call when an aesthetic goal calls for it (1.0 for minimal LoRA, 1.5-1.6 for max annotation density). |
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

### Capability sampling — pick the most-accurate checkpoint (THE primary use case)

Use this any time the user asks "which checkpoint is best / most accurate /
most dataset-aligned". 5–6 varied prompts × 2 fresh-seeds-per-LoRA-per-prompt =
10–12 images per LoRA. Each image has a UNIQUE seed; seeds are NOT shared
across LoRAs — that's the point.

```bash
.venv-captioning/bin/python scripts/fal/inference.py \
    --run v4_capability_sweep \
    --lora v4_0500:output/run/run_000000500.safetensors \
    --lora v4_1000:output/run/run_000001000.safetensors \
    --lora v4_1500:output/run/run_000001500.safetensors \
    --lora v4_2000:output/run/run_000002000.safetensors \
    --lora v4_2500:output/run/run_000002500.safetensors \
    --lora v4_3000:output/run/run_000003000.safetensors \
    --prompt "a dolphin glyph, silex relief, 4m0nsx" \
    --prompt "a key glyph, silex relief, 4m0nsx" \
    --prompt "a horse herd glyph, silex relief, 4m0nsx" \
    --prompt "an industrial landscape glyph, silex relief, 4m0nsx" \
    --prompt "a butterfly glyph, silex relief, 4m0nsx" \
    --per-lora-num-seeds 2
```

Each LoRA gets 5 prompts × 2 unique seeds = 10 images, all with fresh seeds.
The candidate checkpoint is the one whose 10-image set is MOST CONSISTENTLY
dataset-aligned — not the one whose seed-42 image happened to look good. Sample
size ≥10 is the floor; go higher for high-stakes decisions.

### Pinned-seed A/B — controlled comparison between checkpoints

Use this when the question is "did this checkpoint change behavior X" and you
want to isolate the LoRA delta from noise variance. Same seeds across LoRAs.

```bash
.venv-captioning/bin/python scripts/fal/inference.py \
    --run v3_vs_v4_pinned_ab \
    --lora v3_winner:path/to/v3_winner.safetensors \
    --lora v4_candidate:path/to/v4_candidate.safetensors \
    --prompt "<prompt 1>" --prompt "<prompt 2>" \
    --seed 42 --seed 123 --seed 7
```

Pinned A/B is for **diagnostics**, not for ranking. The right tool for "which
of these 10 LoRAs is best" is capability sampling above.

### Validating one checkpoint against artist-supplied prompts

```bash
.venv-captioning/bin/python scripts/fal/inference.py \
    --run pre_ship_validation \
    --lora final:output/<run>/<run>_<step>.safetensors \
    --prompt "<artist prompt 1>" --prompt "<artist prompt 2>" \
    --per-lora-num-seeds 5
```

### Tuning LoRA strength

Strength sweeps are inherently a PINNED-SEED operation — you want the same
noise at each scale so the only variable is strength:

```bash
.venv-captioning/bin/python scripts/fal/inference.py \
    --run scale_sweep \
    --lora my_lora:path/to.safetensors \
    --scale-sweep 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 \
    --prompt "a dolphin glyph, silex relief, 4m0nsx" \
    --prompt "a key glyph, silex relief, 4m0nsx" \
    --prompt "an industrial landscape glyph, silex relief, 4m0nsx" \
    --seed 42 --seed 123
```

Use 0.1 increments in the calibrated useful band; see the
`[fal strength sweep methodology]` memory for the rationale and current
Klein-9B calibration (default scale 1.4).

## Things to watch for

- **Trigger phrase IS required in the prompt**, just like in training. The
  script doesn't auto-append. If the user's prompt looks suspiciously vanilla
  (no `silex relief, 4m0nsx`-style suffix), ask whether they meant to include
  it.
- **Pick the right seed mode for the question.** Pinned-seed A/B (`--seed`) is
  for "did this change behavior X" diagnostic comparisons. Capability sampling
  (`--per-lora-num-seeds`) is for "which model is best" ranking. Don't use
  pinned-seed A/B to rank checkpoints — a single seed never characterizes a
  model. See the [fal capability-sampling methodology] memory.
- **Don't commit conclusions from a single seed.** Even in pinned-A/B mode,
  3+ seeds is the floor. For ranking decisions, capability sample with 10+
  images per model. "Seed-42 of checkpoint X looks great" is a starting
  point, not a conclusion.
- **Pick the endpoint with `--base` to match how the LoRA was trained.** Two
  bases are supported (default `klein-9b` for back-compat):
  - `--base klein-9b` → `fal-ai/flux-2/klein/9b/base/lora` (guidance default 5.0)
    — for LoRAs trained against `black-forest-labs/FLUX.2-klein-base-9B`.
  - `--base flux2-dev` → `fal-ai/flux-2/lora` (guidance default 2.5) — for LoRAs
    trained against `black-forest-labs/FLUX.2-dev` (arch `flux2`). TITLES holds
    a commercial license for Flux.2-dev, so dev LoRAs are shippable.
  Check the training config's `model.name_or_path` / `arch` and set `--base`
  accordingly — a base/endpoint mismatch loads the LoRA incorrectly. Note dev's
  ideal `--scale` is uncalibrated; the 1.4 default is just a sweep starting point.
- **Errors are logged per-image, not fatal.** A flaky inference job won't
  abort the whole batch; the failed job appears in `manifest.json.errors`
  and the rest of the batch completes.
- **LoRA key convention must match the endpoint, or it loads as a silent
  no-op.** ai-toolkit saves Klein/Flux.2 LoRAs in the **PEFT/diffusers**
  convention (`diffusion_model.*.lora_A.weight` / `.lora_B.weight`, no alpha),
  and `fal-ai/flux-2/klein/9b/base/lora` **expects PEFT**. A LoRA in the
  **kohya** convention (`lora_down`/`lora_up`/`alpha`) gets **silently ignored**
  by this endpoint — outputs come back looking like base Klein (style absent),
  which is easy to misread as "wrong scale / under-fired." It is NOT a scale
  problem: bumping `--scale` won't help a LoRA fal didn't load.
  - **Diagnostic:** if the artist says the raw file "works fine" but your run
    looks base-like, suspect a convention conversion broke the load — check the
    file's keys (`lora_A/lora_B` = PEFT = good for fal-Klein; `lora_down/lora_up`
    = kohya = won't load on fal-Klein).
  - **Merging for fal-Klein:** use `scripts/merge_loras.py` (now PEFT- and
    kohya-aware) and let it output PEFT — it defaults to `--out-convention
    match_a`, so put the PEFT LoRA first as `--lora_a`. Do NOT hand-convert a
    Klein LoRA to kohya for fal. (The krea2-turbo endpoint is different — it
    takes the kohya `loras:[{path,scale}]` schema; convention rules are
    per-endpoint.)

## When NOT to use this skill

- The user wants to PICK a checkpoint from existing training samples → use
  `ai-toolkit-sample-reviewer` instead (works from the local sample images
  the trainer already produced).
- The user wants to TRAIN a new LoRA → `ai-toolkit-remote-launch`.
- The model is neither a Klein 9B base LoRA nor a Flux.2-dev LoRA (e.g. Wan22
  video, Flux.1 dev, Z-Image Turbo). The script supports `--base klein-9b` and
  `--base flux2-dev` only; other models need their own fal endpoints added to
  the `BASES` registry in `scripts/fal/inference.py`.

## Related skills

- `ai-toolkit-remote-launch` / `monitor` / `teardown` — training pipeline
- `ai-toolkit-sample-reviewer` — pick a checkpoint from training samples
- `flux2-klein-prompter` — write prompts that play well with Klein
