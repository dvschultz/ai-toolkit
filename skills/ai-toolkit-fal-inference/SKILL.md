---
name: ai-toolkit-fal-inference
description: Run real-prompt inference on trained ai-toolkit LoRAs via fal.ai's hosted endpoints — FLUX.2 Klein 9B base (t2i and base-edit/restyle), Krea-2 Turbo, FLUX.2 [dev], Ideogram V4, and MiniMax-H3 text-to-video. Use whenever the user wants to test a trained LoRA against custom prompts WITHOUT provisioning a custom GPU pod — A/B comparing checkpoints, testing an artist's prompt against multiple versions, generating presentation samples, or validating a checkpoint against prompts that weren't in the training sample matrix. Triggers on phrases like "test this LoRA via fal", "run inference on fal", "generate with my v3 / v4 LoRA against this prompt", "A/B these checkpoints on fal", "send this LoRA to fal", "compare these LoRAs on the same prompt", or any request to make images from a local .safetensors against custom prompts. Distinct from `ai-toolkit-remote-launch` (which spins up a training pod) and from picking-from-existing-samples (which uses the training-config sample matrix). This skill is specifically for ad-hoc real-prompt inference on the fal hosted endpoint.
---

# AI Toolkit fal Inference

Driver for `scripts/fal/inference.py` (images) and
`scripts/fal/h3_video_inference.py` (video) — runs LoRA inference on fal.ai's
hosted endpoints so you can test trained checkpoints against real prompts
without spinning up your own GPU pod. Five image bases are registered plus one
video base; see the `--base` table under **Things to watch for**, and pick the
one matching how the LoRA was trained.

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
4. **Base** — which endpoint the LoRA was trained for (`--base`). Read it off
   the training config's `model.name_or_path` / `arch`; don't guess.
5. **Optional overrides** — LoRA `--scale` (per-base default; sweep it before
   trusting it), `--guidance`, `--steps`, `--image-size`. Default fal settings
   are usually fine.

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
| `--scale` | **per-base** (klein-9b 1.4) | Each base has its own default and each LoRA its own optimum — see the `--base` table. Klein 9B: calibrated via 0.1-step strength sweep on Klein 9B base (see [fal strength sweep methodology] memory). Below ~1.3 the LoRA fires without arrow/callout grammar; at 1.4 full V1-style annotation grammar (arrows + multi-element labels + iridescent variant) lands without compositional crowding. Above ~1.7 labels start clipping the frame. Override per-call when an aesthetic goal calls for it (1.0 for minimal LoRA, 1.5-1.6 for max annotation density). |
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

**Do this before ranking checkpoints, not after** — see the first bullet under
**Things to watch for**. Strength sweeps are inherently a PINNED-SEED operation — you want the same
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

- **Sweep strength BEFORE comparing checkpoints — always, on every base.**
  Checkpoint ranking at an uncalibrated scale measures the scale, not the
  checkpoint. On pawlowski-kineform three fal batches and ~$7 went into ranking
  checkpoints at scale 1.0; step 1250 "clearly beat" 1750 there, and the two
  came out roughly equal once scale (2.75) and prompt dialect were corrected.
  The order is: **(1) pick one mid-run checkpoint, (2) scale-sweep it with
  pinned seeds, (3) rank all checkpoints at the scale that sweep found.** This
  is not the same as Gate A in the sample-reviewer — that one is about *which
  endpoint* you judge on; this is about *which knob you calibrate first* once
  you're there. Both are cheap; both have reversed a verdict.

- **A distilled endpoint can strip a fine register at scale 1.0 — sweep
  before concluding it wasn't trained.** Krea-2-Turbo is the standing case:
  a decker-protocolized checkpoint that scored 2/3 on its artifact texture
  in training samples scored 0-1 on Turbo at 1.0, then came back at 1.6
  (smear on 5 of 6 prompts). Distillation regularizes away high-frequency
  processing texture first; palette and light survive it. So for any LoRA
  whose make-or-break is texture/grain/glitch, run 1.0 / 1.3 / 1.6 before
  saying the register is missing, expect the deploy scale to sit high, and
  check over-drive at that scale **by eye** — a judge reads banding as
  signal. A texture phrase in the assisted prompt can supply modes the LoRA
  carries weakly, and stacks with scale.
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
- **Pick the endpoint with `--base` to match how the LoRA was trained.** A
  base/endpoint mismatch loads the LoRA incorrectly or silently no-ops. Check
  the training config's `model.name_or_path` / `arch` and set `--base`. Five
  bases are registered in `BASES` (default `klein-9b` for back-compat):

  | `--base` | endpoint | default scale | notes |
  |---|---|---|---|
  | `klein-9b` | `fal-ai/flux-2/klein/9b/base/lora` | 1.4 | t2i for `FLUX.2-klein-base-9B` LoRAs. guidance 5.0, negatives supported. |
  | `klein-9b-base-edit` | `fal-ai/flux-2/klein/9b/base/edit/lora` | 1.4 | **image-to-image**; the deploy path for `ctrl_img` restyle LoRAs. Requires `--image`. A t2i endpoint cannot restyle — sa-mayer shipped here. |
  | `krea2-turbo` | `fal-ai/krea-2/turbo/lora` | 1.25 | Krea2 LoRAs **train on Raw, deploy on Turbo**. No guidance/steps/negative in the schema. Always Turbo-test before shipping. |
  | `flux2-dev` | `fal-ai/flux-2/lora` | 1.0 | `arch: flux2`. guidance 2.5, **no negative_prompt field** (fal drops it silently) — text-leak control is scale ≤1.0 only. TITLES holds a commercial license. |
  | `ideogram-v4` | `ideogram/v4/lora` | 1.0 | `arch: ideogram4`. No guidance/steps; `expansion_model` `"None"` tests raw LoRA behavior, `"Medium"`/`"Large"` runs Magic Prompt (the real deploy path). |

  Video LoRAs do NOT go through this script — see **Video (MiniMax-H3)** below.

- **The `--scale` default is per-base, not universal.** The 1.4 figure is the
  Klein 9B calibration; every other base has its own, and every *LoRA* has its
  own within that. Treat the table's number as a sweep starting point, never as
  a verdict-grade setting. Measured spreads so far: Klein 1.4, Flux.2-dev ~1.0
  (over-drives sooner — text-leak ~1.25, collapse 1.5–2.0), Krea2-Turbo 1.0–1.6
  depending on whether a fine texture register has to survive distillation,
  MiniMax-H3 **2.75** (1.0 is far too weak and reads as generic).

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

## Video (MiniMax-H3) — a separate script

Video LoRAs run through `scripts/fal/h3_video_inference.py`, not `inference.py`.
They are split on purpose: the image script speaks `image_size` / `num_images`,
while the H3 schema speaks `resolution` / `aspect_ratio` / `duration` and has
**no `guidance_scale`, `num_inference_steps` or `negative_prompt` at all** —
H3 is guidance-distilled. Every fix has to live in captions, checkpoint choice
or scale.

```bash
.venv-captioning/bin/python scripts/fal/h3_video_inference.py \
    --lora s1750:output/<run>/<run>_000001750.safetensors \
    --prompt "<composition clause>, <count> <1-2 colours> forms <mode>, <trigger>" \
    --scale 2.75 --resolution 768P --aspect-ratio 4:3 --duration 5 \
    --out-dir output/fal_h3_<run>
```

Settings that are not optional (each one cost a batch to learn):

| Flag / field | Use | Why |
|---|---|---|
| `--expansion` | leave unset (null) | Defaults to `balanced` **on the endpoint**, i.e. fal rewrites your prompt before generation. The response's `expanded_prompt` shows what was actually sent. |
| `--safety-checker` | leave OFF | Defaults true on the endpoint and returns a **black video** on a false positive. |
| `--resolution` | `768P` | Only 480P and 768P are native; 2K/4K just upscale a 768P base. |
| `--aspect-ratio` | match the training canvas | `4:3` for a 1024×768 LoRA — also H3's own native canvas. |
| `--duration` | 5 | Endpoint minimum; training clips are usually shorter (4.46s on pawlowski) and cannot be matched exactly. |
| `--scale` | **sweep it, expect a high number** | H3 sits far above the image bases: 1.0 reads as generic dark-and-glowy, the style arrives ~2.4–2.5, and 2.75 was the reliable choice across seeds. Range is 0–4. |

Cost is per second of output: 480P $0.0625 · 768P $0.075 · 2K $0.1625 ·
4K $0.20. A 5s 768P clip is ~$0.375 — ~25× an image, so the "just run 10 of
them" reflex from the image path is a $4 decision here. The script prints an
estimate and asks for confirmation unless `--yes` is passed.

**Trigger-is-not-a-switch check.** Before writing deploy notes for a video LoRA,
run the trigger alone and a no-trigger control. On pawlowski the trigger alone
produced nothing and the no-trigger control still carried the full style — the
style was bound to the model, not the token. That is harmless for "load the LoRA
when you want the look" (and it makes the model immune to fal's prompt
rewriting), but it means the look cannot be toggled per-prompt, and the fix is a
**regularization dataset** of off-style clips captioned without the trigger —
*not* caption dropout, which makes the style more unconditional, not less.

## When NOT to use this skill

- The user wants to PICK a checkpoint from existing training samples → use
  `ai-toolkit-sample-reviewer` instead (works from the local sample images
  the trainer already produced).
- The user wants to TRAIN a new LoRA → `ai-toolkit-remote-launch`.
- The model is a **video** LoRA — see the MiniMax-H3 section below; Wan2.2
  has no wired endpoint yet.
- The base isn't in the `BASES` table above (e.g. Flux.1 dev, Z-Image Turbo).
  Add it to the `BASES` registry in `scripts/fal/inference.py` first — a
  one-dict change — rather than approximating with a neighbouring endpoint.

## Related skills

- `ai-toolkit-remote-launch` / `monitor` / `teardown` — training pipeline
- `ai-toolkit-sample-reviewer` — pick a checkpoint from training samples
- `flux2-klein-prompter` — write prompts that play well with Klein
- `video-lora-dataset-prep` — clip prep and frame math for video LoRAs (H3, Wan)
