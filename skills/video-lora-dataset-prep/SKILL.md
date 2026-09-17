---
name: video-lora-dataset-prep
description: >-
  End-to-end pipeline for preparing a video dataset for AI Toolkit LoRA training (Wan 2.2,
  LTX-2, etc.). Audits source clips, decides chunk vs. trim vs. speedup vs.
---

# Video LoRA dataset prep pipeline

The pipeline:

```
  1. AUDIT  ──→ 2. STRATEGY ──→ 3. SPLIT ──→ 4. VERIFY ──→ 5. CAPTION ──→ 6. CONFIG
  (ffprobe)    (decision     (split_       (diagnose_     (handoff to     (handoff
                tree)         videos_       video_         captioner       to wan22
                              to_chunks.py) read.py)       skill)          /ltx2 skill)
```

Each step is mechanical given the previous step's output. The hard part is
step 2 — picking the right strategy. Most "video LoRA is broken" failures
trace back to a wrong strategy decision here.

## Step 1 — Audit source clips

Before deciding anything, gather data:

```bash
# Per-clip duration + fps
for f in /path/to/sources/*.{mp4,mov}; do
  ffprobe -v error -select_streams v:0 \
    -show_entries stream=duration,r_frame_rate,width,height \
    -of csv=p=0 "$f"
done

# Or just count and total duration:
for f in /path/to/sources/*.{mp4,mov}; do
  ffprobe -v error -show_entries format=duration \
    -of default=nw=1:nk=1 "$f"
done | awk '{s+=$1} END {print s "s total, " NR " clips"}'
```

What to record per clip (or in aggregate if homogeneous):

| Property | Why it matters |
|----------|---------------|
| **Duration** | Drives chunk-vs-trim-vs-sample decision |
| **fps** | Combined with target inference fps tells you temporal compression ratio |
| **Resolution** | Drives bucket choice; affects training memory |
| **Aspect ratio** | Matches inference width/height (Wan22 default 832×480 = 16:9) |
| **Has audio?** | LTX-2 may use it; Wan-2.2 ignores it |
| **Motion arc** | Single complete motion (good) vs. ambient / loopable / multi-motion (different strategy) |

If clips are heterogeneous in any of these dimensions, **note it** — you may
need different strategies per group.

## Step 2 — Pick a preparation strategy

This is the hardest step and the one most likely to be wrong. Read
`references/prep-strategy-decision-tree.md` for the full version.

Quick reference:

| Source clips look like | Strategy |
|-----------------------|----------|
| Many short clips (≤ N×inference-length each) with one complete motion each | **Even-sample only** — point dataset at sources, set `num_frames` correctly |
| A few long clips (>30s) each containing many independent moments | **Chunk** with `split_videos_to_chunks.py --length N`, possibly `--drop-partial` |
| Clips have a clear "complete arc" + lead-in/lead-out you don't want | **Trim** with `ffmpeg -ss START -to END` per clip, then even-sample |
| Motion is too slow for the inference window (e.g., 30s real-time → 5s training window) | **Speedup** with `ffmpeg -filter:v "setpts=PTS/N"` to fit motion into chunk length, then chunk or trim |
| Mix of all of the above | Group by category, run the right strategy per group, combine into one folder afterward |

**The single most common mistake:** assuming a 60s real-time clip will work
fine with `num_frames=81` and even-sampling. It won't. The temporal
compression ratio (60s × 16fps = 960 source frames → 81 sampled frames =
~12× compression) is too aggressive for most motion. Either chunk to 5–10s
or speed up the source first.

## Step 3 — Run split_videos_to_chunks.py

When chunking is the right strategy. The script lives at
`scripts/split_videos_to_chunks.py` in the repo. Read
`references/split-videos-recipes.md` for the full set of invocations.

Defaults:
- `--length 5` (5-second chunks)
- Re-encodes with x264 + AAC + `-force_key_frames` for **frame-accurate cuts**
- Output naming: `<basename>_chunk_001.mp4`, `<basename>_chunk_002.mp4`, …
- Output location: `<input>/chunks_<N>s/` if `--input` is a directory

**DO NOT use `-c copy`** for video LoRA prep. It only cuts at existing
keyframes, producing chunks of uneven length. Even-sampling on uneven
chunks gives the LoRA inconsistent temporal context. The script's re-encode
approach is slow but correct.

Common flags:

| Flag | Use |
|------|-----|
| `--length 5` | 5-second chunks (default) — typical Wan22 5s training window |
| `--length 10` | 10-second chunks — when you want the LoRA to learn a slower temporal compression (e.g., yvonne morph dataset uses 10s chunks @ num_frames=81) |
| `--drop-partial` | Drop the final chunk if it's <length seconds. Use this — partial chunks corrupt training |
| `--overwrite` | Re-chunk after changing strategy. Without this, existing chunks are skipped |
| `--crf 18` | Quality (default 18, lower=better/larger). Don't go above 22 for training data |

The naming convention `<basename>_chunk_NNN.mp4` is **load-bearing**: it
enables `caption_yvonne_videos_from_filename.py`-style deterministic captioning
downstream. If you rename chunks, filename-template captioning breaks.

## Step 4 — Verify chunks

Before captioning (which can take hours), run readability diagnostics.

```bash
python scripts/diagnose_video_read.py --dataset /path/to/chunks_5s/
```

This script opens every clip with cv2, reads the first frame, and reports:
- **OK** — readable and fast
- **slow (>2s)** — usually external-drive IO bottleneck
- **CANNOT OPEN** — corrupt or unsupported codec
- **NO FRAME** — file structure issue, file is too short, or codec mismatch

**If many slow reads** → copy the dataset to local SSD before captioning.
External-drive IO bottlenecks compound over hundreds of clips.

**If any failed reads** → delete or re-encode those clips. They will hang
your captioning run at "0/N" with no progress. Re-encode with:

```bash
ffmpeg -i broken.mp4 -c:v libx264 -crf 18 -c:a aac -y fixed.mp4
```

Read `references/troubleshooting.md` for the full set of diagnostic recipes.

## Step 5 — Hand off to captioner

Pick the captioner mode based on what kind of LoRA this is. Route to
**ai-toolkit-gemini-captioner** with the right mode:

| Dataset goal | Captioner mode | Why |
|--------------|---------------|-----|
| Motion LoRA (binding a transformation/morph to the trigger) | **motion-first-frame** | Caption only the first frame as a static image. Avoids Gemini describing the motion you want to bind to the trigger |
| Subject + motion already known per source video | **filename-template** | Deterministic mapping from `<descriptor>_chunk_NNN.mp4` → caption text. No Gemini call. Use when you've already split a small set of source videos with known content |
| Style LoRA on video data (every clip shares an aesthetic) | **style** captioner adapted for video — typically caption the first frame, or use the LTX-2 video captioner for full-clip context | Styles can usually be captured from the first frame; full-video captioning only matters if temporal style elements differ |
| Generic content captioning with audio | LTX-2 video captioner (uses `config/examples/caption_videos_ltx2.yaml`) — supports `use_audio: true`, hybrid keyframe extraction | Use when motion isn't the trigger and you want rich content descriptions |

For **motion LoRAs specifically**: caption the FIRST FRAME ONLY. The first
frame shows the "before" state — the subject we want to vary at inference.
Captioning later frames or the whole video risks Gemini describing the
motion you want bound to the trigger.

## Step 6 — Hand off to config skill

Once captioned, route to the appropriate config skill:

| Model | Config skill | Notes |
|-------|--------------|-------|
| Wan 2.2 14B (T2V or I2V) | wan22-14b-lora-config (when built) — for now use `ai-toolkit-lora-config` with the wan22 examples as templates | num_frames=81, switch_boundary_every=10, train_high_noise+train_low_noise |
| LTX-2 | `ai-toolkit-lora-config` with LTX-2 templates | Different num_frames rule, audio support |
| Hybrid (frames extracted, train as image LoRA) | flux2-klein-lora-config or similar | When the motion isn't the goal — use first frames as image dataset |

## Critical model-specific rules (don't break these)

### Wan 2.2 14B

```python
assert (num_frames - 1) % 4 == 0
# Valid: 5, 9, 13, ..., 77, 81, 85
```

This is hard-coded in the diffusers pipeline. `num_frames=80` will fail
loudly. `num_frames=81` is the default (5s @ 16fps).

Other Wan22 invariants:
- Default training resolution: `[512, 720]` buckets, 16:9 ratio
- `gradient_checkpointing: true` is **required** regardless of VRAM —
  not optional, even at 95GB
- `cache_text_embeddings: true` saves VRAM during training
- `switch_boundary_every: 10` for MOE high/low-noise alternation
- `train_high_noise: true` AND `train_low_noise: true` (both transformers)

### LTX-2

- Audio toggle exists (`use_audio: true/false` in captioner)
- Different num_frames math (consult LTX-2-specific docs / configs)

## Naming and folder conventions

```
/dataset-root/
  /chunks_5s/                     # output of split_videos_to_chunks.py
    morphing_creature1_chunk_001.mp4
    morphing_creature1_chunk_001.txt   # captioner writes these here
    morphing_creature1_chunk_002.mp4
    morphing_creature1_chunk_002.txt
    ...
```

**Don't** put source videos and chunks in the same folder. The captioner
will pick up both, double-caption, and waste API calls. Either:
- Use a chunks-specific subdirectory (`chunks_5s/`)
- Move sources to a `sources/` sibling folder before captioning

**Don't** mix chunk lengths in one dataset folder. The LoRA assumes uniform
clip length for `num_frames` sampling. If you have a `chunks_5s` and
`chunks_10s` folder, train them as separate datasets or pick one.

## Common mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Used `-c copy` for chunking | Chunks of uneven length, weird training behavior | Use `split_videos_to_chunks.py` (re-encodes correctly) |
| Forgot `--drop-partial` | Final chunks shorter than --length, training data inconsistent | Re-run with `--overwrite --drop-partial` |
| `num_frames` not satisfying `(n-1)%4==0` for Wan22 | Training fails at startup with shape mismatch | Use 81 (default) or another valid value |
| Source + chunks in same folder | Captioner picks up both, double-cost | Move sources to sibling folder |
| Skipped `diagnose_video_read.py` | Captioner stalls at 0/N for hours | Always validate before captioning |
| Mac AppleDouble files (`._*`) in dataset | Captioner sees them as `.mp4`, fails | All scripts in this skill skip dotfiles by default — but verify |
| Trained motion LoRA with full-video captions | Trigger doesn't bind to motion, motion is promptable | Recaption with motion-first-frame mode |
| Too-aggressive temporal compression (60s source → num_frames=81 even-sample) | Training output looks like fast-forward, motion lacks coherence | Chunk first, then sample |
| Caption files have wrong extension | Toolkit can't find captions, treats as unconditional | Check `caption_ext` in YAML matches actual files (default is "txt") |

## When NOT to use this skill

- User has a single short video that doesn't need prep — just point the
  config at it directly.
- User already has captioned chunks and just needs the training config —
  go straight to wan22-14b-lora-config or `ai-toolkit-lora-config`.
- User wants to caption images (not video frames) — use
  `ai-toolkit-gemini-captioner` directly.
- User wants to extract still frames and train as an image LoRA — that's
  a different (simpler) workflow; just `ffmpeg -i clip.mp4 -vf fps=1
  frame_%04d.jpg` and treat as an image dataset.

## Reference files

- `references/frame-math-cheatsheet.md` — `num_frames` rules per model,
  temporal-compression math, source-length × inference-length sizing
- `references/prep-strategy-decision-tree.md` — full chunk vs. trim vs.
  speedup vs. even-sample decision tree with worked examples
- `references/split-videos-recipes.md` — invocation patterns for
  `split_videos_to_chunks.py` (5s, 10s, custom output, drop-partial,
  re-chunking, custom CRF)
- `references/troubleshooting.md` — diagnose_video_read.py interpretation,
  AppleDouble pollution, slow-IO mitigation, codec re-encoding for failed
  files, "0 videos found" debugging
