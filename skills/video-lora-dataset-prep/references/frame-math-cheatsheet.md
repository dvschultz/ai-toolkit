# Frame math cheatsheet

The mechanics that decide whether your video LoRA trains at all (vs. fails
loudly at startup) and whether it learns coherent motion (vs. a confused
mess).

## How AI Toolkit samples frames from training clips

```
training_clip (N frames at fps_source)
        │
        │  evenly sample `num_frames` indices
        ▼
 num_frames frames passed to the model
        │
        │  model treats them as the temporal sequence
        ▼
 LoRA learns the temporal evolution it sees
```

**Key fact**: `num_frames` in the YAML is what the model sees. The source
clip's actual frame count is decoupled — Toolkit evenly samples
`num_frames` indices from `[0, source_frame_count)`.

This means a 5s real-time motion can be trained at the same `num_frames`
setting as a 60s real-time motion — but the LoRA learns those as the same
*temporal density*. The 60s clip looks 12× sped up to the model. That's
usually NOT what you want.

## MiniMax-H3 (text-to-video)

### The 17n+5 rule

```python
assert (num_frames - 5) % 17 == 0
```

**Valid `num_frames`**: 5, 22, 39, 56, 73, 90, …

H3's video VAE uses a 17-frame temporal stride plus a 5-frame pivot, so an
off-grid count fails at the first batch (the trainer's error names the grid
explicitly). `auto_frame_count` snaps dataset clips **down** to the grid, so a
clip that is a few frames short of the next step silently trains shorter than
you think — check what it snapped to rather than assuming your clip length.

**Deploy-side mismatch to plan for:** the fal endpoint's minimum duration is
**5 seconds**, and its durations are integer seconds. Training clips rarely land
exactly there (pawlowski trained 4.46s clips and deployed at 5s). That gap is
normal and not worth contorting the dataset for, but state the training clip
length in the deploy notes so nobody reads the difference as a bug.

**Canvas**: H3's native 4:3 is 1024×768 — train there and deploy with
`aspect_ratio 4:3`. Native resolutions are 480P and 768P only; 2K/4K upscale a
768P base, so evaluate at 768P.

See `ai-toolkit-fal-inference` → **Video (MiniMax-H3)** for the deploy side.

## Wan 2.2 14B (T2V and I2V)

### The (n−1) % 4 == 0 rule

```python
assert (num_frames - 1) % 4 == 0
```

**Valid `num_frames`**: 5, 9, 13, 17, …, 77, **81**, 85, 89, …

This is hard-coded in the Wan 2.2 diffusers pipeline. Wan22 has a 4-frame
temporal stride for VAE encoding plus an extra pivot frame, so total must
fit `4k+1`.

**Default**: `num_frames: 81` → 5 seconds @ 16fps.

If you set `num_frames: 80` or `num_frames: 60`, training fails at the
first batch with a shape mismatch. Stick to valid values.

### fps assumption

Wan 2.2 was trained on **16 fps** clips. AI Toolkit's data loader handles
the resampling internally — you don't need to pre-resample your source —
but the inference target is 16fps regardless. So 81 frames = 5 seconds
of inference output.

### Resolution buckets

Default: `resolution: [512, 720]` for 16:9 widescreen training. Inference
at `width: 832, height: 480` matches the dataset bucket convention.

If your source clips are square or 4:3, you may want different buckets,
but Wan 2.2's training distribution is heavily 16:9 — non-widescreen will
fight the prior.

### Memory math (rough, for 81 frames)

| Hardware | What works |
|----------|-----------|
| 24 GB | Single-frame mode only (`num_frames: 1`). Treats as image LoRA. See `train_lora_wan22_14b_24gb.yaml` |
| 48 GB | num_frames=21–41 with quantization, severe limitations |
| 80–95 GB | num_frames=81 unquantized — the default config target |
| 95+ GB | num_frames=81 unquantized, comfortable headroom |

`gradient_checkpointing: true` is **required** for any video training
regardless of VRAM. Even at 95GB. The activation memory at 81 frames
without checkpointing exceeds available VRAM.

## LTX-2

LTX-2 has different temporal compression than Wan22. Check
`extensions_built_in/diffusion_models/ltx2/ltx2.py` and recent configs
for the current `num_frames` rule. As of this skill's writing the LTX-2
extension is under active development (see uncommitted changes in
`extensions_built_in/diffusion_models/ltx2/ltx2.py`).

LTX-2 supports **audio**. The captioner config (`caption_videos_ltx2.yaml`)
has a `use_audio: false/true` toggle. Set true if audio matters to the
content (dialogue, music sync) and you want captions to reflect it.

## Source-length × inference-length sizing

How long should your training chunks be? Depends on the inference goal:

| Inference target | Recommended chunk length |
|-----------------|--------------------------|
| 5s clips (Wan22 default) | 5–10s chunks. 5s = no temporal compression. 10s = ~2× compression — works for morph/transformation motion but distorts realistic motion |
| 5s clips with motion much slower than 5s | 5s chunks, but **speedup the source first** to fit motion arc into 5s |
| 5s clips with motion much faster than 5s | 5s chunks. Even-sampling will smooth fast motion appropriately |
| Long-form (10s+ inference, future) | Chunk to match. Don't bet on temporal compression to bridge a 4× gap |

### Temporal compression ratio

```
ratio = source_chunk_length_seconds / inference_length_seconds
```

| Ratio | Behavior |
|-------|----------|
| 1.0× | LoRA sees natural-speed motion. Best for realistic motion. |
| 2.0× | LoRA sees 2× sped-up motion. Works for morph / transformation / abstract motion. |
| 4.0×+ | LoRA learns the motion as fast-forward. Output at inference will look unnaturally rushed. Avoid. |

The Yvonne morph dataset trains at 2× compression (10s chunks → 5s output)
and that works because the morph motion is abstract enough to tolerate
compression. A realistic-action dataset (e.g., a person walking) at 2×
would look unnaturally fast.

## Calculating num_frames from a target

Given a target output length:

```
num_frames = round(target_seconds × 16)   # for Wan 2.2

# then snap to nearest valid value:
num_frames = ((num_frames - 1) // 4) * 4 + 1
```

Examples:
- 3s @ 16fps → 48 frames → snap to 49
- 5s @ 16fps → 80 frames → snap to 81 (the default)
- 7s @ 16fps → 112 frames → snap to 113

For very long inference targets, memory becomes the constraint before
math does. 81 is the practical maximum on most hardware.

## I2V vs. T2V

| Variant | First-frame behavior | When to pick it |
|---------|---------------------|-----------------|
| **T2V** (`arch: wan22_14b`) | Generated from prompt, no conditioning image | "Any subject in this style/motion" — fully generative |
| **I2V** (`arch: wan22_14b_i2v`) | First frame supplied as `ctrl_img:` at inference | Style transfer onto a real-world image. Apply LoRA's motion+aesthetic to a user-supplied image |

Training for I2V is **automatic from the same dataset** — AI Toolkit's
data loader extracts the first frame of each training clip as the
conditioning input. You don't need a separate dataset. The yvonne dataset
trains both a T2V and an I2V LoRA from the same 47 chunks; the only
config difference is `arch:` and `name_or_path:`.

## Captions and trigger placement (recap from captioner skill)

For motion LoRAs, captions describe **subject + first-frame state only**.
No motion verbs, no time-evolution words, no anticipatory language. The
trigger word goes at the end (literal token in `.txt`, NOT `[trigger]`).

If captions describe motion, the motion becomes promptable instead of
trigger-bound, defeating the LoRA. The captioner skill's
`motion-first-frame` mode handles this discipline; use it.
