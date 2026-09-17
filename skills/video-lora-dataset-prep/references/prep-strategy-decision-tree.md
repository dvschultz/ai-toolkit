# Preparation strategy decision tree

The hardest decision in video LoRA prep. Get this wrong and the LoRA
either fails to learn or learns the wrong temporal pattern — and you
won't know until inference.

## The four strategies

| Strategy | What it does | When to use |
|----------|-------------|-------------|
| **Even-sample only** | Point training at sources unchanged; let `num_frames` evenly sample | Sources are already the right length; motion fills the clip |
| **Chunk** | Split long clips into N-second pieces with `split_videos_to_chunks.py` | Long clips contain many independent moments |
| **Trim** | Cut each clip to a target window with `ffmpeg -ss/-to` | Clip has lead-in/lead-out around a clear motion arc |
| **Speedup** | Re-encode with `setpts=PTS/N` to fit motion into target window | Motion is too slow for the inference window; can't lose information by chunking |

These compose. A typical pipeline is **trim** to remove lead-in/out, then
**speedup** if the trimmed motion is still too slow, then **chunk** if the
result is still longer than target.

## Decision tree

### Question 1: How long are the source clips?

```
              ┌── ≤ inference_length × 1.2 (clips are already ~target length)
              │       → EVEN-SAMPLE ONLY
              │
   duration ──┼── 1.5 × to 3 × inference_length
              │       → Q2: motion arc check
              │
              └── > 3 × inference_length (clips are much longer)
                      → Q3: motion-content check
```

### Question 2: Does the trimmed clip contain ONE complete motion arc?

For clips 1.5×–3× inference length:

- **Yes, one complete motion** + lead-in/out → **TRIM**. Use `ffmpeg -ss
  START -to END` to isolate the motion.
- **Yes, one complete motion** + no lead-in/out (whole clip is motion) →
  **EVEN-SAMPLE** with appropriate temporal compression (see frame-math
  cheatsheet for ratio limits).
- **Multiple distinct motions** in the clip → go to Q3, treat as long clip.

### Question 3: Long-clip content type?

For clips > 3× inference length:

```
                         ┌── ambient / continuous / no clear segments
                         │       → CHUNK with --length matching inference target
                         │
   long-clip-content ────┼── many independent moments (cuts, transitions)
                         │       → CHUNK with --length matching inference target,
                         │         visually inspect chunks afterward — drop bad cuts
                         │
                         └── one slow, complete motion that takes the whole clip
                                 → SPEEDUP first to fit into inference window
                                   (then chunk or trim if still long)
```

### Question 4: Realistic motion or abstract motion?

After picking a primary strategy, decide if temporal compression is
acceptable:

- **Realistic motion** (a person walking, water flowing, leaves blowing):
  cap temporal compression at 1.0× — chunks should match inference length
  exactly. 2× compression makes realistic motion look unnaturally fast.
- **Abstract / morph motion** (the yvonne dataset, transformations):
  tolerates 2× compression well. 10s chunks → 5s output is fine.
- **Cyclic motion** (loops, oscillations): compression doesn't matter
  much — the model learns a phase pattern. Either ratio is OK.

## Worked examples

### Example A — Yvonne morph dataset

Source: 13 long videos (60–120s each), each one is a slow morph
transformation.

- Q1: long clips (>3×)
- Q3: one slow morph each → speedup OR chunk
- Q4: abstract morph motion, 2× compression OK

**Strategy**: chunk to 10s with `split_videos_to_chunks.py --length 10`.
Train at `num_frames: 81` (5s @ 16fps inference). Achieves 2× compression
which works for morph motion.

Result: 47 chunks across 13 sources.

### Example B — Rock formation morph (gr4r0cks)

Source: ~20 short clips, each ~5s of rock-formation morph motion.

- Q1: clips already ~target length
- Strategy: **EVEN-SAMPLE ONLY**. Point training at sources, `num_frames: 81`.

No chunking needed. The motion-first-frame captioner runs on sources
directly.

### Example C — Hypothetical: 120s timelapse cloud video

Source: one 120s timelapse, slow continuous motion.

- Q1: very long
- Q3: continuous ambient motion, no segments
- Q4: cyclic-ish motion

**Strategy**: chunk to 5s with `--length 5`. Get 24 chunks. Drop the last
partial chunk with `--drop-partial`. Train at `num_frames: 81`.

The cloud motion at 24 different time slices = good diversity for the
LoRA. Each chunk is independent enough that the LoRA learns the *kind*
of motion, not memorizes a specific 5s window.

### Example D — Hypothetical: 60s slow-mo macro video of a flower opening

Source: one 60s slow-mo of a flower bloom, captured at 240fps slowed to
30fps for a 60s playback.

- Q1: long
- Q3: one slow complete motion (the bloom) → speedup
- Q4: realistic motion → don't allow 12× temporal compression

**Strategy**:
1. **Speedup** with `ffmpeg -i bloom.mp4 -filter:v "setpts=PTS/12" sped.mp4`
   to compress the 60s bloom to 5s.
2. EVEN-SAMPLE the sped-up clip. `num_frames: 81`.

Now the LoRA sees the bloom motion at natural-perceived speed (despite the
source being slow-mo), with no temporal compression at training time.

If you skipped step 1 and tried to even-sample 60s @ 16fps = 960 frames →
81, the model would see frames every ~12 source frames apart, missing the
smooth bloom progression. Bad LoRA.

## Speedup commands

```bash
# 2× speedup (compress motion to half-duration)
ffmpeg -i input.mp4 -filter:v "setpts=PTS/2" -c:v libx264 -crf 18 \
  -c:a aac -af "atempo=2" output_2x.mp4

# 4× speedup
ffmpeg -i input.mp4 -filter:v "setpts=PTS/4" -c:v libx264 -crf 18 \
  -c:a aac -af "atempo=2,atempo=2" output_4x.mp4
# (atempo max factor is 2.0; chain for higher factors)

# 12× speedup, video-only (drop audio entirely)
ffmpeg -i input.mp4 -filter:v "setpts=PTS/12" -c:v libx264 -crf 18 \
  -an output_12x.mp4
```

For LoRA training, audio doesn't matter (Wan22 ignores it; LTX-2 may use
it). Use `-an` to drop audio if you don't need it.

## Trim commands

```bash
# Trim from 00:05 to 00:10 (a 5s window starting at 5s mark)
ffmpeg -ss 00:00:05 -to 00:00:10 -i input.mp4 \
  -c:v libx264 -crf 18 -c:a aac trimmed.mp4

# Trim with frame accuracy: -ss after -i (slower but exact)
ffmpeg -i input.mp4 -ss 00:00:05 -to 00:00:10 \
  -c:v libx264 -crf 18 -c:a aac trimmed.mp4
```

The "double -ss" pattern (one before `-i`, one after) is faster but can
silence audio if you're not careful — see the
`ffmpeg-atrim-silenced-by-output-ss` skill if that bites you.

## Mixing strategies

If your source clips are heterogeneous, group them by category and run
the right strategy per group:

```
sources/
  ├── short_morph_clips/     → even-sample only
  ├── long_morph_clips/      → chunk to 10s
  └── slow_macro_clips/      → speedup 6×, then even-sample
```

Then **combine all outputs into one chunked-dataset folder** before
captioning:

```bash
mkdir -p combined_chunks
cp short_morph_clips/*.mp4 combined_chunks/
cp long_morph_clips/chunks_10s/*.mp4 combined_chunks/
cp slow_macro_clips/*_sped.mp4 combined_chunks/
```

Then run the captioner once across `combined_chunks/`. The LoRA trains on
the unified set without caring how each clip got there.

## Anti-patterns

- **Don't even-sample 60s+ source clips** for a 5s inference target. The
  LoRA learns frame-skipped motion. Either chunk or speed up first.
- **Don't mix chunk lengths** in one dataset. Every clip should give the
  LoRA the same temporal context.
- **Don't speed up realistic motion to fit 5s.** It looks unnatural at
  inference. If a realistic 30s motion can't fit in 5s, the inference
  target is wrong — train at a longer `num_frames` if your model supports
  it.
- **Don't drop the no-trigger baseline samples** in your training config
  because you "tested it last time." Bleed detection on video LoRAs
  matters as much as image LoRAs (see flux2-klein-lora-config sample-
  matrix-cookbook).
