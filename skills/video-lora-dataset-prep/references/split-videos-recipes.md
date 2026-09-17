# split_videos_to_chunks.py recipes

The script lives at `scripts/split_videos_to_chunks.py` in the AI Toolkit
repo. This doc is the cookbook for how to invoke it for common tasks.

## What the script does

- Re-encodes with x264 + AAC (slow but **frame-accurate**)
- Uses `-force_key_frames` + segment muxer to cut at exact target times
- Names output `<basename>_chunk_001.mp4`, `<basename>_chunk_002.mp4`, …
- Default output dir: `<input>/chunks_<N>s/` for directory input, sibling
  `chunks_<N>s/` for single-file input
- Skips dotfiles (macOS `._*` AppleDouble, `.DS_Store`)
- Skips already-chunked videos unless `--overwrite`

## Why NOT use `-c copy` (the 30-second-faster way)

```bash
# DO NOT do this for training data
ffmpeg -i input.mp4 -c copy -f segment -segment_time 5 \
  -reset_timestamps 1 chunk_%03d.mp4
```

`-c copy` only cuts at existing keyframes. So your "5-second chunks"
become "5-to-12-second chunks depending on where the next keyframe was."
Even-sampling on uneven chunks gives the LoRA inconsistent temporal
context, which corrupts training.

The script's re-encode approach takes 5–30× longer but produces clips of
*exactly* the requested length. For 100 clips at 30s each, expect 30–90
minutes total. For LoRA training data that gets used for hundreds of
hours of training, the time is well spent.

## Recipe 1: default 5-second chunks from a directory

```bash
python scripts/split_videos_to_chunks.py \
    --input /path/to/sources/
```

- Output: `/path/to/sources/chunks_5s/`
- Defaults: 5s length, CRF 18, keep partial chunks

When to use: matches Wan22's default 5s @ 16fps inference window.
Temporal compression ratio = 1.0× (no compression). Best for realistic
motion.

## Recipe 2: 10-second chunks for morph-style training

```bash
python scripts/split_videos_to_chunks.py \
    --input /path/to/sources/ \
    --length 10 \
    --drop-partial
```

- Output: `/path/to/sources/chunks_10s/`
- Each chunk = 10 seconds exactly (partials dropped)

When to use: you want 2× temporal compression at training time (10s →
`num_frames: 81` = 5s output). This is the yvonne morph pattern.
Tolerated by abstract / morph / transformation motion. Don't use for
realistic motion.

## Recipe 3: custom output directory

```bash
python scripts/split_videos_to_chunks.py \
    --input /path/to/sources/ \
    --output /Volumes/local-ssd/training-chunks-5s/ \
    --length 5 \
    --drop-partial
```

When to use: source is on slow external drive, but you want chunks on
local SSD for fast captioning + training. Common pattern: cut sources
where they live, write chunks to local SSD.

## Recipe 4: re-chunk after changing strategy

```bash
python scripts/split_videos_to_chunks.py \
    --input /path/to/sources/ \
    --length 7 \
    --drop-partial \
    --overwrite
```

- `--overwrite` deletes existing chunks for each source before re-cutting
- Use this when you've decided 5s wasn't right and need to redo

When to use: you initially chunked at 5s, looked at samples / first
training run, decided you need 7s for the motion to be coherent. Re-run
with the new length and overwrite.

## Recipe 5: single-file chunking

```bash
python scripts/split_videos_to_chunks.py \
    --input /path/to/single_video.mp4 \
    --length 5 \
    --drop-partial
```

- Output: `/path/to/chunks_5s/single_video_chunk_001.mp4`, …
- Sibling directory to the input file

When to use: you have one long source and want it chunked. Useful for
testing the chunking pipeline before running on a directory.

## Recipe 6: high-quality chunks (CRF 14)

```bash
python scripts/split_videos_to_chunks.py \
    --input /path/to/sources/ \
    --length 5 \
    --drop-partial \
    --crf 14
```

- CRF 14 ≈ visually lossless, larger files (~50% larger than CRF 18)

When to use: source has fine textural detail (slow-motion, photogram-
adjacent video, anything texture-heavy) and you don't want compression
artifacts to leak into training. Default CRF 18 is fine for most data;
go to 14 only if you see visible compression artifacts in chunks.

Don't go above CRF 22 — block artifacts will appear in training data
and the LoRA may learn them.

## Recipe 7: chunk + immediate captioning pipeline

After chunking, run readability check + captioning:

```bash
# 1. Chunk
python scripts/split_videos_to_chunks.py \
    --input /path/to/sources/ \
    --length 5 \
    --drop-partial

# 2. Verify (catches IO and corruption issues before the long captioning run)
python scripts/diagnose_video_read.py \
    --dataset /path/to/sources/chunks_5s/

# 3. Caption (motion-first-frame mode for motion LoRAs — see
#    ai-toolkit-gemini-captioner skill for which mode to pick)
python scripts/caption_<NAME>_motion_first_frame_gemini.py \
    --dataset /path/to/sources/chunks_5s/
```

## Combining outputs from multiple strategies

```bash
# Source group A: short clips, even-sampled (no chunking)
mkdir -p combined-dataset/

# Just copy in the originals
cp /sources_a/*.mp4 combined-dataset/

# Source group B: long clips, chunked
python scripts/split_videos_to_chunks.py \
    --input /sources_b/ \
    --output /tmp/group_b_chunks/ \
    --length 5 --drop-partial
cp /tmp/group_b_chunks/*.mp4 combined-dataset/

# Source group C: speedup-then-even-sample
ffmpeg -i /sources_c/slow.mp4 -filter:v "setpts=PTS/8" \
    -c:v libx264 -crf 18 -an /tmp/sped/slow_8x.mp4
cp /tmp/sped/*.mp4 combined-dataset/

# Now caption combined-dataset/ as one
python scripts/caption_<NAME>_motion_first_frame_gemini.py \
    --dataset combined-dataset/
```

Result: one folder of training-ready chunks, one captioning pass, one
training config pointing at it.

## Edge cases the script handles

| Edge case | Behavior |
|-----------|----------|
| Source shorter than `--length` | Produces one short chunk, prints `[warn]` |
| Empty / corrupt source | Returns 0 chunks, prints `[error]`, continues to next |
| Output dir doesn't exist | Created with `mkdir -p` |
| Re-running without `--overwrite` | Skips sources whose chunks exist, prints `[skip]` |
| macOS dotfiles in source dir | Skipped via `.startswith('.')` check |
| Source dir contains nested chunks dirs | Skipped via the `chunks_` part filter |

## When this script isn't right

- **Single chunk needed**: just use `ffmpeg -ss/-to` directly (faster).
- **Need keyframe-at-cut behavior**: `-c copy` is fine if you can tolerate
  uneven chunks (e.g., for non-training use).
- **Need different audio handling**: this script always re-encodes audio
  to AAC 192k. Edit the script if you need WAV/FLAC or higher bitrate.
- **Need to re-encode with hardware acceleration** (NVENC, VideoToolbox):
  edit the script's ffmpeg command to use `-c:v h264_nvenc` etc.
- **Need to preserve original codec**: same — script always re-encodes to
  x264.
