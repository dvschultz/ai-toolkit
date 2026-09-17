# Video LoRA prep troubleshooting

When something is broken, this is the diagnostic decision tree.

## Symptom: "0 videos found" or training fails immediately

### Cause: `num_frames > 1` but path points at images

In AI Toolkit, the data loader switches to video mode when `num_frames > 1`
in the dataset config. If your dataset path contains image files
(`.jpg`/`.png`) and you've set `num_frames: 81`, the loader looks for
videos and finds 0.

**Check:** `ls /your/dataset/path/ | head` — are these images or videos?

**Fix:**
- For video LoRA: replace images with video files, OR ensure your dataset
  folder contains `.mp4`/`.mov`/etc.
- For image-from-video pipeline: extract frames first
  (`ffmpeg -i clip.mp4 -vf fps=1 frame_%04d.jpg`) and set `num_frames: 1`
  with `arch: wan22_14b` 24GB-style image config.

### Cause: macOS `._*` AppleDouble pollution

External Mac drives create `._<filename>` resource-fork files that look
like real files to many tools. The toolkit data loader and the captioners
in this repo all skip dotfiles. But some other tools may not.

**Check:** `ls -la /your/dataset/path/ | grep '\._'` — if you see `._*`
files, they're polluting the dataset.

**Fix:** delete them:
```bash
find /your/dataset/path/ -name '._*' -delete
find /your/dataset/path/ -name '.DS_Store' -delete
```

If you're loading the dataset from an external drive frequently, mount
the drive with `noappledouble` or copy to local SSD where macOS won't
add the resource forks.

### Cause: caption file extension mismatch

Toolkit looks for caption files by `caption_ext:` in YAML. Default is
`txt`. If your captions are saved with a different extension (or the
captioner wrote `.captions` or `.json`), the loader treats clips as
unconditional (no caption).

**Check:** `ls /your/dataset/path/morphing_creature1_chunk_001.*`

**Fix:** set `caption_ext` in YAML to match what's on disk, OR rename
caption files to match.

## Symptom: Captioner stalls at "0/N" with no progress

### Cause: slow external-drive IO

Captioners open each video with cv2, which is fast on local SSD (sub-second
per file) but can be 5–30s per file on slow external drives. With 100
clips, you can stall 30+ minutes before any caption writes.

**Diagnose:**
```bash
python scripts/diagnose_video_read.py --dataset /your/dataset/path/
```

Output flags slow reads with ⚠️. If most clips are slow, IO is the
bottleneck.

**Fix:** copy dataset to local SSD before captioning:
```bash
mkdir -p ~/Desktop/dataset-local
cp /your/external/drive/path/*.mp4 ~/Desktop/dataset-local/
python scripts/caption_..._gemini.py --dataset ~/Desktop/dataset-local/
```

You can copy back the `.txt` files when done if you want them on the
external drive next to the videos.

### Cause: corrupt or unsupported video file

cv2 hangs or fails silently on corrupt files. The captioner has retries
but a corrupt file can fail all 4 attempts and gum up the worker pool.

**Diagnose:**
```bash
python scripts/diagnose_video_read.py --dataset /your/dataset/path/
```

Look for "CANNOT OPEN" or "NO FRAME" in the output. The script lists
each failed file and exits with code 2 if any fail.

**Fix:** delete or re-encode the bad files:
```bash
# Delete
rm /your/dataset/path/broken.mp4

# Or re-encode
ffmpeg -i /your/dataset/path/broken.mp4 \
  -c:v libx264 -crf 18 -c:a aac -y \
  /your/dataset/path/broken_fixed.mp4
mv /your/dataset/path/broken_fixed.mp4 /your/dataset/path/broken.mp4
```

If many files are corrupt, the source files themselves may be — re-export
from the original NLE / capture source.

### Cause: Gemini API rate-limiting / quota

Captioning hangs partway through with 429 errors in the retry layer.

**Check:** look for tqdm progress + occasional retry messages. If progress
is stuck and retries are exhausting, you've hit a quota.

**Fix:**
1. Reduce `--workers` from 4 to 2 (less concurrent load on the API).
2. Use a different model: `--model gemini-3-flash-preview` (cheaper /
   higher quota) or `gemini-2.5-pro`.
3. Wait for quota reset (typically per-minute or per-day).
4. Switch to a different API key if you have multiple.

## Symptom: Training crashes immediately with shape mismatch

### Cause: `num_frames` doesn't satisfy the model's stride rule

For Wan 2.2: `(num_frames - 1) % 4 == 0` is required.

**Check:** in your YAML, what's `num_frames`?

**Fix:** snap to nearest valid value. 81 is the standard; 77, 85 also
valid. 80 is NOT valid. See `frame-math-cheatsheet.md`.

### Cause: dataset has clips of inconsistent length

If some chunks are 5s and others are 4.2s (the ffmpeg segment muxer's
rounding), the data loader's batched temporal sampling can shape-mismatch
within a batch.

**Diagnose:**
```bash
for f in /your/dataset/path/*.mp4; do
  ffprobe -v error -show_entries format=duration \
    -of default=nw=1:nk=1 "$f"
done | sort -u
```

If you see a wide range of durations, chunks are uneven.

**Fix:** rerun split_videos_to_chunks.py with `--drop-partial --overwrite`.

## Symptom: Training appears fine but trigger doesn't fire at inference

### Cause: motion described in captions

For motion LoRAs, every motion verb / time-evolution word / anticipatory
phrase in captions makes that motion concept *promptable* instead of
trigger-bound.

**Diagnose:**
```bash
grep -iE 'morph|melt|slump|transform|gradually|over time|about to' \
  /your/dataset/path/*.txt | head -20
```

If you see any of those words, captions are leaking motion vocabulary.

**Fix:** recaption with `motion-first-frame` mode from the
`ai-toolkit-gemini-captioner` skill. The motion-first-frame avoid list
covers all the common motion verb families and time-evolution words.

### Cause: temporal compression too aggressive

If chunks were 60s sources sampled to `num_frames: 81`, the LoRA learned
a 12× sped-up version of the motion. At inference, output looks rushed
or incoherent.

**Diagnose:** what was the source-chunk duration vs. inference target?
Compute compression ratio. If > 2.0× and motion is realistic, that's the
problem.

**Fix:** chunk sources to a length closer to inference target, or speed
up sources with ffmpeg's `setpts=PTS/N` filter before chunking.

### Cause: dataset compositionally uniform → trigger bound to composition

If every training clip has the same camera angle / framing / subject
position, the LoRA may bind composition to the trigger instead of the
motion you wanted.

**Diagnose:** look at first frames of 10 random training clips. Do they
look compositionally similar?

**Fix:** add caption variation describing composition explicitly so it
becomes a variable. Or add training clips with varied composition. This
is the same root cause as the Klein t2i-vs-edit failure mode (see
flux2-klein-lora-config / failure-mode-diagnosis.md §1).

## Symptom: Out-of-memory at training start

### Cause: `gradient_checkpointing` not on for video training

Even at 95GB VRAM, 81-frame video training without gradient checkpointing
exceeds available memory. It's not optional.

**Fix:**
```yaml
train:
  gradient_checkpointing: true
```

### Cause: not quantizing on smaller GPUs

Below 80GB, you need `quantize: true` and `quantize_te: true` with
`qtype: qfloat8`. Or drop to single-frame training (`num_frames: 1`)
on 24GB hardware.

**Fix:**
```yaml
model:
  quantize: true
  quantize_te: true
  qtype: qfloat8
```

Or use the 24GB single-frame fallback config as a template.

## Symptom: split_videos_to_chunks.py hangs on certain files

### Cause: `expr:gte(t,n_forced*N)` ffmpeg expression rare edge case

Some unusually-encoded source files fail the `-force_key_frames` insertion.
ffmpeg gets stuck or produces zero chunks.

**Diagnose:** the script prints `[error]` with stderr from ffmpeg. Look
for codec issues or invalid timebase warnings.

**Fix:** re-encode the source to a known-good format first:
```bash
ffmpeg -i weird_source.mov -c:v libx264 -crf 18 -c:a aac \
  -pix_fmt yuv420p -y normalized.mp4
```

Then chunk `normalized.mp4`.

## Symptom: Captioner runs successfully but Gemini output is empty / nonsense

### Cause: first-frame extraction failing

For motion-first-frame captioner, cv2 extracts the first frame. If the
first frame is black or corrupt, Gemini gets garbage and writes garbage.

**Diagnose:** look at one clip's first frame:
```bash
ffmpeg -i /your/clip.mp4 -frames:v 1 -y /tmp/firstframe.jpg
open /tmp/firstframe.jpg
```

Is the image valid? Black? Mostly-black with subject in last 5 frames?

**Fix:** if first frames are black, your source has a fade-in. Either:
- Trim the fade-in off the sources before chunking
- Modify the captioner to extract frame 5 or 10 instead of frame 0
  (search for `cap.read()` in the captioner script)

## Symptom: dataset works for T2V but I2V crashes

### Cause: `arch:` mismatch with model checkpoint

I2V needs `arch: wan22_14b_i2v` AND the I2V variant of the checkpoint
(`Wan2.2-I2V-A14B-...`). Mixing T2V arch with I2V model (or vice versa)
produces shape errors in the conditioning path.

**Fix:** verify both `arch:` and `name_or_path:` match. See
`train_lora_wan22_14b_yvonne_morph_style_i2v.yaml` for the canonical I2V
config.
