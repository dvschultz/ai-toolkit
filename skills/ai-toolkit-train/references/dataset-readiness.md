# Dataset readiness check (Stage 0.5)

A clean run on a bad dataset is the most expensive failure in this pipeline:
every downstream gate passes, the money gets spent, and the result is still
wrong. This check runs after Stage 0 (frame) and before Stage 1 (config).
It costs ~5 minutes and zero dollars.

This is **curation**, not diagnostics. `ai-toolkit-dataset-diagnostics` is
for mechanical failures (loader errors, AppleDouble pollution, bucket
crashes); this check is "are these images going to teach the model the
right thing?"

## How to run it

0. **Ask the artist which images ARE the style** — "point me at the 3–4
   images that most are the thing you want back." Do this FIRST, before you
   form your own opinion of the dataset. Those images become the reference
   set for step 4 and the anchor for the whole run. See "Why you cannot
   trust your own read" below — this step is not optional politeness, it is
   the correction for a known blind spot that has cost a full project.
1. **List the folder** — count files, note extensions, subfolders, anything
   that isn't an image (or video, for motion LoRAs).
2. **Look at 5–8 representative images** with the Read tool — spread across
   the folder, not the first 8 alphabetically.
3. **Walk the checklist below** and note any failures.
4. **Survey the make-or-break register's coverage** (checklist item 9) with
   `scripts/register_judge_gemini.py` and the artist's reference images.
5. **Reflect back what you see** (see "The reflect-back gate") and get the
   user's confirmation.
6. **Issue a verdict**: GO / FIX FIRST / STOP.

## Why you cannot trust your own read of the images

Your image-reading path summarizes composition, palette, subject and mood
well, and **flattens processing texture into a single adjective** —
"photorealistic", "sharp", "soft". Any style whose signature IS the
processing is therefore invisible to you: lo-fi/GAN smear, datamosh,
scanline streaking, halftone density, grain structure, print artifacts,
chroma noise, compression blocks.

This is not hypothetical. On decker-protocolized the ground truth was
written as "photographic, long-exposure with flares" from exactly this kind
of read. Two complete trainings (~$17 and several hours) optimized a target
that was missing the make-or-break, a reference-calibrated judge later
scored every checkpoint of both runs **0/3** on the register that actually
mattered, and only the artist pasting four of their own images surfaced it.

Two consequences, both mandatory:

- The ground-truth spec is written **from the artist's named reference
  images**, in their words where possible, with the processing named
  explicitly ("vertical pixel-streak curtains", not "stylized").
- Any verdict about texture — at this stage, at Stage 5, at Stage 6 — comes
  from the register judge scored against those references, never from your
  own description of a sample.

## The checklist

### 1. Count vs. goal

| Goal | Sweet spot | Workable floor | Notes |
|---|---|---|---|
| Style LoRA | 20–50 | ~15 | Below ~15, expect memorization; below 10, have the STOP conversation |
| Character LoRA | 20–40 | ~12 | Variety matters more than count past ~20 |
| Combined (char + style) | 20+ per set | — | Two folders, counted separately |
| Motion LoRA (video) | 15–40 clips | ~10 | See `video-lora-dataset-prep` for clip prep |

More than ~100 images is fine but means longer training and makes curation
*more* important, not less — at that size nobody has looked at every image,
and one bad cluster (watermarked exports, a second style) quietly steers
the model.

### 2. Watermarks, logos, URLs, signatures

**Non-negotiable.** Any recurring watermark, logo, or URL in the dataset
reproduces as garbled text in *every* output the model makes — this is a
known, repeated failure, not a maybe. Crop or inpaint them out before
anything else happens. Artist signatures in a consistent corner count.

Text *content* in the images (posters, lettering the artist made) is a
deliberate choice, not an automatic fail: keep it only if reproducing that
text treatment is part of the goal, and flag that it changes the captioning
strategy. If the user doesn't want text in their outputs, images that are
mostly text should come out.

**Ask before cropping anything that might be the style.** Annotation
marks, registration lines, borders, deliberate "flaws" — in one past
project the artist's little blue annotations WERE part of the style and
standard cleanup would have destroyed it. The rule: watermarks/logos/URLs
the artist didn't author go; marks the artist made get a one-line "keep or
crop?" before touching them.

### 3. One thing per folder

- **Style LoRA**: one aesthetic. The model averages whatever it sees — two
  styles in one folder train to a mushy blend of both. If the user's work
  has distinct periods/modes, pick one (or plan a multi-mode caption
  strategy deliberately — that's an advanced path, not a default).
- **Character LoRA**: one subject, consistently recognizable. Different
  haircuts/ages are fine if identity reads through; a second person in half
  the images is not.
- Outliers out. Three images that don't belong drag the whole run; deleting
  them is free.

### 4. Variety within consistency

The model learns what's *constant* as the trigger and what *varies* as
promptable. So the dataset needs variation in everything that should stay
controllable:

- **Style LoRA**: varied subjects and compositions. If every image is the
  same kind of subject in the same layout, the model bakes the subject and
  layout into the style — prompting anything else fights the LoRA.
- **Character LoRA**: varied pose, clothing, setting, lighting, framing —
  same person. If the character only ever appears in one outfit, the outfit
  becomes part of the character.
- **Edit LoRA**: varied framing on the control side (close-up *and*
  full-body) — edit LoRAs trained only on close frontal crops fail on
  everything else.

### 5. Duplicates and near-duplicates

Exact dupes and trivial variants (same shot, slightly different crop or
export) over-weight that one image. Remove them — keep the best version.

### 6. Quality floor

- Long side ideally ≥1024px. A few smaller images are survivable; a dataset
  that's mostly tiny thumbnails is not.
- Heavy JPEG artifacts, screenshot compression, and upscaler smear get
  *learned* — the model will reproduce them as part of the style.
- For style work, the texture in these files IS the product. Re-exporting
  from originals beats rescuing compressed copies.

### 7. Rights

One sentence, asked once: the images should be the user's own work or work
they have permission to train on. Don't interrogate; do surface it.

### 8. Mechanical hygiene

Quick scan, then route — don't fix by hand here:

- `._*` AppleDouble files / `.DS_Store` (external Mac drives always have
  them)
- Mixed images and videos in one folder
- Images hiding one subfolder deeper than the path the user gave
- Weird extensions (`.HEIC`, `.tiff`) the loader won't pick up

Anything found → run `ai-toolkit-dataset-diagnostics`' preflight checklist
before Stage 1.

### 9. Make-or-break register coverage (run the judge)

Identify the one ingredient that, if lost, makes the run a failure — the
palette, the texture, the mark-making, the print artifacts. Then **measure
how much of the dataset actually carries it** rather than assuming the
artist's description applies evenly:

```bash
source .venv-captioning/bin/activate
python scripts/register_judge_gemini.py \
    --refs <the artist's 3-4 reference images> \
    --register "<what it is, in concrete visual terms>" \
    --exclude "<what must not count: usually palette/lighting/subject>" \
    --out output/<run>/register_dataset.json \
    <dataset>/*.png
```

It prints a coverage percentage. Read it as a **config instruction**, not
as trivia:

| Coverage at >=2 | What it means | What the config must do |
|---|---|---|
| ~90-100% | Dominant register | Binds by omission; standard recipe works |
| ~60-90% | Majority register | Binds, but keep EMA modest and check late saves |
| **under ~60%** | **Minority register** | **Will NOT bind by omission.** EMA off, higher rank, native-resolution-only buckets, and either inverse-mark the clean plates in captions or train the heavy subset |

On decker-protocolized the split was 19 heavy / 19 mild / 4 clean — 45%.
That number, available for well under a dollar before any GPU spend,
predicted both failed runs exactly: EMA at 0.99 averaged a minority
register away, and a 768 bucket smeared what was left. The survey is the
cheapest config decision in the pipeline.

Also note which plates score 0. Those are the ones to inverse-mark (caption
the *exception*, e.g. `, clean and sharp`, leaving the register unmarked and
therefore the trigger's default) — see the captioner skill.

## The reflect-back gate (most important step for a first-timer)

After looking at the images, describe **in plain words, no jargon**:

1. What you believe the model is supposed to learn from these images (the
   thing that will be baked in, activated by the trigger word).
2. What will stay controllable at prompt time (subjects, settings, etc.).
3. **How the images are made or processed** — name the texture, the medium,
   the artifacts, the rendering, not just the palette and the subjects. If
   you cannot say this in concrete terms, you have not looked hard enough
   at the reference images, and the run will optimize the wrong thing.
4. The register coverage number from item 9, in plain words ("about half
   your images carry that heavily, so I'm going to configure the run to
   hold on to it").
5. Anything you saw that surprised you or doesn't fit.

Then ask: **"Is that the thing you want the model to learn?"**

A goal mismatch caught here costs one sentence. The same mismatch caught at
Stage 6 costs the whole run. First-time trainers often can't evaluate a
config, but they can absolutely evaluate "here's what I think your work is
about" — this is the gate where their expertise actually applies.

## Verdicts

- **GO** — checklist clean, reflect-back confirmed. Proceed to Stage 1.
- **FIX FIRST** — fixable issues: crop watermarks, delete outliers/dupes,
  re-export low-quality files, clean dotfiles. Name each fix concretely
  ("crop the logo out of these 6 files: ..."), help do it where possible,
  re-check, then GO. Usually same-session.
- **STOP** — not trainable as-is: too few images, no coherent single
  aesthetic/subject, or the goal and the dataset don't match. Say so
  plainly, say what *would* make it trainable ("15 more images of X",
  "split these two styles into two runs"), and don't soften it into a FIX.
  A STOP here is the cheapest outcome in the whole pipeline.
