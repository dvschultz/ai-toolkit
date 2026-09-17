---
name: style-lora-content-uniformity-caption-inversion
description: |
  Invert the standard style-LoRA captioning rule when the training dataset has a
  recurring content motif (a single shape, subject, or framing that dominates most
  images) but the user wants the LEARNED STYLE to apply to ANY content at inference.
  Use when: (1) building a style LoRA from a thematically-uniform dataset — e.g.
  300 Valentine's Day collages that are all heart-shaped, all portraits in a
  painter's series, all packaged-product photos in a brand's catalog, all skateboard
  decks in a graphics archive, (2) the user states they want the style/texture
  to be content-flexible ("I want to say 'dolphin' and get a dolphin in this
  style"), (3) you notice during dataset reconnaissance that the same shape/subject
  appears in >40% of images. The standard rule "describe content, omit style"
  causes shape contamination — the recurring shape becomes part of the trigger.
  Inverted rule: caption the recurring shape/subject EXPLICITLY (so shape becomes
  promptable, not baked) and omit only textures/palette/technique (so those bind
  to the trigger). Companion to ai-toolkit-lora-config and style-vs-content-caption-auditor.
author: Claude Code
version: 1.0.0
date: 2026-05-06
---

# Style LoRA Caption Inversion for Content-Uniform Datasets

## Problem

The standard style-LoRA captioning rule is "describe content, omit style — the LoRA
learns what you don't caption." This works when content varies image-to-image and
style is what's consistent.

It **fails** when the dataset has thematic content uniformity. Example: 300 paper
collages from a Valentine's Day project are nearly all heart-shaped. Apply the
standard rule (caption only style-neutral content like "a layered composition,
oil pastel illustration") and the LoRA bakes "heart-shaped" into the trigger.
At inference, `TRIGGER a dolphin` produces a heart with dolphin features.

The user wanted style flexibility — paper-cutout textures applied to any subject —
and got the opposite: shape lock-in.

## Trigger conditions

Activate this skill when ANY of these are true:

1. **Dataset reconnaissance reveals a recurring shape, subject, or framing in
   >40% of images.** Examples: hearts in a Valentine's project, faces in a
   portrait series, products in a catalog, square format in an Instagram archive.
2. **User explicitly states they want content flexibility** with phrases like
   "I want someone to say X and get X in this style", "the style should apply
   to anything", "more about the textures than the shapes".
3. **The recurring element isn't itself the style.** A dataset of all
   pastel-on-cream illustrations IS just a style — caption normally. A dataset
   of all heart-shaped pastel illustrations is content + style — invert.

## The inverted rule

| Standard style-LoRA rule | Inverted rule (uniform-content style LoRA) |
|---|---|
| Describe: content (what's in the image) | Describe: the recurring shape/subject/framing AND any other variable content |
| Omit: style, medium, palette, technique | Omit: textures, patterns, palette, paper/material, technique, aesthetic descriptors |
| End every caption with: style descriptor | End every caption with: style descriptor (unchanged) |

The principle ("LoRA learns what you don't caption") is identical. Only the
classification of what counts as "content" changes — the recurring shape is
demoted from "style attribute" to "controllable content variable" by being
explicitly described.

## Solution: pre-captioning planning steps

### Step 1: Survey the dataset for recurring elements

Before captioning, identify:
- Dominant shapes (heart, circle, square, silhouette of X)
- Dominant subjects (faces, products, objects, scenes)
- Dominant compositions (centered, grid, diagonal, mirrored)
- Dominant framings (close-up, full-body, top-down)

If any single element appears in >40% of images, it must be captioned.

### Step 2: Define what to describe vs. omit — the three-category test

The standard "describe content, omit style" framing is too coarse for mixed-media
style datasets. There are actually **three** categories of words in any caption,
not two:

| Category | Definition | Examples (collage dataset) | Treatment |
|---|---|---|---|
| **1. Technique words** | True of every image. Describes how the medium is made. | `paper`, `cut paper`, `collage`, `scanned`, `paperboard` | OMIT (binds to trigger) |
| **2. Style attribute words** | True of the LoRA's aesthetic. Describes how the style looks. | `damask`, `halftone`, `woodblock`, `lavender`, `vintage`, `muted`, `mid-century` | OMIT (binds to trigger) |
| **3. Compositional variation words** | Vary image-to-image. Describe how the composition differs. | `torn`, `frayed`, `clean-cut`, `overlapping`, `densely layered`, `single layer`, `fragments`, `dense`, `sparse`, `busy`, `minimal`, `irregular edges` | **DESCRIBE** (controllable variables) |

**The discriminator question**: *"Does this word describe HOW the medium is
made (always true) or HOW the composition varies between images (varies)?"* If
it varies, describe it.

**Why this matters**: putting category-3 words in the avoid list is the most
common subtle failure mode. Category 3 looks superficially like "construction
language" or "process language" — it's tempting to lump it with category 1.
But category 3 is exactly the variation the user wants to be promptable. When
it ends up on the avoid list, the LoRA averages all variation away and produces
"median" outputs: medium density, no layering, clean edges, single-pattern
subjects on clean backgrounds. The pattern vocabulary will look correct, but
the layouts will feel flat and uninspired.

**Compositional axes to apply the test to** (for collage / mixed-media / craft
style datasets — adapt for other domains):

- **Edge quality** of cutouts: clean-cut / torn / frayed / irregular / deckle
- **Layering depth**: single layer / two overlapping layers / densely layered / many overlapping fragments
- **Internal composition of dominant shapes**: one solid pattern / split diagonally / horizontal bands / fragmented from many small scraps
- **Density**: minimal with empty space / medium / dense and crowded / edge-to-edge
- **Composition energy**: orderly grid / off-kilter / chaotic / asymmetric

For other domains, adapt: a portrait painting LoRA's category-3 axes might be
brushstroke confidence (loose/tight), framing crop, lighting direction. A
photography LoRA's category-3 axes might be motion (still/blurred), focus
(sharp/soft), occlusion.

**Describe (becomes promptable):**
- The recurring shape with explicit shape language: `"a heart-shaped cutout"`,
  `"a circular cutout"`, `"a triangular composition"`
- Layout/composition: `"centered"`, `"two overlapping shapes"`, `"a 2x2 grid"`,
  `"a diagonal band"`
- Recognizable subjects within the image (a bird, a flower, a face — when
  visible)
- Number of major elements
- **All category-3 compositional variation words** (see table above)

**Omit (binds to trigger):**
- Category-1 technique words (paper, collage, scanned, the medium itself)
- Category-2 style attribute words (textures, patterns, palette, period,
  aesthetic descriptors)

### Symptom of category-3 words wrongly omitted

If you trained and the output shows:
- Pattern/texture vocabulary captured correctly
- Palette captured correctly
- BUT layouts are flat, centered, single-subject-on-clean-background, no layering, clean edges where the dataset has torn edges, single-pattern subjects where the dataset has multi-pattern subjects

→ Almost certainly category-3 words are on the avoid list. Move them to the
DESCRIBE list and recaption (no need to retrain hyperparameters — only captions
change).

### Step 3: Write a VLM system prompt with explicit ALWAYS/NEVER lists

Adapt the style-LoRA system prompt in
`.claude/skills/ai-toolkit-lora-config/references/captioning.md` with the
inverted classification. Two examples in the wild that demonstrate this:

- `extensions_built_in/sd_trainer/...` chemigram config: vein patterning is
  classified as STYLE (omitted) so it appears on non-leaf subjects at inference.
- This skill's parent task: heart-shape is classified as CONTENT (described) so
  it becomes a controllable variable.

### Step 4: Sample-prompt the inversion at training time

Sample prompts must include shapes the dataset doesn't contain to verify
generalization:

```
- prompt: "a dolphin-shaped cutout, [style descriptor], [trigger]"
- prompt: "a horse-shaped cutout, [style descriptor], [trigger]"
- prompt: "a heart-shaped cutout, [style descriptor], [trigger]"  # the dataset shape — should still work
```

If the dataset-shape prompt works but novel-shape prompts produce dataset-shape-
flavored outputs, captioning didn't suppress shape strongly enough — try
recaptioning with even more specific shape language (silhouettes, profile
descriptors, named animal anatomy).

## Verification

A successful inversion shows three signs at sample time:

1. The dataset shape (heart) renders correctly when prompted
2. Novel shapes (dolphin, horse, teapot) render WITH the trigger style but
   IN their actual silhouettes — not heart-flavored
3. The control prompt (no trigger, just style descriptor) produces generic
   base-model output, not the dataset look

If sign 2 fails (novel shapes come out heart-shaped), shape leakage occurred.
Re-caption with stronger shape language. If sign 3 fails (the descriptor alone
produces dataset look), the descriptor is too specific — pick something more
generic.

## Notes

- This inversion only changes the captioning side. The training config
  (rank, LR, steps, model) is unchanged from a normal style LoRA.
- The recurring-element threshold (~40%) is heuristic. If you're unsure,
  caption the recurring element anyway — describing something that turns out
  not to be uniform costs nothing; failing to describe a uniform element
  costs the entire training run.
- This pattern composes with the standard style-suffix trick. The descriptor
  suffix still ends every caption; the inversion only adds shape/composition
  language earlier in the caption.
- Companion skill `style-vs-content-caption-auditor` audits AFTER captioning;
  this skill plans BEFORE captioning. Use both: this skill to set the rule,
  the auditor to verify the rule was followed.

## Related skills

- `ai-toolkit-lora-config` — overall LoRA config generation; this skill
  modifies its captioning step when content uniformity is detected
- `style-vs-content-caption-auditor` — post-captioning leakage audit
- `ai-toolkit-gemini-captioner` — VLM captioning script that consumes the
  inverted system prompt
