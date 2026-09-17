# Model brief template

Write the filled brief to `briefs/<project>-brief.md`. Target: half a
page. Keep the user's own words wherever possible — especially the dream
prompts (verbatim; they become sample prompts) and the dial sentence.

```markdown
# Model brief — <project name>

**Intent (one line):** <what this model is for, in the artist's words>
**Date:** <date> · **Dataset:** <path or "not collected yet"> · **Goal type:** <style / character / edit / motion / combined>

## Dream prompts (verbatim — these become sample prompts)

1. "<prompt>" → <what they expect back>
2. "<prompt>" → <...>
3. "<prompt>" → <...>

**Stress prompts** (the weird things users will actually type — also
become sample prompts): "<...>", "<...>"

## Make-or-break (the acceptance criterion — verbatim, artist's own words)

**Reference images (the artist's own pick — these ARE the style):** <3-4
paths. Every texture verdict in this project is scored against these, never
against a written description.>
**The one ingredient that, if lost, makes this a failure:** "<...>"
**How it is made/processed:** <name the texture, medium, artifacts,
rendering in concrete terms — "vertical pixel-streak curtains and datamosh
blocks", not "stylized". If this line is vague, the run will optimize the
wrong thing.>
**Feel words:** <2–3 plain adjectives, e.g. delicate · glowing · rough>
**Register coverage (Stage 0.5 judge):** <N/total at >=2 = XX%. Under ~60%
means it will NOT bind by omission: EMA off, higher rank, native-res
buckets, inverse-mark the clean plates. Name the clean plates here.>

## The dial

**Position: <1–5>** (1 = follow the prompt, stretch the style · 5 = stay
close to my work)
In their words: "<the sentence they said>"
**Off-switch:** <must stay clean without the trigger (stacking/merging) /
bleed is fine or welcome (always-on deployment)>

## Embedded vs promptable (ruling per recurring trait)

| Recurring trait (from the dataset look) | Ruling |
|---|---|
| <e.g. the multi-panel collage layout> | automatic (bind to trigger) |
| <e.g. mask color> | keep variety (describe per-image) |
| <e.g. the recurring character> | only when asked (name in captions) |
| Same prompt twice → | <same signature look / different variant each time> |

## Requirements

| Capability | MUST / NICE / DON'T CARE | Evidence |
|---|---|---|
| <e.g. text rendering> | MUST | dream prompt 2 (poster with band name) |
| <e.g. generalize to new subjects> | NICE | prompts 1, 3 |
| <e.g. image editing> | DON'T CARE | confirmed — generation only |

## Constraints

- **Commercial use:** yes / no / unsure-treat-as-yes
- **Inference destination:** <fal / titles.xyz / local ComfyUI / unknown>
- **Runtime knobs there:** <negative prompts? per-prompt edits or fixed
  assisted prompt? strength slider?> — mitigations must be designable for
  these, not assumed
- **Volume/speed at inference:** <normal / high-volume>
- **(Edit LoRAs) Preservation contract:** <overlay or transform; what must
  stay exactly as uploaded; what inputs it must handle (selfies only /
  full-body / anything)>

## Dataset reality check

- Supported: <which MUSTs the dataset can deliver>
- Gaps: <requirement → what's missing → resolution (collect more / demoted to NICE because ...)>
- (No dataset yet: collect → <list derived from the MUSTs>)

## Out of scope (decided, don't re-litigate)

- <things explicitly DON'T CARE, with one-word why>
```

Notes for the filler:

- Every MUST needs an **evidence** pointer (which dream prompt or answer
  it came from) and should map to at least one sample prompt at Stage 1.
- The "Out of scope" section is load-bearing: it's what stops later gates
  from re-opening settled questions.
- If a gap was resolved by demotion, record the *why* — v2 will read this
  file and should know the demotion was a dataset limit, not a preference.
