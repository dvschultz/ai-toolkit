# Sample Prompt Design

Sample prompts during training are the only signal for whether the LoRA is actually learning. Bad sample prompts hide training failures; good sample prompts expose them.

## The golden rule

**Keep sample prompts minimal.** Do NOT include the style description in the sample prompt for a style LoRA. Do NOT include identity details in the sample prompt for a character LoRA.

Why: if the base model can render the thing from the prompt alone (e.g., you asked for "oil pastel illustration" so the base model gives you one), you can't tell if the LoRA learned anything. The LoRA must be forced to do the work.

### Wrong (for a Sharon Zheng style LoRA):
```
shnzng a red bicycle rendered as an oil pastel crayon illustration on white paper with heavy waxy pigment buildup and visible directional hatching strokes...
```

The base model renders that from the prompt. Training results are indistinguishable from baseline.

### Right:
```
shnzng a red bicycle
```

If the style LoRA is working, this produces the target style. If not, it produces whatever the base model defaults to (probably a photo).

## The second rule: minimal, but in the captions' dialect

Minimal does not mean free-form. Write sample prompts in the **same dialect the
captions use** — the trigger in the position the captions put it, the structural
clauses every caption carries, and roughly the caption's length. Strip the
*style description*; keep the *grammar*.

The two rules only look like they conflict. "Minimal" is about not handing the
base model the answer; "dialect" is about addressing the model in the language
it was conditioned in. A prompt can be short and still off-dialect, and an
off-dialect prompt under-fires the LoRA — which then gets misread as a weak
checkpoint, a bad scale, or a failed run.

Concretely, if captions are shaped:

```
[composition clause], [count] [1-2 colours] form(s) [mode], k1n3f0rm
```

then a sample prompt must put `k1n3f0rm` **last** after a comma, carry a
composition clause, use one motion word from the closed set the captions use,
and run about as long as a caption:

```yaml
# Right — minimal content, caption grammar
- "off-centre in the upper frame, a cluster of teal and amber forms roiling, k1n3f0rm"

# Wrong — trigger fronted, no composition clause; under-fires the LoRA
- "k1n3f0rm a cluster of teal and amber forms"
```

On pawlowski-kineform the sample prompts were written trigger-first while every
caption put the trigger last. The training samples were a weak signal for the
whole run, and the error carried into the first fal batches — three batches and
~$7 spent ranking checkpoints on prompts the model had never been addressed in.

**Derive the dialect from the captioner script, not from memory.** Read the
caption template (or five actual `.txt` files) before writing sample prompts,
and write the deploy prompts in the same dialect later — the deploy notes should
state the dialect explicitly so whoever ships the model doesn't re-invent it.

**A rigid dialect is also a warning.** If sample prompts can only fire the LoRA
inside one narrow phrasing, that constraint ships with the model. Decide at the
brief stage whether that's acceptable, and say so in the deploy notes.

## Templates by LoRA type

### Character LoRA sample prompts

Test multiple angles of what the LoRA should do:

```yaml
prompts:
  # 1-2 prompts matching the primary captioned style
  - "[trigger] portrait, studio photograph"
  - "[trigger] walking down a street, modern digital photograph"

  # 1-2 prompts for any secondary style if you trained with one
  - "[trigger] sitting on a couch, 2010s smartphone photograph"

  # 3-4 prompts for styles the LoRA NEVER saw — tests generalization
  - "[trigger] in an oil painting, renaissance style"
  - "[trigger] anime illustration, cherry blossoms in background"
  - "[trigger] pencil sketch on aged paper"
  - "[trigger] watercolor painting, sunlit cafe"

  # 1 control prompt with the trigger but no style descriptor
  - "[trigger] standing in a room"
```

What you're looking for:
- Trained-style prompts: character recognizable + style matches
- Generalization prompts: character recognizable + style from prompt applied correctly
- If character is unrecognizable in generalization prompts → LoRA baked in too much style
- If character is recognizable but all in the same style regardless of prompt → LoRA overfit

### Style LoRA sample prompts

Test that the style transfers to many subjects:

```yaml
prompts:
  # simple content prompts matching training distribution
  - "[trigger] a red bicycle"
  - "[trigger] a teapot with flowers painted on it"
  - "[trigger] a goldfish in a round bowl"
  - "[trigger] a mushroom with spots"
  - "[trigger] a coffee mug with steam rising"

  # one or two more complex prompts
  - "[trigger] a cat sleeping on a stack of books"
  - "[trigger] a bicycle leaning against a brick wall"

  # control prompts — no trigger, shows what base model does on its own
  - "a red bicycle"
  - "a teapot with flowers"
```

What you're looking for:
- Triggered prompts: style visible, subject correct
- Control prompts: plain base-model output (no trigger-bound style)
- If triggered and non-triggered look the same → LoRA isn't activating
- If style is inconsistent across triggered prompts → needs more training or rank bump

### Combined character + style sample prompts

You trained with two style descriptors. Test both activate correctly AND that the character generalizes beyond both:

```yaml
prompts:
  # target style (should match the style dataset)
  - "[trigger] sitting on a couch, 2010s smartphone photograph"
  - "[trigger] in a bathroom mirror selfie, 2010s smartphone photograph"

  # identity-set style (should match the identity dataset)
  - "[trigger] studio portrait, modern digital photograph"
  - "[trigger] outdoors in a park, modern digital photograph"

  # generalization — styles never captioned
  - "[trigger] in an oil painting, renaissance style"
  - "[trigger] anime illustration"
  - "[trigger] watercolor painting, in a cafe"
```

## Config details

```yaml
sample:
  sampler: "flowmatch"        # must match train.noise_scheduler
  sample_every: 250           # sample at each save for visibility into training
  width: 1024
  height: 1024
  prompts: [ ... ]
  neg: ""                     # Flux family doesn't use negative prompts
  seed: 42
  walk_seed: true             # vary seed per prompt for visual variety
  guidance_scale: 4           # 1 for turbo models
  sample_steps: 20            # 8 for turbo models
```

**`[trigger]` placeholder**: the toolkit replaces `[trigger]` with your trigger word automatically. You can use either `[trigger]` or the literal trigger word in the prompts.

**`walk_seed: true`**: each prompt uses a different seed. Without it, all prompts share the same seed which gives less visual variety.

**Turbo models**: always use `guidance_scale: 1` and `sample_steps: 8`. Using normal settings on a turbo model produces terrible samples even if the LoRA is fine.

## Evaluating samples during training

After each sample save, look for:

**Step 0 (baseline)**: what the base model produces without LoRA. This is your reference.

**Step 250-500**: character/style starting to emerge. Don't expect full fidelity yet.

**Step 750-1500**: character/style clearly recognizable. Style prompts activate properly.

**Step 1500-2500**: full fidelity. Check that generalization prompts still work — if they've started looking like the training set, you're overfitting.

**Step 2500+**: usually too far. Samples become rigid, stop responding to style prompts.

The sweet spot is usually somewhere in the 1500-2500 range. Use interval saves to pick manually rather than trusting the final step.
