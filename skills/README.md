# Claude Code skills for the AI Toolkit training workflow

These are [Claude Code](https://claude.com/claude-code) skills that drive the
end-to-end LoRA training workflow in this repo — from config generation
through captioning, remote-GPU training on RunPod, and checkpoint review.
Each `<name>/SKILL.md` is a self-contained skill; some carry a `references/`
dir with deeper material.

## Activating them — nothing to do

Claude Code discovers skills from `.claude/skills/`, not from this tracked
`skills/` dir. Those entries are **relative symlinks back into `skills/`, and
they are committed**, so a fresh clone has every skill active immediately:
open Claude Code in the repo and they trigger by name or by their
`description` triggers. Everything else under `.claude/` (local settings,
worktrees) stays gitignored.

Two consequences worth knowing:

- **Author in `skills/<name>/`, never in `.claude/skills/<name>/`.** The
  latter is a link to the former; a real directory placed there is invisible
  to git and drifts out of the repo.
- **A new skill needs its link committed too**, or it ships to a clone
  inactive:

  ```bash
  ln -s "../../skills/<name>" ".claude/skills/<name>"
  git add ".claude/skills/<name>"
  ```

  A newly added link is picked up at the **next** Claude Code start — the
  skill registry is scanned once at launch.

On Windows, `git clone` only materializes symlinks with `core.symlinks=true`
(and Developer Mode); otherwise copy `skills/*` into `.claude/skills/`
instead.

## The workflow

| Stage | Skill | Use when |
|---|---|---|
| **Orchestrator** | `ai-toolkit-train` | "walk me through training" — guides the full lifecycle, invoking every skill below at the right step with go/no-go gates between stages. Carries its own dataset-readiness check (Stage 0.5) and a first-timer gate protocol (cost/time expectations, plain-language gates) in its `references/` |
| Model brief | `ai-toolkit-model-brief` | "what should my model do" — brainstorms requirements (text rendering, edit vs generate, fidelity↔flexibility) into `briefs/<project>-brief.md` before any model/config choice |
| Config | `ai-toolkit-lora-config` | "train a LoRA on X" — generates the training YAML (reads the brief when one exists) |
| Captioning | `ai-toolkit-gemini-captioner` | generate per-dataset Gemini captions |
| Caption QA | `style-vs-content-caption-auditor` | audit captions for leakage before training |
| Dataset triage | `ai-toolkit-dataset-diagnostics` | "no images found", crashes before step 0, stale cache |
| DOP tuning | `dop-class-advisor`* | pick `diff_output_preservation_class` |
| **Remote launch** | `ai-toolkit-remote-launch` | "train this on RunPod" — preflight + provision + sync + launch (incl. `--gpus N` multi-GPU) |
| **Remote monitor** | `ai-toolkit-remote-monitor` | "check on my run" — watch loop, pull, drive review |
| **Remote teardown** | `ai-toolkit-remote-teardown` | "tear it down" / "is anything still billing" |
| Review | `ai-toolkit-sample-reviewer` | "review my samples / pick a checkpoint" |

\* `dop-class-advisor` is a parameter-specific helper; the rest form the
universal path. Also bundled: `video-lora-dataset-prep` (clip prep and frame
math for video LoRAs — Wan2.2, MiniMax-H3), `ai-toolkit-fal-inference` (deploy
validation on fal's hosted endpoints), `synthetic-control-pair-qa`, and
`style-lora-content-uniformity-caption-inversion`. Model-specific *prompting*
skills (e.g. `flux2-klein-prompter`) are not bundled — they are not part of the
training path; `ai-toolkit-lora-config` covers config generation for all models
in this set.

The three **remote** skills wrap `scripts/remote/cli.py` (the hosted-GPU
pipeline — see `scripts/remote/README.md`). The remaining skills are
model-training methodology and run locally regardless of where training
executes.

## Typical end-to-end (RunPod)

The easiest entry point is the orchestrator — it runs the whole sequence
below with a checkpoint between each stage:

```
ai-toolkit-train                  # "walk me through training" — conducts all of the below
```

Or drive the stages yourself:

```
ai-toolkit-model-brief            # brainstorm what the model must do -> briefs/<project>-brief.md
ai-toolkit-lora-config            # generate config from reference images (+ the brief)
ai-toolkit-gemini-captioner       # caption the dataset
style-vs-content-caption-auditor  # audit captions
ai-toolkit-remote-launch          # preflight -> up (provision/sync/launch)
ai-toolkit-remote-monitor         # watch --once --json loop + sample review
ai-toolkit-sample-reviewer        # pick the checkpoint
ai-toolkit-remote-teardown        # down / rescue / confirm nothing billing
```

Requires `RUNPOD_API_KEY`, `RUNPOD_STOP_API_KEY`, and `HF_TOKEN` in `.env`
(see `scripts/remote/README.md` §1).
