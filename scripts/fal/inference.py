"""Run LoRA inference on fal.ai's flux-2-family base LoRA endpoints.

Supports two bases via --base (default klein-9b for back-compat):
  * klein-9b  -> fal-ai/flux-2/klein/9b/base/lora   (guidance default 5.0)
  * flux2-dev -> fal-ai/flux-2/lora                 (FLUX.2 [dev], guidance default 2.5)
The LoRA must have been trained on the matching base. Both share the
`loras: [{path, scale}]` input schema, so the only per-base differences are the
endpoint id and the default guidance.

Built to do real-prompt A/B testing and strength sweeps of trained LoRA
checkpoints using fal's hosted base models — avoiding the cost + setup of a
custom inference pod for every checkpoint review pass.

Companion to `ai-toolkit-fal-inference` skill. The script is the workhorse;
the skill is the natural-language wrapper.

Capabilities:
  * One or more local .safetensors LoRA files, each given a short label.
  * Auto-upload to fal storage with a local URL cache (skip re-upload by hash).
  * Per-LoRA scale (default 1.4 — calibrated via 0.1-step strength sweep on
    Klein 9B base; see [fal strength sweep methodology] memory).
  * Per-call prompt list, optional per-call seed list (one image per seed per
    prompt per LoRA; defaults to one random seed if none specified).
  * Organized output: output/inference/<run>/<lora_label>/p<idx>_s<seed>.png
  * Manifest JSON of every request + response saved alongside results.
  * --cleanup to delete uploaded LoRAs from fal storage after the run.

Usage:

  # A/B three LoRAs against two prompts × three seeds (18 images)
  python scripts/fal/inference.py \\
      --run amon_v4_ab \\
      --lora v3_step_4250:output/amon_silex_klein_9b_v3/amon_silex_klein_9b_v3_000004250.safetensors \\
      --lora v4_step_1000:output/amon_silex_klein_9b_v4/amon_silex_klein_9b_v4_000001000.safetensors \\
      --lora v4_step_1250:output/amon_silex_klein_9b_v4/amon_silex_klein_9b_v4_000001250.safetensors \\
      --prompt "suburban houses below a smoking industrial complex under a blue sky, silex relief, 4m0nsx" \\
      --prompt "a herd of horses standing in a grassy field under a blue sky, silex relief, 4m0nsx" \\
      --seed 42 --seed 123 --seed 7

  # Single LoRA, one prompt, three random seeds
  python scripts/fal/inference.py \\
      --lora my_lora:path/to/lora.safetensors \\
      --prompt "a chrome key glyph, silex relief, 4m0nsx" \\
      --num-seeds 3

Required env: FAL_KEY in .env or environment. The script auto-loads .env.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# Auto-load .env so FAL_KEY is available
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

import fal_client
import requests


# Supported fal base-model LoRA endpoints. Pick with --base. Each carries its
# own fal-recommended defaults; --guidance/--steps/--image-size/--scale override.
# guidance differs by base: Klein wants ~5.0, Flux.2-dev's fal default is 2.5.
# scale default is the ai-toolkit->fal calibration bump (fal reads lower than
# ai-toolkit's 1.0; see [fal strength sweep methodology] memory) — but the
# RIGHT dev scale is exactly what a sweep determines, so treat 1.4 as a
# starting point on dev, not a calibrated default.
BASES = {
    "klein-9b": {
        "endpoint": "fal-ai/flux-2/klein/9b/base/lora",
        "guidance": 5.0, "steps": 28, "image_size": "landscape_4_3", "scale": 1.4,
        "supports_negative": True, "supports_guidance_steps": True,
        "label": "FLUX.2 Klein 9B base",
    },
    "krea2-turbo": {
        # Krea2 LoRAs train on Krea-2-Raw but DEPLOY here, on Krea-2-Turbo.
        # The turbo (distilled) schema has NO guidance_scale / num_inference_steps /
        # negative_prompt — those fields are omitted from the payload. scale for a
        # Raw-trained LoRA on Turbo is uncalibrated: sweep it (start ~1.0-1.4).
        # image_size default square_hd; pass --image-size landscape_4_3 to match Klein.
        "endpoint": "fal-ai/krea-2/turbo/lora",
        "guidance": None, "steps": None, "image_size": "square_hd", "scale": 1.25,
        "supports_negative": False, "supports_guidance_steps": False,
        "label": "Krea-2 Turbo (Raw-trained LoRA)",
    },
    "flux2-dev": {
        "endpoint": "fal-ai/flux-2/lora",   # FLUX.2 [dev] LoRA T2I (commercial license held)
        # scale 1.0 starting point: a dev strength sweep (arvida v2) landed the
        # sweet spot at ~1.0 — notably LOWER than Klein's 1.4 (dev over-drives
        # sooner: text-leak by ~1.25, shape collapse by ~1.5-2.0). Still
        # per-LoRA; sweep each model. guidance 2.5 is fal's dev default.
        # supports_negative=False: the fal-ai/flux-2/lora schema has NO
        # negative_prompt field — fal silently drops it. Text-leak control on
        # dev is scale<=1.0 only, NOT a negative.
        "guidance": 2.5, "steps": 28, "image_size": "landscape_4_3", "scale": 1.0,
        "supports_negative": False, "supports_guidance_steps": True,
        "label": "FLUX.2 [dev]",
    },
}
DEFAULT_BASE = "klein-9b"  # back-compat: existing invocations keep Klein behavior

# Back-compat module constants (help text / external importers). Per-base values
# are resolved from BASES at runtime once --base is known.
KLEIN_LORA_ENDPOINT = BASES["klein-9b"]["endpoint"]
DEFAULT_LORA_SCALE = BASES[DEFAULT_BASE]["scale"]
DEFAULT_GUIDANCE = BASES[DEFAULT_BASE]["guidance"]
DEFAULT_STEPS = BASES[DEFAULT_BASE]["steps"]
DEFAULT_IMAGE_SIZE = BASES[DEFAULT_BASE]["image_size"]
DEFAULT_NEG = ""

# Cache path: maps lora_hash -> {url, uploaded_at, label}. Re-uploads skipped
# if the local file's content hash matches a cache entry. Lives next to the
# script's parent so it survives runs and is shared across A/B sessions.
CACHE_PATH = Path(__file__).resolve().parent / ".fal_lora_cache.json"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text())
    except json.JSONDecodeError:
        return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


def upload_lora(local_path: Path, label: str, cache: dict) -> str:
    """Upload a LoRA to fal storage if not cached. Returns the fal CDN URL."""
    digest = sha256_of(local_path)
    entry = cache.get(digest)
    if entry and entry.get("url"):
        print(f"  [cache] {label}: {entry['url']} (hash {digest[:12]})")
        return entry["url"]
    print(f"  [upload] {label}: {local_path.name} ({local_path.stat().st_size // (1 << 20)} MB) ...")
    url = fal_client.upload_file(str(local_path))
    cache[digest] = {
        "url": url,
        "label": label,
        "filename": local_path.name,
        "uploaded_at": int(time.time()),
        "local_path": str(local_path),
    }
    save_cache(cache)
    print(f"  [uploaded] {url}")
    return url


def delete_lora(url: str) -> None:
    """Best-effort cleanup. fal-client doesn't expose a delete; the CDN URL
    persists. This is here as a placeholder for future hardening."""
    print(f"  [cleanup] fal-client has no delete API; URL persists: {url}")


def submit_one(*, endpoint: str, lora_url: str, lora_scale: float, prompt: str,
               negative: str, seed: int, image_size, guidance: float, steps: int,
               acceleration: str, enable_safety_checker: bool = True,
               supports_negative: bool = True,
               supports_guidance_steps: bool = True) -> dict:
    """Submit one inference job and return the result dict (fal response).

    Both supported endpoints (Klein 9B base, Flux.2-dev) are flux-2-family LoRA
    endpoints and share the `loras: [{path, scale}]` input schema.

    enable_safety_checker=False disables fal's NSFW filter, which otherwise
    returns a SOLID BLACK image (not an error) when it flags a render. LoRAs
    trained on datasets with any skin/body content can trip false positives on
    certain seeds even for innocuous prompts — disable it to stop legitimate
    renders being blanked.

    supports_negative=False omits negative_prompt entirely: the FLUX.2-dev
    endpoint (fal-ai/flux-2/lora) has NO negative_prompt field and silently
    drops it. Only send it to endpoints that actually support it (Klein).
    """
    payload = {
        "prompt": prompt,
        "image_size": image_size,
        "num_images": 1,
        "seed": seed,
        "acceleration": acceleration,
        "output_format": "png",
        "enable_safety_checker": enable_safety_checker,
        "loras": [{"path": lora_url, "scale": lora_scale}],
    }
    # Krea-2 Turbo's schema has NO guidance_scale / num_inference_steps — only
    # the flux-2-family endpoints take them. Sending them to turbo errors/ignores.
    if supports_guidance_steps:
        payload["guidance_scale"] = guidance
        payload["num_inference_steps"] = steps
    if supports_negative and negative:
        payload["negative_prompt"] = negative
    handler = fal_client.submit(endpoint, arguments=payload)
    return handler.get()


def download_image(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)


def parse_lora_arg(s: str) -> tuple[str, Path]:
    """`label:path` → (label, Path). If no colon, derives label from filename stem."""
    if ":" in s:
        label, path_str = s.split(":", 1)
    else:
        path_str = s
        label = Path(path_str).stem
    p = Path(path_str).expanduser()
    if not p.exists():
        raise SystemExit(f"LoRA not found: {p}")
    return label, p


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=None,
                    help="Run name for output folder. Default: timestamp.")
    ap.add_argument("--base", default=DEFAULT_BASE, choices=list(BASES.keys()),
                    help=f"fal base-model LoRA endpoint. Default '{DEFAULT_BASE}'. "
                         f"'flux2-dev' targets {BASES['flux2-dev']['endpoint']} "
                         f"(guidance default 2.5); 'klein-9b' targets "
                         f"{BASES['klein-9b']['endpoint']} (guidance default 5.0). "
                         f"The LoRA must have been trained on the matching base.")
    ap.add_argument("--lora", action="append", required=True,
                    help="LoRA in 'label:path' form. Repeat to A/B multiple. "
                         "Up to 3 per call (fal API limit per request).")
    ap.add_argument("--scale", type=float, default=None,
                    help="LoRA scale (strength). Applied to ALL --lora entries. "
                         "Default is the chosen --base's scale (1.4; calibrated higher "
                         "than ai-toolkit's 1.0 per fal-strength memory — on flux2-dev "
                         "this is just a starting point, the sweep finds the real value). "
                         "Ignored if --scale-sweep is given.")
    ap.add_argument("--scale-sweep", type=float, nargs="+", default=None,
                    metavar="SCALE",
                    help="Sweep multiple LoRA scales in one run. Pass space-separated values, "
                         "e.g. '--scale-sweep 0.75 1.0 1.25 1.5 1.75 2.0 2.5'. Generates one "
                         "image per (lora × prompt × seed × scale) combination. Output gets a "
                         "'_scale_<n>' suffix on the lora label so checkpoints don't collide.")
    ap.add_argument("--prompt", action="append", required=True,
                    help="Prompt text. Repeat for multiple prompts.")
    ap.add_argument("--negative-prompt", default=DEFAULT_NEG,
                    help="Negative prompt (default empty).")
    ap.add_argument("--seed", action="append", type=int, default=None,
                    help="Explicit seed value, repeatable. Applied to ALL --lora "
                         "entries (pinned-seed A/B comparison mode — same noise → "
                         "different LoRAs are directly comparable). If none given, "
                         "--num-seeds random seeds are used in the same pinned way. "
                         "For capability sampling (where each image gets its own "
                         "fresh seed), use --per-lora-num-seeds instead.")
    ap.add_argument("--num-seeds", type=int, default=1,
                    help="Number of random seeds when --seed is not given. "
                         "Seeds are SHARED across all LoRAs (pinned A/B mode). "
                         "Default 1. Mutually exclusive with --per-lora-num-seeds.")
    ap.add_argument("--per-lora-num-seeds", type=int, default=None,
                    metavar="N",
                    help="CAPABILITY-SAMPLING mode: each (LoRA × prompt) combo "
                         "gets N freshly-generated random seeds, NOT shared across "
                         "LoRAs. Use this when assessing what a model is CAPABLE "
                         "of producing (vs A/B-comparing LoRAs at identical noise). "
                         "Recommended for find-the-best-checkpoint tasks. Typical: "
                         "--per-lora-num-seeds 2 with 5 varied prompts → 10 images "
                         "per LoRA per the [fal capability-sampling methodology] memory.")
    ap.add_argument("--guidance", type=float, default=None,
                    help="guidance_scale. Default is the chosen --base's value "
                         "(klein-9b: 5.0, flux2-dev: 2.5).")
    ap.add_argument("--steps", type=int, default=None,
                    help="num_inference_steps. Default is the chosen --base's value (28).")
    ap.add_argument("--image-size", default=None,
                    help="image_size preset or WxH e.g. '1024x1024'. "
                         "Default is the chosen --base's value ('landscape_4_3').")
    ap.add_argument("--acceleration", default="regular",
                    choices=["none", "regular", "high"],
                    help="fal acceleration tier. Default 'regular'.")
    ap.add_argument("--disable-safety-checker", action="store_true",
                    help="Disable fal's NSFW safety checker. The checker returns a "
                         "SOLID BLACK image (not an error) when it flags a render; "
                         "LoRAs trained on data with any skin/body content trip false "
                         "positives on some seeds. Use this if you see black outputs.")
    ap.add_argument("--workers", type=int, default=4,
                    help="Concurrent inference requests.")
    ap.add_argument("--output-dir", default="output/inference",
                    help="Root output dir. Run goes under <root>/<run>/.")
    ap.add_argument("--cleanup", action="store_true",
                    help="Attempt to delete uploaded LoRAs after run. "
                         "(fal-client has no delete API as of v1.0.0; this is a no-op stub.)")
    args = ap.parse_args()

    if not os.environ.get("FAL_KEY"):
        print("FAL_KEY not set. Add it to .env or export it.", file=sys.stderr)
        sys.exit(1)

    # Resolve base-model endpoint + fill any unset knobs from the base's defaults
    # (so --guidance etc. override, else follow the chosen base).
    base = BASES[args.base]
    endpoint = base["endpoint"]
    if args.scale is None:
        args.scale = base["scale"]
    if args.guidance is None:
        args.guidance = base["guidance"]
    if args.steps is None:
        args.steps = base["steps"]
    if args.image_size is None:
        args.image_size = base["image_size"]
    print(f"Base: {args.base} ({base['label']})  |  endpoint: {endpoint}  |  "
          f"guidance: {args.guidance}  steps: {args.steps}")

    # Parse LoRAs
    loras: list[tuple[str, Path]] = [parse_lora_arg(s) for s in args.lora]
    if len(loras) == 0:
        print("Provide at least one --lora", file=sys.stderr)
        sys.exit(1)

    # Parse image_size: support "WxH" → object, otherwise pass through as preset
    def parse_image_size(s):
        if "x" in s.lower():
            w, h = s.lower().split("x", 1)
            return {"width": int(w), "height": int(h)}
        return s
    image_size = parse_image_size(args.image_size)

    # Run name + output dir
    run_name = args.run or f"fal_run_{int(time.time())}"
    out_root = Path(args.output_dir) / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    # Seed strategy:
    #   - --per-lora-num-seeds N → capability sampling: each (lora × prompt) gets
    #     N fresh seeds, NOT shared across LoRAs. Seeds generated inside the job-
    #     build loop below.
    #   - --seed (explicit) or --num-seeds → pinned A/B: seeds shared across all
    #     LoRAs (same noise → direct visual comparison).
    capability_mode = args.per_lora_num_seeds is not None
    if capability_mode and (args.seed or args.num_seeds != 1):
        print("Note: --per-lora-num-seeds overrides --seed and --num-seeds "
              "(capability-sampling mode ignores shared-seed flags).", file=sys.stderr)
    if capability_mode:
        seeds = None  # generated per-job below
    elif args.seed:
        seeds = args.seed
    else:
        seeds = [random.randint(0, 2**31 - 1) for _ in range(args.num_seeds)]

    # Upload all LoRAs (with cache)
    print(f"Uploading {len(loras)} LoRA(s) to fal storage:")
    cache = load_cache()
    lora_urls: dict[str, str] = {}  # label -> url
    for label, path in loras:
        lora_urls[label] = upload_lora(path, label, cache)
    print()

    # Build the full job matrix: every (lora × prompt × seed × scale) combination.
    # --scale-sweep overrides --scale and adds an extra axis; the lora label gets a
    # "_scale_<n>" suffix so per-scale outputs land in separate folders.
    scales = args.scale_sweep if args.scale_sweep else [args.scale]
    jobs = []
    for label, _ in loras:
        for scale in scales:
            effective_label = f"{label}_scale_{scale}" if len(scales) > 1 else label
            for p_idx, prompt in enumerate(args.prompt):
                # Per-LoRA-per-prompt fresh seeds in capability mode; shared seeds
                # in pinned-A/B mode.
                effective_seeds = (
                    [random.randint(0, 2**31 - 1) for _ in range(args.per_lora_num_seeds)]
                    if capability_mode else seeds
                )
                for seed in effective_seeds:
                    jobs.append({
                        "label": effective_label,
                        "lora_url": lora_urls[label],
                        "scale": scale,
                        "p_idx": p_idx,
                        "prompt": prompt,
                        "seed": seed,
                    })

    sweep_note = f"  |  scales: {scales}" if len(scales) > 1 else f"  |  scale: {args.scale}"
    seed_note = (
        f"  |  seeds: {args.per_lora_num_seeds}/lora/prompt (capability mode)"
        if capability_mode else f"  |  seeds: {len(seeds)} (pinned A/B mode)"
    )
    print(f"Run: {run_name}  |  loras: {len(loras)}  |  prompts: {len(args.prompt)}"
          f"{seed_note}  |  total images: {len(jobs)}{sweep_note}")
    print(f"Output: {out_root}")
    print()

    # Manifest skeleton — append per-job results as they land
    manifest = {
        "run": run_name,
        "base": args.base,
        "endpoint": endpoint,
        "scale": args.scale,
        "scale_sweep": scales if len(scales) > 1 else None,
        "guidance": args.guidance,
        "steps": args.steps,
        "image_size": image_size,
        "acceleration": args.acceleration,
        "negative_prompt": args.negative_prompt,
        "loras": [{"label": label, "local_path": str(path), "url": lora_urls[label]}
                  for label, path in loras],
        "prompts": list(args.prompt),
        "seed_mode": "capability" if capability_mode else "pinned_ab",
        "seeds": seeds,
        "per_lora_num_seeds": args.per_lora_num_seeds,
        "results": [],
    }

    # Run jobs in parallel
    results = []
    errors = []
    def run_one(job):
        try:
            resp = submit_one(
                endpoint=endpoint,
                lora_url=job["lora_url"], lora_scale=job["scale"],
                prompt=job["prompt"], negative=args.negative_prompt,
                seed=job["seed"], image_size=image_size,
                guidance=args.guidance, steps=args.steps,
                acceleration=args.acceleration,
                enable_safety_checker=not args.disable_safety_checker,
                supports_negative=base.get("supports_negative", True),
                supports_guidance_steps=base.get("supports_guidance_steps", True),
            )
            img_url = resp["images"][0]["url"]
            dest = out_root / job["label"] / f"p{job['p_idx']}_s{job['seed']}.png"
            download_image(img_url, dest)
            return {**job, "image_url": img_url, "local_path": str(dest),
                    "actual_seed": resp.get("seed"), "timings": resp.get("timings", {})}
        except Exception as e:
            return {**job, "error": f"{type(e).__name__}: {e}"}

    print(f"Submitting {len(jobs)} inference jobs (workers={args.workers}):")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, j): j for j in jobs}
        done = 0
        for fut in as_completed(futures):
            done += 1
            res = fut.result()
            label = res["label"]
            p_idx = res["p_idx"]
            seed = res["seed"]
            if "error" in res:
                errors.append(res)
                print(f"  [{done:>3}/{len(jobs)}] ✗ {label} p{p_idx} s{seed} sc{res['scale']}: {res['error']}")
            else:
                results.append(res)
                print(f"  [{done:>3}/{len(jobs)}] ✓ {label} p{p_idx} s{seed} sc{res['scale']} -> {res['local_path']}")

    manifest["results"] = results
    manifest["errors"] = errors
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    print()
    print(f"Done. {len(results)} image(s) saved, {len(errors)} error(s).")
    print(f"Manifest: {manifest_path}")

    if args.cleanup:
        print("\nCleanup:")
        for label, url in lora_urls.items():
            delete_lora(url)


if __name__ == "__main__":
    main()
