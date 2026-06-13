"""Run LoRA inference on fal.ai's Klein 9B base LoRA endpoint.

Built to do real-prompt A/B testing of trained LoRA checkpoints (and v3-vs-v4
comparisons) using fal's hosted Klein 9B base model — avoiding the cost +
setup of a custom inference pod for every checkpoint review pass.

Companion to `ai-toolkit-fal-inference` skill. The script is the workhorse;
the skill is the natural-language wrapper.

Capabilities:
  * One or more local .safetensors LoRA files, each given a short label.
  * Auto-upload to fal storage with a local URL cache (skip re-upload by hash).
  * Per-LoRA scale (default 1.5 — calibrated higher than ai-toolkit's 1.0 per
    the [fal LoRA strength calibration] memory).
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


# fal endpoint for Klein 9B base + LoRA
KLEIN_LORA_ENDPOINT = "fal-ai/flux-2/klein/9b/base/lora"

# Defaults — accept fal defaults except scale (calibrated higher per memory)
DEFAULT_LORA_SCALE = 1.5
DEFAULT_GUIDANCE = 5.0
DEFAULT_STEPS = 28
DEFAULT_IMAGE_SIZE = "landscape_4_3"
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


def submit_one(*, lora_url: str, lora_scale: float, prompt: str, negative: str,
               seed: int, image_size, guidance: float, steps: int,
               acceleration: str) -> dict:
    """Submit one inference job and return the result dict (fal response)."""
    payload = {
        "prompt": prompt,
        "negative_prompt": negative,
        "guidance_scale": guidance,
        "num_inference_steps": steps,
        "image_size": image_size,
        "num_images": 1,
        "seed": seed,
        "acceleration": acceleration,
        "output_format": "png",
        "loras": [{"path": lora_url, "scale": lora_scale}],
    }
    handler = fal_client.submit(KLEIN_LORA_ENDPOINT, arguments=payload)
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
    ap.add_argument("--lora", action="append", required=True,
                    help="LoRA in 'label:path' form. Repeat to A/B multiple. "
                         "Up to 3 per call (fal API limit per request).")
    ap.add_argument("--scale", type=float, default=DEFAULT_LORA_SCALE,
                    help=f"LoRA scale (strength). Applied to ALL --lora entries "
                         f"unless --scale-per-lora is given. Default {DEFAULT_LORA_SCALE} "
                         f"(calibrated higher than ai-toolkit's 1.0; see fal-strength memory).")
    ap.add_argument("--prompt", action="append", required=True,
                    help="Prompt text. Repeat for multiple prompts.")
    ap.add_argument("--negative-prompt", default=DEFAULT_NEG,
                    help="Negative prompt (default empty).")
    ap.add_argument("--seed", action="append", type=int, default=None,
                    help="Seed value. Repeat for multiple. If none given, "
                         "--num-seeds random seeds are used.")
    ap.add_argument("--num-seeds", type=int, default=1,
                    help="Number of random seeds when --seed is not given. Default 1.")
    ap.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE,
                    help=f"guidance_scale (default {DEFAULT_GUIDANCE}).")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                    help=f"num_inference_steps (default {DEFAULT_STEPS}).")
    ap.add_argument("--image-size", default=DEFAULT_IMAGE_SIZE,
                    help="image_size preset or WxH e.g. '1024x1024'. "
                         f"Default '{DEFAULT_IMAGE_SIZE}'.")
    ap.add_argument("--acceleration", default="regular",
                    choices=["none", "regular", "high"],
                    help="fal acceleration tier. Default 'regular'.")
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

    # Seeds: explicit list, or sample --num-seeds random integers
    if args.seed:
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

    # Build the full job matrix: every (lora × prompt × seed) combination.
    jobs = []
    for label, _ in loras:
        for p_idx, prompt in enumerate(args.prompt):
            for seed in seeds:
                jobs.append({
                    "label": label,
                    "lora_url": lora_urls[label],
                    "p_idx": p_idx,
                    "prompt": prompt,
                    "seed": seed,
                })

    print(f"Run: {run_name}  |  loras: {len(loras)}  |  prompts: {len(args.prompt)}  |  "
          f"seeds: {len(seeds)}  |  total images: {len(jobs)}  |  scale: {args.scale}")
    print(f"Output: {out_root}")
    print()

    # Manifest skeleton — append per-job results as they land
    manifest = {
        "run": run_name,
        "endpoint": KLEIN_LORA_ENDPOINT,
        "scale": args.scale,
        "guidance": args.guidance,
        "steps": args.steps,
        "image_size": image_size,
        "acceleration": args.acceleration,
        "negative_prompt": args.negative_prompt,
        "loras": [{"label": label, "local_path": str(path), "url": lora_urls[label]}
                  for label, path in loras],
        "prompts": list(args.prompt),
        "seeds": seeds,
        "results": [],
    }

    # Run jobs in parallel
    results = []
    errors = []
    def run_one(job):
        try:
            resp = submit_one(
                lora_url=job["lora_url"], lora_scale=args.scale,
                prompt=job["prompt"], negative=args.negative_prompt,
                seed=job["seed"], image_size=image_size,
                guidance=args.guidance, steps=args.steps,
                acceleration=args.acceleration,
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
                print(f"  [{done:>3}/{len(jobs)}] ✗ {label} p{p_idx} s{seed}: {res['error']}")
            else:
                results.append(res)
                print(f"  [{done:>3}/{len(jobs)}] ✓ {label} p{p_idx} s{seed} -> {res['local_path']}")

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
