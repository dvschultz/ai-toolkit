#!/usr/bin/env python3
"""Run MiniMax-H3 t2v LoRAs on fal (minimax/h3/text-to-video/lora).

Separate from scripts/fal/inference.py on purpose: that script is built for
image endpoints (image_size / num_images / output_format), while the H3
video schema takes resolution / aspect_ratio / duration and has NO
guidance_scale, num_inference_steps or negative_prompt at all (H3 is
guidance-distilled).

Schema notes that matter (from the endpoint's OpenAPI):
  * loras: [{path, weight_name?, scale}], scale 0-4, default 1
  * resolution: 480P | 768P | 2K | 4K -- only 480P and 768P are NATIVE;
    2K/4K upscale a 768P base, so test at 768P
  * aspect_ratio: 21:9 16:9 4:3 1:1 3:4 9:16 -- use 4:3 to match a LoRA
    trained at 1024x768 (H3's own native 4:3 canvas)
  * duration: 5-15s, default 5. Training clips were 4.46s, so 5s is the
    closest the endpoint can generate.
  * prompt_expansion_mode: DEFAULTS TO "balanced", i.e. fal REWRITES the
    prompt before generation. Pass null (the default here) to disable it and
    see the LoRA's raw behaviour; the response's expanded_prompt tells you
    what was actually sent.
  * enable_safety_checker defaults TRUE and returns black video on a false
    positive -- disabled here by default.

Pricing per second: 480P $0.0625 | 768P $0.075 | 2K $0.1625 | 4K $0.20.
A 5s 768P clip is ~$0.375, and the script prints a cost estimate and asks
for confirmation unless --yes is passed.

Usage:
    export FAL_KEY=...
    python3 scripts/fal/h3_video_inference.py \
        --lora s1250:output/morph_h3_v2/morph_h3_v2_000001250.safetensors \
        --lora s1750:output/morph_h3_v2/morph_h3_v2_000001750.safetensors \
        --prompt "m0rphf0rm a single crimson form unfurling" \
        --out-dir output/fal_h3_compare
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import fal_client
import requests

ENDPOINT = "minimax/h3/text-to-video/lora"
PRICE_PER_SEC = {"480P": 0.0625, "768P": 0.075, "2K": 0.1625, "4K": 0.20}
CACHE = Path.home() / ".cache" / "aitk-fal" / "h3_lora_urls.json"


def load_cache():
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text())
        except Exception:
            return {}
    return {}


def save_cache(c):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(c, indent=2))


def file_hash(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def upload(path: Path, label: str, cache: dict) -> str:
    """Upload to fal storage, skipping re-upload of an identical file."""
    key = file_hash(path)
    if key in cache:
        print(f"  [cached] {label}: {cache[key]['url']}")
        return cache[key]["url"]
    mb = path.stat().st_size // (1 << 20)
    print(f"  [upload] {label}: {path.name} ({mb} MB) ...", flush=True)
    url = fal_client.upload_file(str(path))
    cache[key] = {"url": url, "label": label, "uploaded_at": int(time.time())}
    save_cache(cache)
    print(f"  [uploaded] {url}")
    return url


def submit(lora_url, scale, prompt, duration, resolution, aspect, expansion,
           seed, safety):
    payload = {
        "prompt": prompt,
        "loras": [{"path": lora_url, "scale": scale}],
        "duration": duration,
        "resolution": resolution,
        "aspect_ratio": aspect,
        "enable_safety_checker": safety,
        # null disables fal's server-side prompt rewriting
        "prompt_expansion_mode": expansion,
    }
    if seed is not None:
        payload["seed"] = seed
    return fal_client.submit(ENDPOINT, arguments=payload).get()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", action="append", required=True,
                    metavar="LABEL:PATH", help="repeatable; e.g. s1250:path/to.safetensors")
    ap.add_argument("--prompt", action="append", required=True, help="repeatable")
    ap.add_argument("--scale", type=float, default=1.0, help="LoRA scale 0-4 (default 1.0)")
    ap.add_argument("--duration", type=int, default=5, help="seconds, 5-15 (default 5)")
    ap.add_argument("--resolution", default="768P", choices=list(PRICE_PER_SEC))
    ap.add_argument("--aspect-ratio", default="4:3",
                    choices=["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"])
    ap.add_argument("--expansion", default=None,
                    help="prompt_expansion_mode: omit for null/disabled, or "
                         "fast|balanced|quality to test fal's rewriter")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--safety-checker", action="store_true",
                    help="leave fal's NSFW filter ON (off by default: it returns "
                         "BLACK video on a false positive)")
    ap.add_argument("--out-dir", default="output/fal_h3")
    ap.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    args = ap.parse_args()

    if not os.environ.get("FAL_KEY"):
        print("FAL_KEY not set", file=sys.stderr); sys.exit(1)

    loras = []
    for spec in args.lora:
        if ":" not in spec:
            print(f"--lora needs LABEL:PATH, got {spec!r}", file=sys.stderr); sys.exit(1)
        label, path = spec.split(":", 1)
        p = Path(path)
        if not p.is_file():
            print(f"missing LoRA file: {p}", file=sys.stderr); sys.exit(1)
        loras.append((label, p))

    n = len(loras) * len(args.prompt)
    per = PRICE_PER_SEC[args.resolution] * args.duration
    print(f"\n{n} video(s) = {len(loras)} lora(s) x {len(args.prompt)} prompt(s)")
    print(f"{args.duration}s @ {args.resolution} {args.aspect_ratio}, scale {args.scale}, "
          f"expansion={args.expansion or 'disabled'}")
    print(f"~${per:.3f} each  ->  ~${per * n:.2f} total")
    if not args.yes:
        if input("proceed? [y/N] ").strip().lower() not in ("y", "yes"):
            print("aborted"); return

    cache = load_cache()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    urls = {label: upload(p, label, cache) for label, p in loras}

    rows = []
    for label in urls:
        for i, prompt in enumerate(args.prompt):
            print(f"\n[{label} / prompt {i}] {prompt!r}")
            try:
                r = submit(urls[label], args.scale, prompt, args.duration,
                           args.resolution, args.aspect_ratio, args.expansion,
                           args.seed, args.safety_checker)
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}")
                rows.append((label, i, "FAILED", str(e)[:120])); continue
            vid = (r or {}).get("video") or {}
            url = vid.get("url")
            exp = (r or {}).get("expanded_prompt")
            if exp:
                print(f"  expanded_prompt: {exp[:160]}")
            if not url:
                print(f"  no video in response: {str(r)[:200]}")
                rows.append((label, i, "NO_VIDEO", str(r)[:120])); continue
            dest = out / f"{label}_p{i}.mp4"
            dest.write_bytes(requests.get(url, timeout=300).content)
            mb = dest.stat().st_size / (1 << 20)
            print(f"  saved {dest} ({mb:.1f} MB)")
            rows.append((label, i, "OK", str(dest)))

    print(f"\n{'label':10s} {'prompt':7s} {'status':9s} detail")
    for label, i, status, detail in rows:
        print(f"{label:10s} {i:<7d} {status:9s} {detail}")
    ok = sum(1 for r in rows if r[2] == "OK")
    # bill only successful renders: fal does not charge for a downstream
    # failure, and counting them overstated an earlier run by ~$0.75
    print(f"\n{ok}/{len(rows)} succeeded  ·  ~${per * ok:.2f} spent"
          + (f" ({len(rows)-ok} failed, not billed)" if ok != len(rows) else ""))


if __name__ == "__main__":
    main()
