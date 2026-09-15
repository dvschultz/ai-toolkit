"""Score images 0-3 on a make-or-break style register, calibrated on the artist's own reference images.

WHY THIS EXISTS
---------------
Claude's image-reading path summarizes composition, palette and subject, and
flattens *processing texture* into words like "photorealistic" or "sharp". A
style whose signature IS the processing (lo-fi GAN smear, datamosh, halftone
density, grain structure, print artifacts) is therefore invisible to it — the
decker-protocolized run shipped two complete trainings (~$17) against a ground
truth that had the palette right and the texture missing entirely.

This script takes that judgment out of the proxy's hands: the artist's own
reference images define the 0-3 scale, and Gemini scores every other image
against them.

USE IT TWICE
------------
1. Stage 0.5, over the DATASET — what fraction of plates actually carry the
   register? Coverage under ~60% means it will not bind by omission; the run
   needs EMA off / higher rank / inverse-marked clean plates, or a curated
   subset. Costs well under a dollar and redirects the whole config.
2. Stage 5/6, over SAMPLES and deployment renders — the only trustworthy
   read on whether the register actually trained, and whether it survives a
   distilled endpoint (it often does not: see krea2 Turbo at scale 1.0).

USAGE
-----
    export GEMINI_API_KEY="..."
    source .venv-captioning/bin/activate

    python scripts/register_judge_gemini.py \
        --refs /path/to/dataset/09.png /path/to/dataset/02.png /path/to/dataset/39.png \
        --register "oil-paint smearing with melted edges, vertical pixel-streak
                    curtains, blocky datamosh tearing, RGB chromatic fringing" \
        --exclude "palette, sunsets, lens flares, motion blur, silhouettes" \
        --axes smear streak datamosh fringing \
        --out output/<run>/register_dataset.json \
        /path/to/dataset/*.png

    # pairwise A/B when absolute scores saturate (see the memory note
    # ab_style_judge_saturates_switch_to_detail_metric)
    python scripts/register_judge_gemini.py --refs ... --register ... \
        --pairwise a.png:b.png c.png:d.png

Install: pip install google-genai pillow
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

from google import genai
from google.genai import types
from PIL import Image

REQUEST_TIMEOUT_MS = 120_000
MAX_SIDE = 1024

SCORE_PROMPT = """You judge whether an image carries one specific visual REGISTER. The reference images shown first define it; they are the artist's own work and they anchor the top of the scale.

THE REGISTER: {register}

DOES NOT COUNT (these are present in the references but are NOT what you are scoring — an image showing only these scores 0): {exclude}

For the TEST image return ONLY JSON:
{{"register": 0-3, {axes}, "note": "<=15 words"}}

register: 0 = none of it, the image is clean/unprocessed in this respect; 1 = a trace; 2 = clearly present, comparable to the references; 3 = heavier than the references.
Score what you can actually see. Do NOT compress a 0 up to 1 to be generous — a clean image scoring 0 is the single most useful output this judge produces."""

PAIR_PROMPT = """You judge which of two images carries MORE of one specific visual REGISTER. The reference images shown first define it.

THE REGISTER: {register}

DOES NOT COUNT: {exclude}

Both images share a subject, so ignore subject and composition entirely. Return ONLY JSON:
{{"winner": "A"|"B"|"tie", "margin": "none"|"slight"|"clear"|"large", "note": "<=20 words, in REGISTER terms only"}}"""


def load(path):
    im = Image.open(path).convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    return im


def ref_block(refs):
    out = []
    for i, im in enumerate(refs, 1):
        out += [f"REFERENCE {i}:", im]
    return out


def call(client, model, system, contents, attempts=3):
    delay = 2.0
    for n in range(1, attempts + 1):
        try:
            r = client.models.generate_content(
                model=model, contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system, temperature=0.1,
                    response_mime_type="application/json"))
            out = json.loads(r.text)
            return out[0] if isinstance(out, list) and out else out
        except Exception as e:
            if n == attempts:
                return {"error": str(e)[:120]}
            time.sleep(delay)
            delay *= 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", help="Images to score (ignored with --pairwise)")
    ap.add_argument("--refs", nargs="+", required=True,
                    help="2-4 reference images that ARE the register (the artist picks these)")
    ap.add_argument("--register", required=True,
                    help="What the register IS, in concrete visual terms")
    ap.add_argument("--exclude", default="palette, lighting, subject, composition",
                    help="What must NOT count toward the score")
    ap.add_argument("--axes", nargs="*", default=[],
                    help="Optional sub-axes scored 0-3 each, e.g. --axes smear streak datamosh")
    ap.add_argument("--pairwise", nargs="*", default=None, metavar="A:B",
                    help="Compare pairs instead of scoring absolutely")
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--out", default=None, help="Write results as JSON here")
    args = ap.parse_args()

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("Set GEMINI_API_KEY before running.", file=sys.stderr)
        sys.exit(1)
    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))

    refs = [load(p) for p in args.refs]
    results = {}

    if args.pairwise is not None:
        system = PAIR_PROMPT.format(register=args.register, exclude=args.exclude)
        for spec in args.pairwise:
            a, b = spec.split(":", 1)
            contents = ref_block(refs) + ["IMAGE A:", load(a), "IMAGE B:", load(b),
                                          "Which carries more of the reference register?"]
            res = call(client, args.model, system, contents)
            results[spec] = res
            print(f"{os.path.basename(a)} vs {os.path.basename(b)}: {res}", flush=True)
    else:
        axes = ", ".join(f'"{a}": 0-3' for a in args.axes) if args.axes else '"detail": 0-3'
        system = SCORE_PROMPT.format(register=args.register, exclude=args.exclude, axes=axes)
        for p in args.images:
            contents = ref_block(refs) + ["TEST IMAGE:", load(p), "Score the TEST image."]
            res = call(client, args.model, system, contents)
            results[p] = res
            print(f"{os.path.basename(p)}: {res}", flush=True)
        scored = [v["register"] for v in results.values() if isinstance(v.get("register"), int)]
        if scored:
            hist = sorted(Counter(scored).items())
            heavy = sum(n for s, n in hist if s >= 2)
            print(f"\ncoverage: {heavy}/{len(scored)} at >=2  ({heavy / len(scored):.0%})  "
                  f"histogram {hist}")
            if heavy / len(scored) < 0.6:
                print("  -> under ~60%: this register will NOT bind by omission. "
                      "Plan on EMA off, higher rank, inverse-marking the clean plates, "
                      "or training the heavy subset.")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
