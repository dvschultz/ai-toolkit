"""
Merge two ai-toolkit LoRAs into a single .safetensors file by concatenating ranks.

Concatenation is the *exact* way to combine two LoRAs without SVD recompression.
For LoRA modules A and B with the same target weight matrix:

    delta_A = (alpha_A / r_A) * up_A @ down_A
    delta_B = (alpha_B / r_B) * up_B @ down_B
    delta_merged = w_A * delta_A + w_B * delta_B

This is exactly reproducible by stacking:

    new_down = concat([w_A * (alpha_A/r_A) * down_A,  w_B * (alpha_B/r_B) * down_B], dim=0)
    new_up   = concat([up_A, up_B], dim=1)

A naive "same-rank weighted sum" (down_A + down_B) introduces cross terms and
is wrong unless followed by SVD — concatenation avoids that entirely. The cost
is a rank-(r_A + r_B) output, which roughly doubles the file size.

KEY-NAMING CONVENTIONS (this script handles BOTH, and you must pick the right
OUTPUT convention or your merged LoRA silently won't load):
  - PEFT / diffusers:  <module>.lora_A.weight  (= down, shape (r,in)),
                       <module>.lora_B.weight  (= up,   shape (out,r)),  NO alpha key.
  - kohya / sd-scripts: <module>.lora_down.weight, <module>.lora_up.weight, <module>.alpha.

ai-toolkit saves Klein/Flux.2 LoRAs in the **PEFT** convention, and the fal
`fal-ai/flux-2/klein/9b/base/lora` endpoint expects PEFT. Emitting kohya for a
fal-Klein LoRA makes fal SILENTLY IGNORE it — outputs come back looking like
base Klein (style absent), which is easy to misread as "wrong scale". So the
default output convention here is `match_a` (match the first LoRA's convention).
Use `--out-convention` to force one.

Usage:
    python scripts/merge_loras.py \\
        --lora_a output/.../style.safetensors --weight_a 1.0 \\
        --lora_b output/.../turbo.safetensors --weight_b 0.5 \\
        --output output/merged/style_x_turbo.safetensors
        # add --out-convention peft|kohya if you don't want match_a

Both LoRAs MUST share the same base model and target the same modules.
"""
import argparse
import os
from typing import Dict, Tuple, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file


# (down_suffix, up_suffix) per convention. alpha is kohya-only.
_PEFT = (".lora_A.weight", ".lora_B.weight")
_KOHYA = (".lora_down.weight", ".lora_up.weight")


def _load(path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    """Load a safetensors file as fp32 tensors plus metadata."""
    tensors: Dict[str, torch.Tensor] = {}
    metadata: Dict[str, str] = {}
    with safe_open(path, framework="pt") as f:
        metadata.update(f.metadata() or {})
        for k in f.keys():
            tensors[k] = f.get_tensor(k).to(torch.float32)
    return tensors, metadata


def detect_convention(state: Dict[str, torch.Tensor]) -> str:
    """Return 'peft' or 'kohya' based on which down/up suffixes appear. Raises
    if neither (or both ambiguously) is present."""
    has_peft = any(k.endswith(_PEFT[0]) for k in state)
    has_kohya = any(k.endswith(_KOHYA[0]) for k in state)
    if has_peft and not has_kohya:
        return "peft"
    if has_kohya and not has_peft:
        return "kohya"
    if has_peft and has_kohya:
        raise ValueError("file mixes PEFT (lora_A/B) and kohya (lora_down/up) keys; cannot detect convention")
    raise ValueError("no LoRA modules found (neither lora_A/lora_B nor lora_down/lora_up)")


def index_modules(state: Dict[str, torch.Tensor], conv: str) -> Dict[str, dict]:
    """Return {module: {'down': T(r,in), 'up': T(out,r), 'alpha': float|None}}
    normalized across conventions. PEFT: lora_A=down, lora_B=up, no alpha."""
    down_suf, up_suf = (_PEFT if conv == "peft" else _KOHYA)
    out: Dict[str, dict] = {}
    for k, v in state.items():
        if k.endswith(down_suf):
            out.setdefault(k[:-len(down_suf)], {})["down"] = v
        elif k.endswith(up_suf):
            out.setdefault(k[:-len(up_suf)], {})["up"] = v
    # alpha (kohya only)
    if conv == "kohya":
        for k, v in state.items():
            if k.endswith(".alpha"):
                m = k[:-len(".alpha")]
                if m in out:
                    out[m]["alpha"] = float(v.item())
    for m, d in out.items():
        d.setdefault("alpha", None)
    return out


def _scale(weight: float, alpha: Optional[float], rank: int) -> float:
    """Effective baked scale = weight * (alpha/rank). alpha=None (PEFT / no
    alpha key) → alpha=rank → scale 1.0 * weight (the standard interpretation)."""
    a = float(rank) if alpha is None else alpha
    return weight * (a / rank)


def merge(idx_a: Dict[str, dict], idx_b: Dict[str, dict],
          weight_a: float, weight_b: float) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Concat-merge. Returns {module: (new_down, new_up)} with weights baked into
    new_down (so the output's implied scale is 1.0 in either convention)."""
    mods_a, mods_b = set(idx_a), set(idx_b)
    only_a, only_b, shared = mods_a - mods_b, mods_b - mods_a, mods_a & mods_b
    if only_a or only_b:
        print(f"[info] {len(only_a)} modules only in A, {len(only_b)} only in B; "
              f"passed through with their own weight. (If one side is ~0 and you "
              f"expected overlap, suspect a convention/base mismatch.)")

    out: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    for m in sorted(shared):
        da, ua = idx_a[m]["down"], idx_a[m]["up"]
        db, ub = idx_b[m]["down"], idx_b[m]["up"]
        ra, rb = da.shape[0], db.shape[0]
        if da.shape[1] != db.shape[1] or ua.shape[0] != ub.shape[0]:
            raise ValueError(f"target shape mismatch for {m}: in {da.shape[1]} vs {db.shape[1]}, "
                             f"out {ua.shape[0]} vs {ub.shape[0]}")
        sa = _scale(weight_a, idx_a[m]["alpha"], ra)
        sb = _scale(weight_b, idx_b[m]["alpha"], rb)
        new_down = torch.cat([sa * da, sb * db], dim=0)
        new_up = torch.cat([ua, ub], dim=1)
        out[m] = (new_down, new_up)

    for m in sorted(only_a):
        d, u = idx_a[m]["down"], idx_a[m]["up"]
        out[m] = (_scale(weight_a, idx_a[m]["alpha"], d.shape[0]) * d, u)
    for m in sorted(only_b):
        d, u = idx_b[m]["down"], idx_b[m]["up"]
        out[m] = (_scale(weight_b, idx_b[m]["alpha"], d.shape[0]) * d, u)

    return out


def emit(merged: Dict[str, Tuple[torch.Tensor, torch.Tensor]], conv: str,
         out_dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    """Serialize merged modules into the chosen convention's keys. Scale is
    already baked into new_down, so kohya alpha = rank (scale 1.0) and PEFT
    needs no alpha — both apply exactly delta = up @ down."""
    down_suf, up_suf = (_PEFT if conv == "peft" else _KOHYA)
    state: Dict[str, torch.Tensor] = {}
    for m, (down, up) in merged.items():
        state[m + down_suf] = down.to(out_dtype)
        state[m + up_suf] = up.to(out_dtype)
        if conv == "kohya":
            state[m + ".alpha"] = torch.tensor(float(down.shape[0]))  # alpha=rank → scale 1.0
    return state


def main():
    p = argparse.ArgumentParser(description="Concat-merge two ai-toolkit LoRAs (PEFT- and kohya-aware).")
    p.add_argument("--lora_a", required=True, help="Path to first LoRA .safetensors")
    p.add_argument("--lora_b", required=True, help="Path to second LoRA .safetensors")
    p.add_argument("--output", required=True, help="Output merged .safetensors path")
    p.add_argument("--weight_a", type=float, default=1.0)
    p.add_argument("--weight_b", type=float, default=1.0)
    p.add_argument("--out-convention", choices=["match_a", "peft", "kohya"], default="match_a",
                   help="Output key convention. 'match_a' (default) follows lora_a. "
                        "fal-ai/flux-2/klein wants PEFT; kohya output there loads as a no-op.")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    args = p.parse_args()

    out_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    print(f"Loading A: {args.lora_a}")
    state_a, meta_a = _load(args.lora_a)
    conv_a = detect_convention(state_a)
    print(f"Loading B: {args.lora_b}")
    state_b, _ = _load(args.lora_b)
    conv_b = detect_convention(state_b)
    print(f"Detected conventions — A: {conv_a}, B: {conv_b}")

    out_conv = conv_a if args.out_convention == "match_a" else args.out_convention
    print(f"Merging weight_a={args.weight_a}, weight_b={args.weight_b} (concat); output convention: {out_conv}")

    idx_a = index_modules(state_a, conv_a)
    idx_b = index_modules(state_b, conv_b)
    merged = merge(idx_a, idx_b, args.weight_a, args.weight_b)
    state = emit(merged, out_conv, out_dtype)

    metadata = {
        "merge_source_a": os.path.basename(args.lora_a),
        "merge_source_b": os.path.basename(args.lora_b),
        "merge_weight_a": str(args.weight_a),
        "merge_weight_b": str(args.weight_b),
        "merge_mode": "concat",
        "merge_out_convention": out_conv,
    }
    for k, v in meta_a.items():
        if k.startswith("ss_") and k not in metadata:
            metadata[k] = v

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_file(state, args.output, metadata=metadata)

    n_modules = len(merged)
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Wrote {args.output}  ({n_modules} modules, {out_conv} convention, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
