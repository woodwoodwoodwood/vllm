"""Diagnostic tool: inspect the structure of a MiLo native checkpoint.

A MiLo HF runtime checkpoint typically lives in a directory like:

    <ckpt>/
        config.json              # vanilla HF config (NO quantization_config)
        qmodel.pt                # main quantized weights (~7-30 GB)
        compensators.pt          # low-rank compensator V/U buffers
        ranks.json               # per-module rank mapping
        tokenizer.* / vocab.*    # standard HF tokenizer files

This script dumps the layout of `qmodel.pt` + `compensators.pt` so we know
exactly which keys / shapes / dtypes to handle when writing the
vLLM-compatible converter (`convert_milo_to_vllm.py`).

Usage:
    python inspect_milo_checkpoint.py --ckpt /path/to/Qwen1.5-Moe-A2.7B_w3s16d512
    # or just edit DEFAULT_CKPT below and run with no args.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import torch


DEFAULT_CKPT = "/usr/local/app/models/Qwen1.5-Moe-A2.7B_w3s16d512"


def _print_tensor_row(key: str, val) -> None:
    """Pretty-print one (key, tensor_or_obj) entry."""
    if hasattr(val, "shape"):
        print(f"  {key:100s}  shape={tuple(val.shape)}  dtype={val.dtype}")
    else:
        print(f"  {key:100s}  type={type(val).__name__}")


def _walk_value(prefix: str, v, *, depth: int = 0, max_depth: int = 4):
    """Recursively yield (path, leaf) where leaf is a tensor or scalar.

    Handles nested dicts, tuples and lists, which is the layout that MiLo's
    `qmodel.pt` / `compensators.pt` actually use (each module is stored as
    a nested dict whose values are the actual quantized tensors)."""
    if depth > max_depth:
        yield (prefix, "<MAX_DEPTH>")
        return
    if isinstance(v, dict):
        if not v:
            yield (prefix, "<EMPTY DICT>")
            return
        for k, sub in v.items():
            yield from _walk_value(f"{prefix}.{k}", sub,
                                   depth=depth + 1, max_depth=max_depth)
    elif isinstance(v, (tuple, list)):
        if not v:
            yield (prefix, f"<EMPTY {type(v).__name__}>")
            return
        for i, sub in enumerate(v):
            yield from _walk_value(f"{prefix}[{i}]", sub,
                                   depth=depth + 1, max_depth=max_depth)
    else:
        yield (prefix, v)


def _print_subtree(prefix: str, v, *, max_lines: int = 60) -> None:
    """Recursively print the full sub-tree rooted at `prefix`."""
    n = 0
    truncated = False
    for path, leaf in _walk_value(prefix, v):
        if n >= max_lines:
            truncated = True
            break
        if hasattr(leaf, "shape"):
            print(f"  {path:100s}  shape={tuple(leaf.shape)}  dtype={leaf.dtype}")
        else:
            # show small scalars / strings inline; stringify the rest as type
            if isinstance(leaf, (int, float, bool, str)) or leaf is None:
                print(f"  {path:100s}  {type(leaf).__name__}={leaf!r}")
            else:
                print(f"  {path:100s}  type={type(leaf).__name__}")
        n += 1
    if truncated:
        print(f"  ... (truncated at {max_lines} entries)")


def _dump_dict_section(d: dict, header: str, key_filter, *, max_show: int = 40) -> None:
    """Print up to `max_show` (key, val) rows whose keys satisfy key_filter(k)."""
    matching = [k for k in d if key_filter(k)]
    print(f"\n--- {header} ---")
    print(f"  count: {len(matching)}")
    for k in matching[:max_show]:
        _print_tensor_row(k, d[k])
    if len(matching) > max_show:
        print(f"  ... ({len(matching) - max_show} more)")


def _summarise_per_layer(d: dict, label: str) -> None:
    """Bucket keys by `layers.<i>.` index and print per-layer key count."""
    counts: Counter = Counter()
    no_layer = 0
    for k in d:
        # extract layer index if present
        if "layers." in k:
            try:
                idx = int(k.split("layers.", 1)[1].split(".", 1)[0])
                counts[idx] += 1
            except ValueError:
                no_layer += 1
        else:
            no_layer += 1
    if counts:
        layer_idxs = sorted(counts)
        # show only the spread, not all
        sample_idxs = layer_idxs if len(layer_idxs) <= 5 else (
            layer_idxs[:2] + layer_idxs[len(layer_idxs)//2:len(layer_idxs)//2+1] + layer_idxs[-2:]
        )
        per_layer = ", ".join(f"L{i}={counts[i]}" for i in sample_idxs)
        print(f"  [{label}] layers seen: {len(layer_idxs)} "
              f"(idx range {layer_idxs[0]}..{layer_idxs[-1]}), "
              f"sample counts: {per_layer}")
    if no_layer:
        print(f"  [{label}] non-layer keys: {no_layer}")


def _summarise_dtypes(d: dict, label: str) -> None:
    """Print histogram of dtypes among the tensors in `d`."""
    dtypes: Counter = Counter()
    n_tensors = 0
    for v in d.values():
        if hasattr(v, "dtype"):
            dtypes[str(v.dtype)] += 1
            n_tensors += 1
    if dtypes:
        breakdown = ", ".join(f"{k}={n}" for k, n in dtypes.most_common())
        print(f"  [{label}] {n_tensors} tensors, dtype histogram: {breakdown}")


def inspect_qmodel(path: str) -> None:
    print("=" * 80)
    print(f"=== {path}")
    print("=" * 80)
    if not os.path.exists(path):
        print(f"  FILE NOT FOUND: {path}")
        return
    sz = os.path.getsize(path) / 1e9
    print(f"  size: {sz:.2f} GB")

    sd = torch.load(path, map_location="cpu", weights_only=False)
    print(f"  type: {type(sd).__name__}")
    if not isinstance(sd, dict):
        print(f"  (not a dict — content: {sd!r})")
        return

    keys = list(sd.keys())
    print(f"  total top-level keys: {len(keys)}")

    _summarise_per_layer(sd, "qmodel")

    # Show the *types* of top-level values (most important new info — values
    # are typically nested dicts representing whole module state-dicts).
    print()
    top_types: Counter = Counter()
    for v in sd.values():
        top_types[type(v).__name__] += 1
    print(f"  top-level value types: "
          f"{', '.join(f'{n}={c}' for n, c in top_types.most_common())}")

    # Pick a few representative top-level keys and recursively print their
    # full content so we can see what actual tensor names sit inside.
    interesting_top_keys = [
        ("model.embed_tokens",                     "embed_tokens"),
        ("model.layers.0.self_attn.q_proj",        "layer 0 self_attn.q_proj"),
        ("model.layers.0.mlp.experts.0.gate_proj", "layer 0 expert 0 gate_proj"),
        ("model.layers.0.mlp.experts.0.down_proj", "layer 0 expert 0 down_proj"),
        ("model.layers.0.mlp.shared_expert.gate_proj",
                                                   "layer 0 shared_expert.gate_proj"),
        ("model.layers.0.mlp.shared_expert_gate",  "layer 0 shared_expert_gate"),
        ("model.layers.0.mlp.gate",                "layer 0 mlp.gate (router)"),
        ("model.norm",                             "model.norm"),
        ("lm_head",                                "lm_head"),
    ]
    for top_key, label in interesting_top_keys:
        if top_key not in sd:
            continue
        print(f"\n--- {label}  (top key = {top_key!r}) ---")
        print(f"  outer type: {type(sd[top_key]).__name__}")
        _print_subtree(top_key, sd[top_key], max_lines=80)

    # Histogram of leaf tensor shapes for one expert across all layers — this
    # tells us whether layers share schema (they should).
    print("\n--- leaf-key suffix histogram across ALL layer-0 expert-0 modules ---")
    suffix_counts: Counter = Counter()
    for top_key, val in sd.items():
        if "layers.0." in top_key and "experts.0." in top_key:
            for path, leaf in _walk_value(top_key, val):
                if hasattr(leaf, "shape"):
                    # take the suffix after the top_key
                    tail = path[len(top_key) + 1:] if path.startswith(top_key) else path
                    suffix_counts[tail] += 1
    for suf, n in suffix_counts.most_common(30):
        print(f"  {suf:60s}  occurrences: {n}")


def inspect_compensators(path: str) -> None:
    print("=" * 80)
    print(f"=== {path}")
    print("=" * 80)
    if not os.path.exists(path):
        print(f"  FILE NOT FOUND: {path}")
        return
    sz = os.path.getsize(path) / 1e6
    print(f"  size: {sz:.1f} MB")

    comp = torch.load(path, map_location="cpu", weights_only=False)
    print(f"  type: {type(comp).__name__}")
    if not isinstance(comp, dict):
        print(f"  (not a dict — content: {comp!r})")
        return

    keys = list(comp.keys())
    print(f"  total keys: {len(keys)}")

    _summarise_per_layer(comp, "compensators")

    print()
    top_types: Counter = Counter()
    for v in comp.values():
        top_types[type(v).__name__] += 1
    print(f"  top-level value types: "
          f"{', '.join(f'{n}={c}' for n, c in top_types.most_common())}")

    # If values are tuples (V, U), print one example fully expanded.
    interesting_top_keys = [
        ("model.layers.0.mlp.experts.0.gate_proj",   "expert 0 gate_proj compensator"),
        ("model.layers.0.mlp.experts.0.down_proj",   "expert 0 down_proj compensator"),
        ("model.layers.0.self_attn.q_proj",          "self_attn.q_proj compensator"),
        ("model.layers.0.self_attn.o_proj",          "self_attn.o_proj compensator"),
        ("model.layers.0.mlp.shared_expert.gate_proj",
                                                     "shared_expert.gate_proj compensator"),
    ]
    for top_key, label in interesting_top_keys:
        if top_key not in comp:
            continue
        print(f"\n--- {label}  (top key = {top_key!r}) ---")
        print(f"  outer type: {type(comp[top_key]).__name__}")
        _print_subtree(top_key, comp[top_key], max_lines=20)

    # Cross-layer leaf shape histogram for one MoE expert vs one attn proj —
    # tells us whether all layers share the same V/U shape (they should).
    def _shape_set(predicate):
        shapes: Counter = Counter()
        for k, v in comp.items():
            if not predicate(k):
                continue
            for path, leaf in _walk_value(k, v):
                if hasattr(leaf, "shape"):
                    tail = path[len(k) + 1:] if path.startswith(k) else path
                    shapes[(tail, tuple(leaf.shape), str(leaf.dtype))] += 1
        return shapes

    print("\n--- compensator leaf shape histogram (all expert-0 gate_proj across 24 layers) ---")
    h = _shape_set(lambda k: "experts.0.gate_proj" in k)
    for (tail, shp, dt), n in h.most_common(10):
        print(f"  {tail:40s}  shape={shp}  dtype={dt}  ×{n}")

    print("\n--- compensator leaf shape histogram (all self_attn.q_proj across 24 layers) ---")
    h = _shape_set(lambda k: "self_attn.q_proj" in k)
    for (tail, shp, dt), n in h.most_common(10):
        print(f"  {tail:40s}  shape={shp}  dtype={dt}  ×{n}")

    print("\n--- compensator leaf shape histogram (all shared_expert.gate_proj across 24 layers) ---")
    h = _shape_set(lambda k: "shared_expert.gate_proj" in k)
    for (tail, shp, dt), n in h.most_common(10):
        print(f"  {tail:40s}  shape={shp}  dtype={dt}  ×{n}")


def inspect_ranks(path: str) -> None:
    print("=" * 80)
    print(f"=== {path}")
    print("=" * 80)
    if not os.path.exists(path):
        print(f"  FILE NOT FOUND: {path}")
        return
    with open(path) as f:
        ranks = json.load(f)
    print(f"  content: {ranks}")


def inspect_config(path: str) -> None:
    print("=" * 80)
    print(f"=== {path}")
    print("=" * 80)
    if not os.path.exists(path):
        print(f"  FILE NOT FOUND: {path}")
        return
    with open(path) as f:
        cfg = json.load(f)

    keys_of_interest = [
        "model_type", "architectures",
        "hidden_size", "intermediate_size", "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "num_experts", "num_experts_per_tok",
        "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "vocab_size", "torch_dtype",
    ]
    for k in keys_of_interest:
        if k in cfg:
            print(f"  {k:40s}  = {cfg[k]}")

    if "quantization_config" in cfg:
        print("\n  quantization_config:")
        for k, v in cfg["quantization_config"].items():
            print(f"    {k}: {v}")
    else:
        print("\n  (no quantization_config block in config.json)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a MiLo native checkpoint directory.")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT,
                        help=f"Checkpoint directory (default: {DEFAULT_CKPT})")
    parser.add_argument("--no-compensators", action="store_true",
                        help="Skip compensators.pt (it's faster to skip).")
    args = parser.parse_args()

    ckpt = args.ckpt
    if not os.path.isdir(ckpt):
        print(f"ERROR: not a directory: {ckpt}", file=sys.stderr)
        sys.exit(1)

    print(f"Inspecting: {ckpt}")
    print(f"Files in dir:")
    for fname in sorted(os.listdir(ckpt)):
        full = os.path.join(ckpt, fname)
        if os.path.isfile(full):
            print(f"  {fname:30s}  {os.path.getsize(full):>12,d} B")

    inspect_config(os.path.join(ckpt, "config.json"))
    inspect_ranks(os.path.join(ckpt, "ranks.json"))
    inspect_qmodel(os.path.join(ckpt, "qmodel.pt"))
    if not args.no_compensators:
        inspect_compensators(os.path.join(ckpt, "compensators.pt"))


if __name__ == "__main__":
    main()
