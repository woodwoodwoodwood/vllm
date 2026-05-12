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


DEFAULT_CKPT = "/data/home/cakejiang/models/Qwen1.5-Moe-A2.7B_w3s16d512"


def _print_tensor_row(key: str, val) -> None:
    """Pretty-print one (key, tensor_or_obj) entry."""
    if hasattr(val, "shape"):
        print(f"  {key:100s}  shape={tuple(val.shape)}  dtype={val.dtype}")
    else:
        print(f"  {key:100s}  type={type(val).__name__}")


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
    _summarise_dtypes(sd, "qmodel")

    print("\n--- first 10 keys (any) ---")
    for k in keys[:10]:
        _print_tensor_row(k, sd[k])

    _dump_dict_section(sd, "layer 0 expert 0",
                       lambda k: "layers.0." in k and "experts.0." in k)
    _dump_dict_section(sd, "layer 0 shared_expert",
                       lambda k: "layers.0." in k and "shared_expert" in k)
    _dump_dict_section(sd, "layer 0 self_attn",
                       lambda k: "layers.0." in k and "self_attn" in k)
    _dump_dict_section(sd, "non-layer keys (embed / norm / lm_head / router)",
                       lambda k: "layers." not in k)

    # collect the *suffix patterns* used for quantized linear modules.
    # Anything after "experts.<E>.<rail>." or after "self_attn.<proj>."
    suffixes: Counter = Counter()
    for k in keys:
        if "experts.0." in k and "layers.0." in k:
            tail = k.rsplit(".", 1)[-1]
            suffixes[tail] += 1
    if suffixes:
        print("\n--- expert-rail key suffixes (layer 0 expert 0) ---")
        for s, n in suffixes.most_common():
            print(f"  .{s:40s}  occurrences: {n}")


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
    _summarise_dtypes(comp, "compensators")

    print("\n--- first 10 keys (any) ---")
    for k in keys[:10]:
        _print_tensor_row(k, comp[k])

    _dump_dict_section(comp, "layer 0 expert 0",
                       lambda k: "layers.0" in k and "experts.0" in k)
    _dump_dict_section(comp, "layer 0 shared_expert",
                       lambda k: "layers.0" in k and "shared_expert" in k)
    _dump_dict_section(comp, "layer 0 self_attn",
                       lambda k: "layers.0" in k and "self_attn" in k)

    # suffix histogram
    suffixes: Counter = Counter()
    for k in keys:
        tail = k.rsplit(".", 1)[-1]
        suffixes[tail] += 1
    if suffixes:
        print("\n--- compensator key suffix histogram ---")
        for s, n in suffixes.most_common(15):
            print(f"  .{s:40s}  occurrences: {n}")


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
