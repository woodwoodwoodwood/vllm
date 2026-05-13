"""Milestone 2: convert a MiLo native checkpoint into a vLLM-loadable
safetensors checkpoint.

Pipeline
--------
    1. Load the source checkpoint via MiLo's own `from_compressed`
       (handles qmodel.pt / compensators.pt / ranks.json automatically).
    2. Walk the model, replacing every `MiLoLinear` (HQQ-format `W_q`)
       with a `MiLo_Asymmetric_Linear` (Marlin-prepacked `Wq_packed1/2`).
       This is exactly what every MiLo benchmark script already does.
       After this step each quantized linear exposes the kernel-ready
       buffers we need to dump.
    3. Walk again and dump every parameter into a flat state-dict:
         * Quantized linear modules (attn / shared_expert / routed
           experts):  Wq_packed1, Wq_packed2, scales, zeros, V, U, bias
         * Non-quantized weights (embed, norm, lm_head, router gates):
           plain fp16 .weight tensors verbatim.
    4. Save as sharded safetensors (≤ ~5 GB per shard so each file is
       below the 2^32-byte safetensors limit), generate the index json,
       write `config.json` with a `quantization_config` block matching
       the runtime expectations of `MiloConfig`, and copy tokenizer
       files verbatim from the source dir.

Usage
-----
    PYTHONPATH=/usr/local/app/milo python convert_milo_to_vllm.py \\
        --src /usr/local/app/models/Qwen1.5-Moe-A2.7B_w3s16d512 \\
        --dst /usr/local/app/models/Qwen1.5-MoE-A2.7B-vllm-int3
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from collections import OrderedDict
from typing import Dict, Optional

import torch
from safetensors.torch import save_file


# ---------------------------------------------------------------------------
# 1. Helpers
# ---------------------------------------------------------------------------
DEFAULT_SRC = "/usr/local/app/models/Qwen1.5-Moe-A2.7B_w3s16d512"
DEFAULT_DST = "/usr/local/app/models/Qwen1.5-MoE-A2.7B-vllm-int3"

# Keys we expose on each MiLo_Asymmetric_Linear (constants in the upstream
# class definition).  Every key is optional except (Wq_packed1/2, scales,
# zeros).  See /usr/local/app/milo/MiLo/backends/milo.py.
_QUANT_BUFFER_NAMES = (
    "Wq_packed1",   # int32  [k/16, n]
    "Wq_packed2",   # int32  [k/16, n*16/32]
    "scales",       # fp16   [k/gs, n]
    "zeros",        # fp16   [k/gs, n]
    "V",            # fp16   [in,   rank]   compensator left
    "U",            # fp16   [rank, out]    compensator right
    "bias",         # fp16   [out]          (only on attn projections that have bias)
)


def _import_milo():
    """Lazy import + clear error if MiLo can't be loaded."""
    try:
        import MiLo.core.quantize as _q              # noqa: F401  registers backend
        from MiLo.core.quantize import MiLoLinear
        from MiLo.models.hf.qwen2_moe import Qwen2MoEMiLo
        from MiLo.backends.milo import (
            MiLo_Asymmetric_Linear, patch_hqq_to_milo_asymmetric)
    except ImportError as e:
        print(f"ImportError: {e}", file=sys.stderr)
        print("Make sure /usr/local/app/milo is on PYTHONPATH "
              "and milo_cuda is installed.", file=sys.stderr)
        sys.exit(1)
    return (Qwen2MoEMiLo, MiLoLinear, MiLo_Asymmetric_Linear,
            patch_hqq_to_milo_asymmetric)


def _patch_all_milolinear(module, MiLoLinear, patch_fn,
                          *, verbose: bool = False) -> int:
    """Recursively walk `module`, replacing every MiLoLinear (nbits==3)
    with the Marlin-prepacked MiLo_Asymmetric_Linear.

    Returns the number of layers that were patched.
    """
    n_patched = 0
    for name, child in list(module.named_children()):
        if isinstance(child, MiLoLinear):
            if child.meta.get("nbits") != 3:
                if verbose:
                    print(f"  skip non-3bit MiLoLinear: nbits="
                          f"{child.meta.get('nbits')}")
                continue
            new_child = patch_fn(child, None)
            if new_child is not child:
                setattr(module, name, new_child)
                n_patched += 1
        else:
            n_patched += _patch_all_milolinear(child, MiLoLinear, patch_fn,
                                               verbose=verbose)
    return n_patched


def _collect_state_dict(model, MiLo_Asymmetric_Linear,
                        *, verbose: bool = False) -> "OrderedDict[str, torch.Tensor]":
    """Build a flat {key: cpu fp16 tensor} dict for every parameter in
    `model`, using:

      * MiLo_Asymmetric_Linear modules → 5–7 buffers per the schema in
        `_QUANT_BUFFER_NAMES`.
      * Everything else → its plain `nn.Parameter` / `nn.Buffer`s.

    The result is a "vLLM-style" state dict: each tensor lives at a key
    like `model.layers.{L}.mlp.experts.{E}.gate_proj.{Wq_packed1|...}`.
    """
    sd: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    n_quant_modules = 0
    n_quant_buffers = 0
    n_other_params  = 0

    visited_quant_modules = set()

    for mod_name, mod in model.named_modules():
        # ---- Quantized linear ----
        if isinstance(mod, MiLo_Asymmetric_Linear):
            visited_quant_modules.add(mod_name)
            n_quant_modules += 1
            for buf_name in _QUANT_BUFFER_NAMES:
                if not hasattr(mod, buf_name):
                    continue
                t = getattr(mod, buf_name)
                if t is None:
                    continue
                if not torch.is_tensor(t):
                    continue
                # Move to CPU + contiguous before serialisation; safetensors
                # only writes contiguous tensors.
                cpu_t = t.detach().to("cpu", copy=True).contiguous()
                key = f"{mod_name}.{buf_name}" if mod_name else buf_name
                sd[key] = cpu_t
                n_quant_buffers += 1
            if verbose:
                # print one in N modules to keep the log readable
                if n_quant_modules <= 3 or n_quant_modules % 50 == 0:
                    print(f"  quant module #{n_quant_modules}: {mod_name}")

    # ---- Everything else: walk leaves whose owning module wasn't already
    #      handled above.  We do this via named_parameters() and skip keys
    #      that fall under any of the visited quant modules.
    for pname, param in model.named_parameters():
        # Skip parameters owned by a quant module (they were dumped via the
        # custom buffer schema above).
        if any(pname.startswith(m + ".") or pname == m
               for m in visited_quant_modules):
            continue
        cpu_t = param.detach().to("cpu", copy=True).contiguous()
        sd[pname] = cpu_t
        n_other_params += 1

    # Buffers (e.g. RMSNorm running stats are usually parameters in HF
    # configs, but include them defensively).
    for bname, buf in model.named_buffers():
        if any(bname.startswith(m + ".") or bname == m
               for m in visited_quant_modules):
            continue
        if bname in sd:
            continue
        cpu_t = buf.detach().to("cpu", copy=True).contiguous()
        sd[bname] = cpu_t
        n_other_params += 1

    print(f"  collected state dict: "
          f"{n_quant_modules} quant modules ({n_quant_buffers} buffers), "
          f"{n_other_params} other params, "
          f"{len(sd)} total keys")
    return sd


def _shard_state_dict(sd: "OrderedDict[str, torch.Tensor]",
                      *, max_shard_bytes: int) -> list:
    """Greedy bin-packing into shards, each strictly ≤ max_shard_bytes."""
    shards: list[OrderedDict[str, torch.Tensor]] = [OrderedDict()]
    cur_bytes = 0
    for k, t in sd.items():
        sz = t.numel() * t.element_size()
        if sz > max_shard_bytes:
            print(f"  WARN: tensor {k} alone is {sz/1e9:.2f} GB > "
                  f"max_shard {max_shard_bytes/1e9:.2f} GB. "
                  f"Putting it in its own shard.", file=sys.stderr)
            if cur_bytes > 0:
                shards.append(OrderedDict())
                cur_bytes = 0
            shards[-1][k] = t
            shards.append(OrderedDict())
            cur_bytes = 0
            continue
        if cur_bytes + sz > max_shard_bytes:
            shards.append(OrderedDict())
            cur_bytes = 0
        shards[-1][k] = t
        cur_bytes += sz

    # Drop trailing empty shard
    if shards and not shards[-1]:
        shards.pop()
    return shards


def _write_safetensors(shards: list, dst_dir: str) -> dict:
    """Write each shard to dst_dir, return the index-json `weight_map`."""
    n_shards = len(shards)
    # Use 5-digit padding so 99999 shards still sort right; HF convention.
    fmt = "model-{idx:05d}-of-{total:05d}.safetensors"
    weight_map: dict[str, str] = {}
    total_size = 0
    for i, shard in enumerate(shards, start=1):
        fname = fmt.format(idx=i, total=n_shards)
        path  = os.path.join(dst_dir, fname)
        # safetensors metadata: HF convention recommends storing the format.
        meta  = {"format": "pt"}
        save_file(shard, path, metadata=meta)
        sz = os.path.getsize(path)
        total_size += sz
        for k in shard:
            weight_map[k] = fname
        print(f"  wrote {fname:48s}  {len(shard):>4d} keys  "
              f"{sz/1e9:>6.2f} GB")
    return {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }


# ---------------------------------------------------------------------------
# 2. Config writers
# ---------------------------------------------------------------------------
def _build_quantization_config(*, group_size: int = 64,
                               compensator_rank: int = 16,
                               has_compensator: bool = True,
                               extra_skip: Optional[list[str]] = None) -> dict:
    """Match the schema consumed by vllm/.../quantization/milo.py:MiloConfig."""
    return {
        "quant_method": "milo",
        "bits": 3,
        "group_size": group_size,
        "has_zp": True,
        "has_compensator": has_compensator,
        "compensator_rank": compensator_rank,
        # For the Qwen1.5-MoE checkpoint *every* linear is INT3 (attn,
        # shared_expert and routed experts), so leave this empty.  If you
        # want to selectively skip layers, pass them via `--skip-modules`.
        "modules_to_not_convert": list(extra_skip or []),
    }


def _write_config_json(src_dir: str, dst_dir: str, qcfg: dict,
                       *, ranks: dict) -> None:
    src_cfg_path = os.path.join(src_dir, "config.json")
    if not os.path.exists(src_cfg_path):
        raise FileNotFoundError(src_cfg_path)
    with open(src_cfg_path) as f:
        cfg = json.load(f)
    cfg["quantization_config"] = qcfg
    # Persist the source ranks.json alongside, in case the runtime ever needs
    # to know per-module-class rank distinctions.
    cfg["quantization_config"]["_milo_ranks"] = ranks
    with open(os.path.join(dst_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  wrote config.json (with quantization_config)")


def _copy_tokenizer(src_dir: str, dst_dir: str) -> None:
    n_copied = 0
    for fname in sorted(os.listdir(src_dir)):
        if fname.startswith(("tokenizer", "special_tokens_map", "vocab",
                             "merges", "added_tokens", "generation_config",
                             "chat_template")):
            shutil.copy(os.path.join(src_dir, fname),
                        os.path.join(dst_dir, fname))
            n_copied += 1
    print(f"  copied {n_copied} tokenizer / generation files verbatim")


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=DEFAULT_SRC,
                   help="MiLo native checkpoint dir (qmodel.pt + compensators.pt + ...)")
    p.add_argument("--dst", default=DEFAULT_DST,
                   help="Output dir for the vLLM-compatible safetensors checkpoint.")
    p.add_argument("--device", default="cuda",
                   help="Device to load and patch the model on. CPU works "
                        "but the patching is much slower (re-runs the "
                        "Marlin pack on CPU).")
    p.add_argument("--shard-size-gb", type=float, default=4.5,
                   help="Approx max size of each safetensors shard (GB).")
    p.add_argument("--skip-modules", nargs="*", default=[],
                   help="modules_to_not_convert hint to write into "
                        "config.json (does NOT affect the dump itself; the "
                        "runtime decides what to skip).")
    p.add_argument("--dry-run", action="store_true",
                   help="Load + patch + collect state_dict, but don't write "
                        "anything.  Useful to sanity-check shapes.")
    args = p.parse_args()

    if not os.path.isdir(args.src):
        print(f"ERROR: src not a directory: {args.src}", file=sys.stderr)
        sys.exit(1)

    print("=" * 76)
    print("Milestone-2: convert MiLo native checkpoint → vLLM safetensors")
    print("=" * 76)
    print(f"  src       : {args.src}")
    print(f"  dst       : {args.dst}")
    print(f"  device    : {args.device}")
    print(f"  shard size: {args.shard_size_gb:.1f} GB")
    print(f"  dry run   : {args.dry_run}")
    print()

    # ---- 1. Load & patch ----
    print("[1/5] Loading via MiLo from_compressed() ...", flush=True)
    t0 = time.time()
    (Qwen2MoEMiLo, MiLoLinear, MiLo_Asymmetric_Linear,
     patch_fn) = _import_milo()
    model = Qwen2MoEMiLo.from_compressed(args.src, device=args.device)
    model.eval()
    print(f"      ... done in {time.time() - t0:.1f}s")

    print("[2/5] Patching MiLoLinear → MiLo_Asymmetric_Linear "
          "(HQQ → Marlin prepack) ...", flush=True)
    t0 = time.time()
    n_patched = _patch_all_milolinear(model, MiLoLinear, patch_fn,
                                      verbose=False)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"      ... patched {n_patched} layers in {time.time() - t0:.1f}s")
    if n_patched == 0:
        print("  WARN: zero layers patched.  Either the checkpoint has no "
              "INT3 linears, or it was already patched.  Check the source "
              "ckpt.", file=sys.stderr)

    # ---- 2. Collect ----
    print("[3/5] Collecting state dict (CPU copies) ...", flush=True)
    t0 = time.time()
    sd = _collect_state_dict(model, MiLo_Asymmetric_Linear, verbose=False)
    print(f"      ... done in {time.time() - t0:.1f}s, {len(sd)} keys")
    total_bytes = sum(t.numel() * t.element_size() for t in sd.values())
    print(f"      total size: {total_bytes / 1e9:.2f} GB")

    # Free the GPU model now that everything is on CPU.
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.dry_run:
        print("\n[dry-run] skipping write step.")
        # Print a sample of keys for sanity
        sample_keys = [k for k in sd.keys() if "layers.0." in k][:10]
        print("  sample keys (layer 0):")
        for k in sample_keys:
            t = sd[k]
            print(f"    {k:90s}  shape={tuple(t.shape)}  dtype={t.dtype}")
        return

    # ---- 3. Shard + write safetensors ----
    print("[4/5] Sharding + writing safetensors ...", flush=True)
    t0 = time.time()
    os.makedirs(args.dst, exist_ok=True)
    max_shard_bytes = int(args.shard_size_gb * 1024**3)
    shards = _shard_state_dict(sd, max_shard_bytes=max_shard_bytes)
    index = _write_safetensors(shards, args.dst)
    with open(os.path.join(args.dst, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"      ... done in {time.time() - t0:.1f}s, "
          f"{len(shards)} shard(s)")

    # ---- 4. Config + tokenizer ----
    print("[5/5] Writing config.json + tokenizer files ...", flush=True)
    with open(os.path.join(args.src, "ranks.json")) as f:
        ranks = json.load(f)
    qcfg = _build_quantization_config(
        group_size=64,
        compensator_rank=ranks.get("mlp.experts", 16),  # routed expert rank
        has_compensator=True,
        extra_skip=args.skip_modules,
    )
    _write_config_json(args.src, args.dst, qcfg, ranks=ranks)
    _copy_tokenizer(args.src, args.dst)

    print()
    print("=" * 76)
    print(f"DONE.  vLLM-loadable checkpoint at: {args.dst}")
    print("=" * 76)
    print("Next step: launch vLLM with --quantization milo on this dir.")
    print("Note: vLLM's MiloMoEMethod is currently a TODO — full inference "
          "won't work end-to-end yet, but `vllm serve` should at least be "
          "able to inspect the checkpoint without errors.")


if __name__ == "__main__":
    main()
