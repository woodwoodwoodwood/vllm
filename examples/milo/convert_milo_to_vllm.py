"""Convert a MiLo HF runtime model into a vLLM-compatible checkpoint.

Produces:
  <output_dir>/
    config.json                 (with `quantization_config` block)
    model-{0001-of-NNNN}.safetensors
    model.safetensors.index.json
    tokenizer.* (copied from source)

Usage:
    python convert_milo_to_vllm.py \
        --src /path/to/Qwen3-MoE_w3       \
        --dst /path/to/Qwen3-MoE-vllm-int3

This expects the source checkpoint to be loadable by MiLo's HF runtime
(see eval_qwen3.py).  We load the model once with MiLo, then walk every
patched layer and dump its int3 buffers + compensator V/U with the
naming scheme that `MiloMoEMethod.create_weights` expects.

WARNING: skeleton only — the per-layer dump loop is left as TODO.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import save_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_quant_config_block(
    *, weight_bits: int = 3, group_size: int = 64,
    has_compensator: bool = True, compensator_rank: int = 32,
) -> dict:
    return {
        "quant_method": "milo",
        "bits": weight_bits,
        "group_size": group_size,
        "has_zp": True,
        "has_compensator": has_compensator,
        "compensator_rank": compensator_rank,
        "modules_to_not_convert": ["lm_head"],
    }


def _expert_rail_keys(layer_idx: int, expert_idx: int, rail: str) -> dict:
    """Return the safetensors key naming for one rail of one expert."""
    base = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{rail}"
    return {
        "Wq_packed1": f"{base}.Wq_packed1",
        "Wq_packed2": f"{base}.Wq_packed2",
        "scales":     f"{base}.scales",
        "zeros":      f"{base}.zeros",
        "V":          f"{base}.V",
        "U":          f"{base}.U",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True,
                        help="Source MiLo HF checkpoint dir (e.g. Qwen3-MoE_w3).")
    parser.add_argument("--dst", required=True,
                        help="Output directory for the vLLM-compatible checkpoint.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-size-gb", type=float, default=5.0,
                        help="Approximate target size for each safetensors shard.")
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: load the MiLo model and run the patching pipeline so that
    # every routed-expert linear becomes a MiLo_Asymmetric_Linear with
    # populated Wq_packed1 / Wq_packed2 / scales / zeros / V / U.
    # ------------------------------------------------------------------
    print(f"[1/4] Loading MiLo model from {args.src}...", flush=True)
    # TODO(milo): import MiLo's loader and invoke it.  The minimum
    # equivalent of:
    #     from MiLo.models.hf.qwen3_moe import Qwen3MoEMiLo
    #     model = Qwen3MoEMiLo.from_compressed(args.src).to(args.device)
    # so that after this line `model.model.layers[L].mlp.experts[E].<rail>`
    # is a `MiLo_Asymmetric_Linear` with .Wq_packed1 etc.
    raise NotImplementedError(
        "TODO: load the source checkpoint with the MiLo HF loader."
    )

    # ------------------------------------------------------------------
    # Step 2: walk every MoE layer, dump per-rail buffers under the
    # vLLM-expected naming scheme.
    # ------------------------------------------------------------------
    print("[2/4] Dumping expert buffers...", flush=True)
    state_dict: dict[str, torch.Tensor] = {}

    base_model = model.model  # noqa: F821  (model defined above once Step 1 done)
    for layer_idx, layer in enumerate(base_model.layers):
        # TODO(milo): if this layer is a MoE layer, walk experts × rails:
        #     for e, expert in enumerate(layer.mlp.experts):
        #         for rail in ("gate_proj", "up_proj", "down_proj"):
        #             m = getattr(expert, rail)
        #             keys = _expert_rail_keys(layer_idx, e, rail)
        #             state_dict[keys["Wq_packed1"]] = m.Wq_packed1.cpu()
        #             ... etc
        # Otherwise (dense MLP layer): dump as raw fp16 with the
        # original HF key names.
        pass

    # Also dump non-MoE params (embeddings, attn projections, layer norms,
    # lm_head) verbatim.
    # TODO(milo): for name, param in model.state_dict().items(): ...

    # ------------------------------------------------------------------
    # Step 3: write safetensors shards.
    # ------------------------------------------------------------------
    print(f"[3/4] Writing safetensors to {args.dst}...", flush=True)
    # TODO(milo): split into shards of ~`shard_size_gb` GB each;
    # produce model.safetensors.index.json.
    save_file(state_dict, os.path.join(args.dst, "model.safetensors"))

    # ------------------------------------------------------------------
    # Step 4: write config.json with the milo quantization_config block,
    # plus copy tokenizer files verbatim from the source dir.
    # ------------------------------------------------------------------
    print("[4/4] Writing config.json + copying tokenizer...", flush=True)
    src_cfg = json.load(open(os.path.join(args.src, "config.json")))
    src_cfg["quantization_config"] = _build_quant_config_block()
    with open(os.path.join(args.dst, "config.json"), "w") as f:
        json.dump(src_cfg, f, indent=2)

    for fname in os.listdir(args.src):
        if fname.startswith(("tokenizer", "special_tokens_map", "vocab",
                             "merges", "added_tokens")):
            shutil.copy(os.path.join(args.src, fname),
                        os.path.join(args.dst, fname))

    print("Done.")


if __name__ == "__main__":
    main()
