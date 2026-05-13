"""Milestone 1: sanity-check that MiLo HF runtime works on this checkpoint.

Goal: prove that
    Qwen2MoEMiLo.from_compressed(<ckpt>) → tokenize → forward → decode
runs end-to-end and produces sensible output, BEFORE we start writing the
vLLM-side converter / methods that depend on the same loader.

If this script fails, vLLM integration cannot work either — fix MiLo first.

Usage:
    python sanity_check_milo_hf.py
    # or with custom paths:
    python sanity_check_milo_hf.py \
        --quant-ckpt /usr/local/app/models/Qwen1.5-Moe-A2.7B_w3s16d512 \
        --tokenizer  /data1/models/Qwen1.5-Moe-A2.7B-18188 \
        --prompt     "Hello, my name is"
"""
from __future__ import annotations

import argparse
import sys
import time

import torch


DEFAULT_QUANT = "/usr/local/app/models/Qwen1.5-Moe-A2.7B_w3s16d512"
DEFAULT_TOK   = "/data1/models/Qwen1.5-Moe-A2.7B-18188"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quant-ckpt", default=DEFAULT_QUANT,
                   help="MiLo native quantized checkpoint dir.")
    p.add_argument("--tokenizer",  default=DEFAULT_TOK,
                   help="Original (unquantized) HF model dir for tokenizer.")
    p.add_argument("--prompt",     default="Hello, my name is")
    p.add_argument("--max-new",    type=int, default=32)
    p.add_argument("--no-generate", action="store_true",
                   help="Only verify load + single forward; skip .generate().")
    args = p.parse_args()

    print("=" * 70)
    print("Milestone-1 sanity check: MiLo HF runtime end-to-end")
    print("=" * 70)
    print(f"  quant ckpt : {args.quant_ckpt}")
    print(f"  tokenizer  : {args.tokenizer}")
    print(f"  prompt     : {args.prompt!r}")
    print()

    # Imports are deferred so the user sees a clean error if the env is bad.
    print("[1/4] Importing MiLo HF runtime ...", flush=True)
    t0 = time.time()
    try:
        from MiLo.core.quantize import *  # noqa: F401, F403  (registers backend)
        from MiLo.models.hf.qwen2_moe import Qwen2MoEMiLo
        from MiLo.engine.hf import AutoTokenizer
    except ImportError as e:
        print(f"  ImportError: {e}")
        print("  Make sure /usr/local/app/milo is on PYTHONPATH and milo_cuda "
              "is installed.")
        sys.exit(1)
    print(f"      ... done in {time.time() - t0:.1f}s")

    # ---- Load ----
    print("[2/4] Loading model via Qwen2MoEMiLo.from_compressed() ...", flush=True)
    t0 = time.time()
    model = Qwen2MoEMiLo.from_compressed(args.quant_ckpt)
    model.eval()
    print(f"      ... done in {time.time() - t0:.1f}s")

    # Quick structure sniff
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      total params (incl. compensator V/U dequantized to fp16): "
          f"{n_params / 1e6:.1f} M")

    # Sample a single quantized linear and confirm we have the expected attrs.
    sample_module = None
    for name, m in model.named_modules():
        if hasattr(m, "Wq_packed1") and hasattr(m, "scales"):
            sample_module = (name, m)
            break
    if sample_module is None:
        print("  WARN: could not find any module exposing Wq_packed1; the "
              "model may not have been patched into MiLo_Asymmetric_Linear "
              "form yet.  This may or may not be a problem.")
    else:
        name, m = sample_module
        print(f"      sampled quantized linear: {name}")
        print(f"        Wq_packed1.shape = {tuple(m.Wq_packed1.shape)}  "
              f"dtype={m.Wq_packed1.dtype}")
        print(f"        scales.shape     = {tuple(m.scales.shape)}  "
              f"dtype={m.scales.dtype}")
        if getattr(m, "U", None) is not None:
            print(f"        U.shape          = {tuple(m.U.shape)}  "
                  f"dtype={m.U.dtype}")
            print(f"        V.shape          = {tuple(m.V.shape)}  "
                  f"dtype={m.V.dtype}")
        print(f"        bias             = "
              f"{'None' if m.bias is None else tuple(m.bias.shape)}")

    # ---- Tokenizer ----
    print("[3/4] Loading tokenizer ...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer,
                                              trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"      ... done in {time.time() - t0:.1f}s")

    # ---- Forward / generate ----
    print("[4/4] Running forward + generate ...", flush=True)
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    print(f"      input_ids.shape = {tuple(inputs['input_ids'].shape)}")

    # Step A: single forward to verify the kernel chain works.
    t0 = time.time()
    with torch.no_grad():
        out = model(**inputs)
    torch.cuda.synchronize()
    fwd_ms = (time.time() - t0) * 1000
    logits = out.logits if hasattr(out, "logits") else out[0]
    print(f"      single forward: {fwd_ms:.1f} ms  "
          f"logits.shape={tuple(logits.shape)}  "
          f"dtype={logits.dtype}")

    # Step B (optional): autoregressive decoding.
    if args.no_generate:
        print("      (skipping .generate())")
    else:
        t0 = time.time()
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new,
                do_sample=False,           # greedy → reproducible
                pad_token_id=tokenizer.pad_token_id,
            )
        torch.cuda.synchronize()
        gen_s = time.time() - t0
        full_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        new_tokens = gen_ids.shape[1] - inputs["input_ids"].shape[1]
        print(f"      generate: {gen_s:.2f} s for {new_tokens} new tokens "
              f"({new_tokens / gen_s:.1f} tok/s)")
        print()
        print(f"--- decoded output ---")
        print(full_text)
        print(f"--- end ---")

    print()
    print("=" * 70)
    print("MILESTONE-1 PASSED")
    print("MiLo HF runtime works end-to-end on this checkpoint.")
    print("Next: run convert_milo_to_vllm.py to produce a vLLM-readable ckpt.")
    print("=" * 70)


if __name__ == "__main__":
    main()
