"""Compare MiLo vLLM output vs MiLo HF native runtime.

Usage:
    PYTHONPATH=/usr/local/app/milo CUDA_VISIBLE_DEVICES=6 \
    python verify_correctness.py
"""
from __future__ import annotations

import torch
import numpy as np

# ============================================================
# Config
# ============================================================
VLLM_MODEL = "/usr/local/app/models/Qwen1.5-MoE-A2.7B-vllm-int3"
HF_QUANT_CKPT = "/usr/local/app/models/Qwen1.5-Moe-A2.7B_w3s16d512"
HF_TOKENIZER = "/data1/models/Qwen1.5-Moe-A2.7B-18188"

PROMPTS = [
    "The capital of France is",
    "1 + 1 =",
    "Hello, my name is",
    "The meaning of life is",
]
MAX_NEW_TOKENS = 30


def run_vllm():
    """Run inference via vLLM offline API."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=VLLM_MODEL,
        quantization="milo",
        dtype="float16",
        max_model_len=512,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
    )
    params = SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS)
    outputs = llm.generate(PROMPTS, params)
    results = []
    for o in outputs:
        results.append({
            "prompt": o.prompt,
            "text": o.outputs[0].text,
            "tokens": list(o.outputs[0].token_ids),
        })
    # Free GPU memory
    del llm
    torch.cuda.empty_cache()
    return results


def run_hf_native():
    """Run inference via MiLo HF native runtime."""
    import MiLo.core.quantize  # noqa: F401 (registers backend)
    from MiLo.models.hf.qwen2_moe import Qwen2MoEMiLo
    from MiLo.engine.hf import AutoTokenizer

    model = Qwen2MoEMiLo.from_compressed(HF_QUANT_CKPT)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(HF_TOKENIZER,
                                              trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_ids = gen_ids[0, inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        results.append({
            "prompt": prompt,
            "text": text,
            "tokens": new_ids.tolist(),
        })

    del model
    torch.cuda.empty_cache()
    return results


def compare(vllm_results, hf_results):
    """Compare and print results."""
    print("\n" + "=" * 70)
    print("COMPARISON: vLLM vs HF Native")
    print("=" * 70)

    all_match = True
    for i, (v, h) in enumerate(zip(vllm_results, hf_results)):
        prompt = v["prompt"]
        v_tokens = v["tokens"][:10]
        h_tokens = h["tokens"][:10]
        match = v_tokens == h_tokens

        status = "MATCH" if match else "MISMATCH"
        if not match:
            all_match = False

        print(f"\n[{i+1}] Prompt: {prompt!r}  [{status}]")
        print(f"    vLLM:   {v['text'][:80]!r}")
        print(f"    HF:     {h['text'][:80]!r}")
        print(f"    vLLM tokens (first 10): {v_tokens}")
        print(f"    HF   tokens (first 10): {h_tokens}")

        if not match:
            # Find first diverging position
            for j in range(min(len(v_tokens), len(h_tokens))):
                if v_tokens[j] != h_tokens[j]:
                    print(f"    First mismatch at position {j}: "
                          f"vLLM={v_tokens[j]} vs HF={h_tokens[j]}")
                    break

    print("\n" + "=" * 70)
    if all_match:
        print("RESULT: ALL PROMPTS MATCH - MiLo vLLM integration is correct!")
    else:
        print("RESULT: MISMATCH DETECTED - check compensator/weight loading.")
    print("=" * 70)
    return all_match


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-only", action="store_true",
                        help="Only run vLLM path (skip HF comparison)")
    parser.add_argument("--hf-only", action="store_true",
                        help="Only run HF native path")
    args = parser.parse_args()

    if args.vllm_only:
        print("Running vLLM only...")
        results = run_vllm()
        for r in results:
            print(f"Prompt: {r['prompt']!r}")
            print(f"Output: {r['text']!r}")
            print(f"Tokens: {r['tokens'][:10]}")
            print()
    elif args.hf_only:
        print("Running HF native only...")
        results = run_hf_native()
        for r in results:
            print(f"Prompt: {r['prompt']!r}")
            print(f"Output: {r['text']!r}")
            print(f"Tokens: {r['tokens'][:10]}")
            print()
    else:
        print("Step 1/2: Running HF native runtime...")
        hf_results = run_hf_native()

        print("\nStep 2/2: Running vLLM...")
        vllm_results = run_vllm()

        compare(vllm_results, hf_results)
