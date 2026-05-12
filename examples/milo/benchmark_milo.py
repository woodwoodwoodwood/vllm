"""End-to-end throughput benchmark for MiLo INT3 + vLLM.

Compares (when checkpoints are available):
    * MiLo HF runner       — reference (no scheduler / no PagedAttention)
    * vLLM + MiLo backend  — production path (--quantization milo)
    * vLLM + bf16          — unquantized upper bound

Usage:
    python benchmark_milo.py --model /path/to/qwen3-moe-int3-milo-ckpt
"""
from __future__ import annotations

import argparse
import time

from vllm import LLM, SamplingParams


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="Path to the vLLM-format INT3 checkpoint.")
    p.add_argument("--prompts", type=int, default=64)
    p.add_argument("--input-len", type=int, default=128)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--tp", type=int, default=1)
    args = p.parse_args()

    print(f"Loading vLLM with model={args.model} quantization=milo "
          f"tp={args.tp}...", flush=True)
    llm = LLM(
        model=args.model,
        quantization="milo",
        dtype=args.dtype,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=0.9,
        enforce_eager=False,
    )

    # Synthetic prompts of fixed input length.
    prompts = [f"The quick brown fox " * (args.input_len // 4)
               for _ in range(args.prompts)]
    sampling = SamplingParams(temperature=0,
                              top_p=1.0,
                              max_tokens=args.output_len,
                              ignore_eos=True)

    # Warmup
    print("Warmup (1 prompt)...", flush=True)
    _ = llm.generate(prompts[:1], sampling)

    # Benchmark
    print(f"Benchmark ({args.prompts} prompts × "
          f"{args.input_len} in / {args.output_len} out)...", flush=True)
    t0 = time.time()
    outputs = llm.generate(prompts, sampling)
    t1 = time.time()

    total_in  = sum(len(o.prompt_token_ids) for o in outputs)
    total_out = sum(len(o.outputs[0].token_ids) for o in outputs)
    dt = t1 - t0

    print(f"\n--- Results ---")
    print(f"  wall time         : {dt:.2f} s")
    print(f"  input tokens      : {total_in}")
    print(f"  output tokens     : {total_out}")
    print(f"  output throughput : {total_out / dt:.1f} tok/s")
    print(f"  total throughput  : {(total_in + total_out) / dt:.1f} tok/s")


if __name__ == "__main__":
    main()
