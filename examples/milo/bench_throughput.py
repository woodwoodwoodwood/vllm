"""Simple throughput benchmark for MiLo INT3+comp on vLLM.

Usage:
    PYTHONPATH=/usr/local/app/milo CUDA_VISIBLE_DEVICES=6 \
    python bench_throughput.py [--num-prompts 100] [--input-len 128] [--output-len 128]
"""
import argparse
import time

import torch
from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/usr/local/app/models/Qwen1.5-MoE-A2.7B-vllm-int3")
    parser.add_argument("--quantization", default="milo",
                        help="Quantization method, e.g. 'milo' or None for fp16")
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=None,
                        help="If set, sweep these batch sizes instead of --num-prompts")
    args = parser.parse_args()

    max_model_len = args.input_len + args.output_len + 64  # headroom
    print(f"Model: {args.model}")
    print(f"Input len: {args.input_len}, Output len: {args.output_len}")
    print(f"max_model_len: {max_model_len}")
    print()

    llm_kwargs = dict(
        model=args.model,
        dtype="float16",
        max_model_len=max_model_len,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
    )
    if args.quantization and args.quantization.lower() != "none":
        llm_kwargs["quantization"] = args.quantization
    llm = LLM(**llm_kwargs)

    tokenizer = llm.get_tokenizer()
    # Build fixed-length prompts by repeating a token
    dummy_ids = [1] * args.input_len
    dummy_prompt = tokenizer.decode(dummy_ids)

    params = SamplingParams(
        temperature=0,
        max_tokens=args.output_len,
        ignore_eos=True,
    )

    batch_sizes = args.batch_sizes or [args.num_prompts]

    for bs in batch_sizes:
        prompts = [dummy_prompt] * bs

        # Warmup
        llm.generate(prompts[:min(2, bs)], params)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        outputs = llm.generate(prompts, params)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        total_in = sum(len(o.prompt_token_ids) for o in outputs)
        total_out = sum(len(o.outputs[0].token_ids) for o in outputs)

        print(f"--- Batch size: {bs} ---")
        print(f"  Time:              {elapsed:.2f} s")
        print(f"  Requests/s:        {bs / elapsed:.2f}")
        print(f"  Input tokens:      {total_in}")
        print(f"  Output tokens:     {total_out}")
        print(f"  Output tokens/s:   {total_out / elapsed:.2f}")
        print(f"  Total tokens/s:    {(total_in + total_out) / elapsed:.2f}")
        print()


if __name__ == "__main__":
    main()
