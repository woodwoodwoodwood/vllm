"""Minimal e2e test: load MiLo checkpoint via vLLM's offline LLM API.

Usage:
    kill $(lsof -t -i:8001) 2>/dev/null
    PYTHONPATH=/usr/local/app/milo CUDA_VISIBLE_DEVICES=6 \
    python verify_e2e_simple.py
"""
from vllm import LLM, SamplingParams

llm = LLM(
    model="/usr/local/app/models/Qwen1.5-MoE-A2.7B-vllm-int3",
    quantization="milo",
    dtype="float16",
    max_model_len=512,
    gpu_memory_utilization=0.5,
    enforce_eager=True,
)

prompts = [
    "The capital of France is",
    "1 + 1 =",
    "Hello, my name is",
]
params = SamplingParams(temperature=0, max_tokens=30)

outputs = llm.generate(prompts, params)
for o in outputs:
    print(f"Prompt: {o.prompt!r}")
    print(f"Output: {o.outputs[0].text!r}")
    print(f"Tokens: {o.outputs[0].token_ids[:10]}")
    print()
