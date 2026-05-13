"""Verify a single MiLo expert: load from vLLM safetensors ckpt,
run milo.mul_3bit_with_zeros, compare to MiLo native model output.

Usage:
    PYTHONPATH=/usr/local/app/milo CUDA_VISIBLE_DEVICES=6 \
    python verify_single_expert.py
"""
import torch
import json
import os
from safetensors.torch import load_file

CKPT = "/usr/local/app/models/Qwen1.5-MoE-A2.7B-vllm-int3"
SRC  = "/usr/local/app/models/Qwen1.5-Moe-A2.7B_w3s16d512"

device = torch.device("cuda")


def pick_tile(prob_n, prob_k):
    if prob_n % 256 == 0 and prob_k % 64 == 0: return 256, 64
    if prob_n % 128 == 0 and prob_k % 128 == 0: return 128, 128
    if prob_n % 64 == 0 and prob_k % 256 == 0: return 64, 256
    return -1, -1


def test_single_expert():
    """Load expert 0, layer 0, gate_proj from safetensors and run kernel."""
    import milo

    # --- Load from safetensors ---
    idx_path = os.path.join(CKPT, "model.safetensors.index.json")
    with open(idx_path) as f:
        idx = json.load(f)
    wm = idx["weight_map"]

    prefix = "model.layers.0.mlp.experts.0.gate_proj"
    keys = [k for k in wm if k.startswith(prefix)]
    print(f"Keys for {prefix}:")
    for k in sorted(keys):
        print(f"  {k} → {wm[k]}")

    # Load tensors
    tensors = {}
    files_needed = set(wm[k] for k in keys)
    for fname in files_needed:
        path = os.path.join(CKPT, fname)
        all_t = load_file(path)
        for k in keys:
            if k in all_t:
                tensors[k] = all_t[k]

    B1 = tensors[f"{prefix}.Wq_packed1"].to(device)
    B2 = tensors[f"{prefix}.Wq_packed2"].to(device)
    scales = tensors[f"{prefix}.scales"].to(device)
    zeros = tensors[f"{prefix}.zeros"].to(device)

    K, N = scales.shape[0] * 64, scales.shape[1]  # group_size=64
    print(f"\nExpert 0 gate_proj: K={K}, N={N}")
    print(f"  B1: {B1.shape} {B1.dtype}")
    print(f"  B2: {B2.shape} {B2.dtype}")
    print(f"  scales: {scales.shape} {scales.dtype}")
    print(f"  zeros: {zeros.shape} {zeros.dtype}")

    thread_n, thread_k = pick_tile(N, K)
    print(f"  tile: thread_n={thread_n}, thread_k={thread_k}")

    workspace = torch.zeros(N // 128 * 16, dtype=torch.int32, device=device)

    # --- Run kernel ---
    torch.manual_seed(42)
    x = torch.randn(4, K, dtype=torch.float16, device=device) * 0.1
    out = torch.empty(4, N, dtype=torch.float16, device=device)

    milo.mul_3bit_with_zeros(
        x, B1, B2, out, scales, zeros, workspace,
        thread_k=thread_k, thread_n=thread_n,
    )
    print(f"\n  vLLM ckpt kernel output: mean={out.mean().item():.6f}, "
          f"std={out.std().item():.6f}, abs_max={out.abs().max().item():.6f}")

    return x, out


def test_native_expert(x_input):
    """Load same expert from MiLo native, patch to Marlin, run, compare."""
    from MiLo.core.quantize import MiLoLinear
    from MiLo.models.hf.qwen2_moe import Qwen2MoEMiLo
    from MiLo.backends.milo import (
        MiLo_Asymmetric_Linear, patch_hqq_to_milo_asymmetric)

    print("\n--- Loading MiLo native model ---")
    model = Qwen2MoEMiLo.from_compressed(SRC, device="cuda")
    model.eval()

    # Get layer 0, expert 0, gate_proj
    expert = model.model.layers[0].mlp.experts[0]
    gate_hqq = expert.gate_proj

    # Patch to Marlin
    gate_milo = patch_hqq_to_milo_asymmetric(gate_hqq, None)

    print(f"  Native B1: {gate_milo.Wq_packed1.shape}")
    print(f"  Native B2: {gate_milo.Wq_packed2.shape}")
    print(f"  Native scales: {gate_milo.scales.shape}")

    # Compare weights byte-for-byte
    from safetensors.torch import load_file
    import json
    idx_path = os.path.join(CKPT, "model.safetensors.index.json")
    with open(idx_path) as f:
        idx = json.load(f)
    prefix = "model.layers.0.mlp.experts.0.gate_proj"
    fname = idx["weight_map"][f"{prefix}.Wq_packed1"]
    all_t = load_file(os.path.join(CKPT, fname))
    vllm_B1 = all_t[f"{prefix}.Wq_packed1"].to("cuda")

    match_B1 = torch.equal(vllm_B1, gate_milo.Wq_packed1)
    print(f"\n  B1 byte-exact match: {match_B1}")
    if not match_B1:
        diff = (vllm_B1.int() - gate_milo.Wq_packed1.int()).abs()
        print(f"  B1 max diff: {diff.max().item()}, nonzero: {diff.nonzero().shape[0]}")

    # Run native forward
    out_native = gate_milo.matmul(x_input)
    print(f"  Native kernel output: mean={out_native.mean().item():.6f}, "
          f"std={out_native.std().item():.6f}, abs_max={out_native.abs().max().item():.6f}")

    return out_native


if __name__ == "__main__":
    print("=" * 60)
    print("Step 1: Test single expert from vLLM safetensors ckpt")
    print("=" * 60)
    x, out_vllm = test_single_expert()

    print("\n" + "=" * 60)
    print("Step 2: Compare with MiLo native (byte-exact + numerical)")
    print("=" * 60)
    out_native = test_native_expert(x)

    print("\n" + "=" * 60)
    print("Step 3: Numerical comparison")
    print("=" * 60)
    diff = (out_vllm - out_native).abs()
    print(f"  max abs diff: {diff.max().item():.6e}")
    print(f"  mean abs diff: {diff.mean().item():.6e}")
    print(f"  PASS: {diff.max().item() < 1e-3}")
