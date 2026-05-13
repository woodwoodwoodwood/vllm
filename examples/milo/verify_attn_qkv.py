"""Verify QKV stacking: does concat of q/k/v Wq_packed1 along N produce
correct kernel output vs running q/k/v separately?

Usage:
    PYTHONPATH=/usr/local/app/milo CUDA_VISIBLE_DEVICES=6 python verify_attn_qkv.py
"""
import torch, json, os, milo
from safetensors.torch import load_file

CKPT = "/usr/local/app/models/Qwen1.5-MoE-A2.7B-vllm-int3"
device = torch.device("cuda")


def pick_tile(prob_n, prob_k):
    if prob_n % 256 == 0 and prob_k % 64 == 0: return 256, 64
    if prob_n % 128 == 0 and prob_k % 128 == 0: return 128, 128
    if prob_n % 64 == 0 and prob_k % 256 == 0: return 64, 256
    return -1, -1


def load_proj(all_tensors, proj_name):
    """Load B1/B2/scales/zeros for one projection."""
    prefix = f"model.layers.0.self_attn.{proj_name}"
    return {
        "B1": all_tensors[f"{prefix}.Wq_packed1"].to(device),
        "B2": all_tensors[f"{prefix}.Wq_packed2"].to(device),
        "scales": all_tensors[f"{prefix}.scales"].to(device),
        "zeros": all_tensors[f"{prefix}.zeros"].to(device),
    }


def run_single(x, proj, label):
    """Run kernel on a single proj."""
    B1, B2, s, z = proj["B1"], proj["B2"], proj["scales"], proj["zeros"]
    K = s.shape[0] * 64
    N = s.shape[1]
    tn, tk = pick_tile(N, K)
    ws = torch.zeros(N // 128 * 16, dtype=torch.int32, device=device)
    out = torch.empty(x.shape[0], N, dtype=torch.float16, device=device)
    milo.mul_3bit_with_zeros(x, B1, B2, out, s, z, ws, thread_k=tk, thread_n=tn)
    print(f"  {label}: shape={out.shape}, mean={out.mean().item():.6f}, "
          f"std={out.std().item():.6f}")
    return out


def run_fused(x, q, k, v):
    """Concat q/k/v along N dim and run kernel on the fused tensor."""
    B1 = torch.cat([q["B1"], k["B1"], v["B1"]], dim=1)
    B2 = torch.cat([q["B2"], k["B2"], v["B2"]], dim=1)
    s  = torch.cat([q["scales"], k["scales"], v["scales"]], dim=1)
    z  = torch.cat([q["zeros"], k["zeros"], v["zeros"]], dim=1)
    K = s.shape[0] * 64
    N = s.shape[1]
    tn, tk = pick_tile(N, K)
    ws = torch.zeros(N // 128 * 16, dtype=torch.int32, device=device)
    out = torch.empty(x.shape[0], N, dtype=torch.float16, device=device)
    print(f"  Fused QKV: B1={B1.shape}, N={N}, tile=({tn},{tk})")
    milo.mul_3bit_with_zeros(x, B1, B2, out, s, z, ws, thread_k=tk, thread_n=tn)
    print(f"  Fused: shape={out.shape}, mean={out.mean().item():.6f}, "
          f"std={out.std().item():.6f}")
    return out


def main():
    # Load all tensors from both shards
    idx_path = os.path.join(CKPT, "model.safetensors.index.json")
    with open(idx_path) as f:
        idx = json.load(f)

    all_tensors = {}
    for fname in set(idx["weight_map"].values()):
        all_tensors.update(load_file(os.path.join(CKPT, fname)))

    q = load_proj(all_tensors, "q_proj")
    k = load_proj(all_tensors, "k_proj")
    v = load_proj(all_tensors, "v_proj")

    torch.manual_seed(42)
    x = torch.randn(4, 2048, dtype=torch.float16, device=device) * 0.1

    print("=== Separate q/k/v kernel runs ===")
    out_q = run_single(x, q, "q_proj")
    out_k = run_single(x, k, "k_proj")
    out_v = run_single(x, v, "v_proj")
    out_separate = torch.cat([out_q, out_k, out_v], dim=1)

    print("\n=== Fused QKV (concat along N) ===")
    out_fused = run_fused(x, q, k, v)

    print("\n=== Comparison ===")
    diff = (out_separate - out_fused).abs()
    print(f"  max abs diff: {diff.max().item():.6e}")
    print(f"  mean abs diff: {diff.mean().item():.6e}")

    # Check q portion
    N_q = out_q.shape[1]
    diff_q = (out_q - out_fused[:, :N_q]).abs()
    print(f"  q portion max diff: {diff_q.max().item():.6e}")

    if diff.max().item() < 1e-3:
        print("  PASS: Fused QKV == separate runs")
    else:
        print("  FAIL: Fused QKV differs from separate runs!")
        print("  This means Marlin prepack N-concat is NOT safe for this model.")


if __name__ == "__main__":
    main()
