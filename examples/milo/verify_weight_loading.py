"""Verify that FusedMoE weight_loader correctly fills the stacked w13 buffer.

Loads the ckpt directly, then checks if w13_Wq_packed1[expert=0, :, :I]
(gate half) matches the raw experts.0.gate_proj.Wq_packed1 from safetensors.

Usage:
    PYTHONPATH=/usr/local/app/milo CUDA_VISIBLE_DEVICES=6 \
    python verify_weight_loading.py
"""
import json, os, torch
from safetensors.torch import load_file

CKPT = "/usr/local/app/models/Qwen1.5-MoE-A2.7B-vllm-int3"


def main():
    idx_path = os.path.join(CKPT, "model.safetensors.index.json")
    with open(idx_path) as f:
        idx = json.load(f)
    wm = idx["weight_map"]

    # Load raw expert 0 gate_proj tensors from ckpt
    prefix = "model.layers.0.mlp.experts.0.gate_proj"
    raw = {}
    files_needed = set(wm[k] for k in wm if k.startswith(prefix))
    for fname in files_needed:
        all_t = load_file(os.path.join(CKPT, fname))
        for k in all_t:
            if k.startswith(prefix):
                raw[k.split(".")[-1]] = all_t[k]

    print("Raw ckpt expert 0 gate_proj:")
    for name, t in sorted(raw.items()):
        print(f"  {name}: {t.shape} {t.dtype}")

    # Now simulate what vLLM's FusedMoE weight_loader does:
    # 1. Create w13_Wq_packed1 with shape [E, K/16, 2*I]
    # 2. For gate_proj (shard_id="w1"), expert_data = param[expert_id]
    #    shard_dim = SHARD_ID_TO_SHARDED_DIM["w1"] = 0
    #    is_transposed=True → shard_dim = int(not 0) = 1
    #    shard_size = expert_data.shape[shard_dim] // 2  (is_act_and_mul)
    #    For w1: expert_data.narrow(shard_dim=1, 0, shard_size)
    #    Then loaded_weight is narrowed for TP: narrow(shard_dim=1, tp_rank*shard_size, shard_size)
    #    Copy.

    K = 2048
    I = 1408
    E = 60

    # Simulate w13_Wq_packed1
    w13_B1 = torch.zeros(E, K // 16, 2 * I, dtype=torch.int32)

    # Load gate_proj.Wq_packed1 (shard_id=w1)
    loaded = raw["Wq_packed1"]  # [128, 1408]
    print(f"\nLoaded gate B1: {loaded.shape}")

    # weight_loader logic for w1 with is_transposed=True:
    # shard_dim_orig = SHARD_ID_TO_SHARDED_DIM["w1"] = 0
    # is_transposed → shard_dim = 1
    # full_load = False (loaded_weight is 2D, not 3D)
    # expert_data = w13_B1[0]  → [128, 2816]
    # shard_size = expert_data.shape[1] // 2 = 1408  (is_act_and_mul)
    # w1 → narrow(1, 0, 1408)
    # Then loaded_weight.narrow(1, tp_rank*shard_size, shard_size) = narrow(1, 0, 1408)

    expert_data = w13_B1[0]
    shard_dim = 1
    shard_size = expert_data.shape[shard_dim] // 2  # 2816//2 = 1408

    # gate (w1): first half
    dst = expert_data.narrow(shard_dim, 0, shard_size)
    # TP=1, tp_rank=0
    src = loaded.narrow(shard_dim, 0, shard_size)
    print(f"dst (gate half): {dst.shape}, src: {src.shape}")
    assert dst.shape == src.shape, f"Shape mismatch: {dst.shape} vs {src.shape}"
    dst.copy_(src)

    # Check
    gate_B1_from_w13 = w13_B1[0, :, :I]  # should equal raw Wq_packed1
    match = torch.equal(gate_B1_from_w13, raw["Wq_packed1"])
    print(f"\nSimulated gate B1 matches raw: {match}")

    if not match:
        diff = (gate_B1_from_w13.int() - raw["Wq_packed1"].int()).abs()
        print(f"  max diff: {diff.max().item()}, nonzero: {diff.nonzero().shape[0]}")
    else:
        print("  weight_loader simulation: CORRECT for B1 gate path")

    # Now check B2 (packed_factor=2 issue)
    # w13_Wq_packed2: [E, K/16, I]  (because N/2 where N=2*I → I)
    w13_B2 = torch.zeros(E, K // 16, I, dtype=torch.int32)
    loaded_B2 = raw["Wq_packed2"]  # [128, 704]
    print(f"\nLoaded gate B2: {loaded_B2.shape}")

    expert_data_B2 = w13_B2[0]  # [128, I=1408]
    # For B2: shard_size should be I // 2 = 704 (because B2 is N/2 packed)
    # But weight_loader sees shard_size = expert_data_B2.shape[1] // 2 = 704
    shard_size_B2 = expert_data_B2.shape[1] // 2  # 1408//2 = 704
    dst_B2 = expert_data_B2.narrow(1, 0, shard_size_B2)
    src_B2 = loaded_B2.narrow(1, 0, shard_size_B2)
    print(f"dst B2 (gate half): {dst_B2.shape}, src B2: {src_B2.shape}")
    dst_B2.copy_(src_B2)

    gate_B2_from_w13 = w13_B2[0, :, :I//2]
    match_B2 = torch.equal(gate_B2_from_w13, raw["Wq_packed2"])
    print(f"Simulated gate B2 matches raw: {match_B2}")

    # Also check scales
    w13_scales = torch.zeros(E, K // 64, 2 * I, dtype=torch.float16)
    loaded_scales = raw["scales"]  # [32, 1408]
    expert_data_s = w13_scales[0]  # [32, 2816]
    shard_size_s = expert_data_s.shape[1] // 2  # 1408
    dst_s = expert_data_s.narrow(1, 0, shard_size_s)
    src_s = loaded_scales.narrow(1, 0, shard_size_s)
    dst_s.copy_(src_s)
    gate_scales_from_w13 = w13_scales[0, :, :I]
    match_s = torch.equal(gate_scales_from_w13, raw["scales"])
    print(f"Simulated gate scales matches raw: {match_s}")

    print("\n=== Summary ===")
    print(f"  B1 gate:   {'PASS' if match else 'FAIL'}")
    print(f"  B2 gate:   {'PASS' if match_B2 else 'FAIL'}")
    print(f"  scales gate: {'PASS' if match_s else 'FAIL'}")


if __name__ == "__main__":
    main()
