# MiLo INT3 + Compensator quantization for vLLM

This directory hosts the integration glue for running MiLo INT3-quantized
MoE models inside vLLM.

## Status

| Component | Status | Notes |
|---|---|---|
| `MiloConfig`            | ✅ done    | Registered as `--quantization milo`. |
| `MiloLinearMethod`      | 🟡 stub   | Falls back to unquantized; use `modules_to_not_convert` for now. |
| `MiloMoEMethod` weights | ❌ TODO   | `create_weights` + `process_weights_after_loading`. |
| `MiloMoEMethod.apply`   | ❌ TODO   | Sorted dispatch + INT3 kernel + compensator. |
| Checkpoint converter    | ❌ TODO   | `convert_milo_to_vllm.py`. |
| End-to-end test         | ❌ TODO   | Smoke test on Qwen3-MoE-30B-A3B. |

## Prerequisites

```bash
# 1. Install vLLM in editable mode (this fork).
cd ~/github/vllm
pip install -e . --no-build-isolation -v

# 2. Install MiLo CUDA kernel.
cd ~/MiLo/MiLo/kernels
python setup_milo.py install

# 3. Add MiLo Python package to PYTHONPATH.
export PYTHONPATH=~/MiLo:$PYTHONPATH

# 4. Sanity-check.
python -c "import milo_cuda; from MiLo.fused_moe.milo_moe import milo_int3_moe; print('ok')"
```

## Checkpoint format

vLLM expects HF-style weights with `config.json`:

```json
{
  ...
  "quantization_config": {
    "quant_method":     "milo",
    "bits":              3,
    "group_size":        64,
    "has_zp":            true,
    "has_compensator":   true,
    "compensator_rank":  32,
    "modules_to_not_convert": ["lm_head"]
  }
}
```

And per-expert per-rail safetensors keys (TODO: finalise schema):

```
model.layers.{L}.mlp.experts.{E}.gate_proj.Wq_packed1   int32   [K/16, N]
model.layers.{L}.mlp.experts.{E}.gate_proj.Wq_packed2   int32   [K/16, N/2]
model.layers.{L}.mlp.experts.{E}.gate_proj.scales       fp16    [K/gs, N]
model.layers.{L}.mlp.experts.{E}.gate_proj.zeros        fp16    [K/gs, N]
model.layers.{L}.mlp.experts.{E}.gate_proj.V            fp16    [K, rank]
model.layers.{L}.mlp.experts.{E}.gate_proj.U            fp16    [rank, N]
... same for up_proj and down_proj
```

## Running

```bash
vllm serve /path/to/qwen3-moe-int3-milo-ckpt \
    --quantization milo \
    --dtype float16 \
    --tensor-parallel-size 1   # TP > 1 not yet supported (see TODO)
```

## TODO list

See `vllm/model_executor/layers/quantization/milo.py` for in-line `TODO`
markers.  Implementation plan, in dependency order:

1. Write `convert_milo_to_vllm.py` so we can produce a real checkpoint.
2. Implement `MiloMoEMethod.create_weights` (mirrors `AWQMarlinMoEMethod`).
3. Implement `process_weights_after_loading` to build `MiLoMoERail` + V/U.
4. Implement `apply` (sorted dispatch + `milo_int3_moe` + `compensator_batched`).
5. Implement `MiloLinearMethod` properly (stop falling back to unquantized).
6. Add tensor-parallel support (kernel needs `thread_n` smaller than 64
   or we split N differently).
