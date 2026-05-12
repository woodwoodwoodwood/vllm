"""MiLo INT3 + compensator quantization backend for vLLM.

This is a thin glue layer that wires the MiLo INT3 grouped-MoE CUDA kernel
(`milo_cuda.mul_3bit_moe_with_zeros`) and the batched compensator path
(`MiLo.fused_moe.compensator_batched`) into vLLM's quantization framework.

External dependencies (must be installed in the same Python env as vLLM):
    pip install -e .            # in /path/to/MiLo/MiLo/kernels  (builds milo_cuda)
    PYTHONPATH includes /path/to/MiLo

Currently supports:
    * Linear layers      (q_proj / k_proj / v_proj / o_proj / lm_head etc.)
    * FusedMoE layers    (Qwen2-MoE / Qwen3-MoE / DeepSeek-V2)

The kernel itself is documented in `MiLo/kernels/milo/milo_cuda_moe_kernel.cu`.
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy-import MiLo runtime — fail fast at construction time if missing,
# but don't crash on `import vllm.model_executor.layers.quantization.milo`.
# ---------------------------------------------------------------------------
def _import_milo():
    """Return (milo_cuda, MiLo.fused_moe.milo_moe, MiLo.fused_moe.compensator_batched)."""
    try:
        import milo_cuda  # noqa: F401
        from MiLo.fused_moe import milo_moe as _milo_moe
        from MiLo.fused_moe import compensator_batched as _compbatch
    except ImportError as e:
        raise ImportError(
            "MiLo runtime is not installed.  Install it with:\n"
            "    cd /path/to/MiLo/MiLo/kernels && python setup_milo.py install\n"
            "and ensure /path/to/MiLo is on PYTHONPATH."
        ) from e
    return milo_cuda, _milo_moe, _compbatch


# ===========================================================================
#  MiLo quantization config
# ===========================================================================
class MiloConfig(QuantizationConfig):
    """MiLo INT3 + (low-rank) compensator quantization config.

    Expected `config.json["quantization_config"]` schema:

        {
            "quant_method":   "milo",
            "bits":           3,
            "group_size":     64,
            "has_zp":         true,
            "has_compensator": true,
            "compensator_rank": 32,
            "modules_to_not_convert": ["lm_head"]   # optional
        }
    """

    SUPPORTED_BITS = (3,)
    SUPPORTED_GROUP_SIZES = (64,)

    def __init__(
        self,
        weight_bits: int = 3,
        group_size: int = 64,
        has_zp: bool = True,
        has_compensator: bool = True,
        compensator_rank: int = 0,
        modules_to_not_convert: Optional[list[str]] = None,
        lm_head_quantized: bool = False,
    ):
        super().__init__()
        if weight_bits not in self.SUPPORTED_BITS:
            raise ValueError(
                f"MiLo only supports {self.SUPPORTED_BITS}-bit weights, "
                f"got {weight_bits}."
            )
        if group_size not in self.SUPPORTED_GROUP_SIZES:
            raise ValueError(
                f"MiLo only supports group_size in "
                f"{self.SUPPORTED_GROUP_SIZES}, got {group_size}."
            )
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.has_zp = has_zp
        self.has_compensator = has_compensator
        self.compensator_rank = compensator_rank
        self.modules_to_not_convert = modules_to_not_convert or []
        self.lm_head_quantized = lm_head_quantized

    def __repr__(self) -> str:
        return (
            f"MiloConfig(bits={self.weight_bits}, group_size={self.group_size}, "
            f"has_zp={self.has_zp}, has_compensator={self.has_compensator}, "
            f"compensator_rank={self.compensator_rank})"
        )

    # ----- vLLM QuantizationConfig contract -----
    @classmethod
    def get_name(cls) -> "QuantizationMethods":
        return "milo"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # MiLo CUDA kernel is currently only built for sm_90a (H20 / H100).
        return 90

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MiloConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        has_zp = cls.get_from_keys_or(config, ["has_zp"], default=True)
        has_compensator = cls.get_from_keys_or(
            config, ["has_compensator"], default=True
        )
        compensator_rank = cls.get_from_keys_or(
            config, ["compensator_rank"], default=0
        )
        modules_to_not_convert = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], default=[]
        )
        lm_head_quantized = cls.get_from_keys_or(
            config, ["lm_head"], default=False
        )
        return cls(
            weight_bits=weight_bits,
            group_size=group_size,
            has_zp=has_zp,
            has_compensator=has_compensator,
            compensator_rank=compensator_rank,
            modules_to_not_convert=modules_to_not_convert,
            lm_head_quantized=lm_head_quantized,
        )

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> "QuantizationMethods | None":
        if hf_quant_cfg is None:
            return None
        quant_method = hf_quant_cfg.get("quant_method", "").lower()
        if quant_method == "milo":
            return cls.get_name()
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        # Skip blacklisted modules.
        for blacklisted in self.modules_to_not_convert:
            if blacklisted in prefix:
                return UnquantizedLinearMethod()

        if isinstance(layer, FusedMoE):
            return MiloMoEMethod(self, layer.moe_config)
        if isinstance(layer, LinearBase) or (
            isinstance(layer, ParallelLMHead) and self.lm_head_quantized
        ):
            return MiloLinearMethod(self)
        return None


# ===========================================================================
#  Linear method  (TODO: stub — falls back to unquantized for now)
# ===========================================================================
class MiloLinearMethod(LinearMethodBase):
    """Linear method for MiLo.

    Stage 0 / smoke-test mode: this is a stub that falls back to the
    unquantized path (bf16/fp16 PyTorch matmul).  This is acceptable for
    initial bring-up because the bottleneck of Qwen3-MoE is the MoE layer,
    not the dense linear projections.

    TODO: implement actual INT3 + compensator dense-linear matmul using
    `MiLo_Asymmetric_Linear.matmul()` semantics.
    """

    def __init__(self, quant_config: MiloConfig):
        self.quant_config = quant_config
        self._unquant = UnquantizedLinearMethod()
        logger.warning_once(
            "MiloLinearMethod is currently a stub (falls back to "
            "unquantized matmul).  This will OOM if applied to a "
            "weights-quantized checkpoint with INT3 buffers; for now "
            "list dense-linear modules in `modules_to_not_convert`."
        )

    def create_weights(self, layer, *args, **kwargs):
        return self._unquant.create_weights(layer, *args, **kwargs)

    def apply(self, layer, x, bias=None):
        return self._unquant.apply(layer, x, bias)


# ===========================================================================
#  Fused-MoE method  (the real work)
# ===========================================================================
class MiloMoEMethod(FusedMoEMethodBase):
    """Fused-MoE method backed by the MiLo INT3 + compensator CUDA kernel.

    Forward path (mirrors `_moe_forward_moefused` in MiLo's HF runner):
        1. router has already been applied by FusedMoE upstream → we
           receive (topk_weights, topk_ids).
        2. Sorted dispatch: gather rows by expert id.
        3. One CUDA launch per rail (gate / up / down) via
           `milo_cuda.mul_3bit_moe_with_zeros`.
        4. Batched compensator on each rail.
        5. SwiGLU between gate/up; reduce-by-routing-weight; scatter-add.
    """

    def __init__(self, quant_config: MiloConfig, moe_config):
        super().__init__(moe_config)
        self.quant_config = quant_config
        # Resolve MiLo runtime modules (will raise if missing).
        self._milo_cuda, self._milo_moe, self._compbatch = _import_milo()
        # Defer kernel-tile selection until we know (K, N) per rail.
        self._tile_cache: dict[tuple[int, int], tuple[int, int]] = {}

    # ------------------------------------------------------------------
    #  Weight registration
    # ------------------------------------------------------------------
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        """Register raw INT3-packed buffers for w13 (gate+up) and w2 (down).

        Layout convention (matches MiLo `Layer3bitWithZeros.pack`):
            For each expert and each rail (in_features=K, out_features=N):
                B1     [K/16, N]            int32
                B2     [K/16, N/2]          int32
                scales [K/gs, N]            params_dtype
                zeros  [K/gs, N]            params_dtype
                V      [K, rank]            params_dtype  (compensator lhs)
                U      [rank, N]            params_dtype  (compensator rhs)

        FusedMoE convention stacks gate+up into w13 with N = 2*I (gate first,
        then up).  We follow the same convention.
        """
        # TODO(milo): fill in.  For each of w13_/w2_:
        #   - register Wq_packed1 / Wq_packed2 / scales / zeros / V / U
        #     as nn.Parameter with the correct expert-stacked shape
        #   - attach a custom `weight_loader` (or use vLLM's default) so the
        #     converter-produced safetensors keys map to the right slices
        #   - call `set_weight_attrs(param, extra_weight_attrs)` so vLLM
        #     can do TP / EP sharding correctly
        raise NotImplementedError(
            "MiloMoEMethod.create_weights is not implemented yet.  "
            "See vllm/model_executor/layers/quantization/awq_marlin.py "
            "AWQMarlinMoEMethod.create_weights for the pattern to follow."
        )

    # ------------------------------------------------------------------
    #  Layout finalisation after all weights are loaded
    # ------------------------------------------------------------------
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Convert raw expert-stacked buffers into MiLoMoERail form, build
        compensator bmm caches, and pre-compute kernel-tile selection.

        After this call the layer should expose:
            layer._milo_gate_rail : MiLoMoERail
            layer._milo_up_rail   : MiLoMoERail
            layer._milo_down_rail : MiLoMoERail
            layer._milo_gate_V / _milo_gate_U : torch.Tensor
            layer._milo_up_V   / _milo_up_U   : torch.Tensor
            layer._milo_down_V / _milo_down_U : torch.Tensor
        """
        # TODO(milo): implement.  Algorithm:
        #   1. Split layer.w13_qweight into gate (first I cols) and up (last I cols).
        #      Same for w13_scales / w13_zeros / w13_V / w13_U.
        #   2. For each rail, build a MiLoMoERail dataclass (do NOT call
        #      `stack_from_experts` — that expects nn.Module instances; just
        #      construct the dataclass directly with the already-stacked
        #      tensors).
        #   3. For each rail, stack V/U via
        #      `compensator_batched.stack_compensator_weights`.
        #   4. Free the now-redundant per-expert buffers
        #      (set layer.w13_qweight = None; gc.collect; cuda.empty_cache).
        raise NotImplementedError(
            "MiloMoEMethod.process_weights_after_loading is not "
            "implemented yet."
        )

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """MiLo INT3 + compensator MoE forward.

        Args:
            x: [N, H] fp16/bf16 — flattened token activations after router.
            topk_weights: [N, top_k] fp32 — already softmaxed.
            topk_ids:     [N, top_k] int32 — selected expert ids per token.

        Returns:
            [N, H] in the same dtype as `x`.
        """
        # TODO(milo): implement.  Algorithm:
        #
        #   1. Sorted dispatch (verbatim port from
        #      `_moe_forward_moefused` in MiLo/models/hf/qwen3_moe.py):
        #        flat_experts = topk_ids.reshape(-1)
        #        sort_idx     = torch.argsort(flat_experts)
        #        sorted_tokens = ...
        #        bin_edges    = bincount.cumsum(0)
        #        x_sorted     = x.index_select(0, sorted_tokens)
        #
        #   2. Build MoE block descriptors:
        #        work = self._milo_moe.build_moe_block_descriptors(
        #            bin_edges_cpu, active_experts,
        #            prob_n=I, thread_n=gate_rail.thread_n)
        #        work = work.to(device, non_blocking=True)
        #
        #   3. Layer 1a (gate):
        #        gate_sorted = self._milo_moe.milo_int3_moe(
        #            x_sorted, layer._milo_gate_rail, work)
        #        self._compbatch.compensator_batched(
        #            x_sorted, layer._milo_gate_V, layer._milo_gate_U,
        #            active_experts, bin_edges_cpu, gate_sorted)
        #
        #   4. Layer 1b (up): same as gate.
        #
        #   5. SwiGLU:  h_sorted = F.silu(gate_sorted) * up_sorted
        #
        #   6. Layer 2 (down): same pattern; use a separate `work_down`
        #      descriptor since (prob_n_down, thread_n_down) differ.
        #
        #   7. Apply routing weights and scatter-add back.
        raise NotImplementedError(
            "MiloMoEMethod.apply is not implemented yet."
        )

    def get_fused_moe_quant_config(self, layer):
        """Return the metadata that vLLM uses to pick the right runner."""
        # TODO(milo): adapt to the schema in fused_moe/config.py.  For
        # now return None to force the monolithic apply() path.
        return None
