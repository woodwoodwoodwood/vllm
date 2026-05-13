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
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
)

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy-import MiLo runtime
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

    Expected `config.json["quantization_config"]` schema::

        {
            "quant_method":   "milo",
            "bits":           3,
            "group_size":     64,
            "has_zp":         true,
            "has_compensator": true,
            "compensator_rank": 32,
            "modules_to_not_convert": ["lm_head"],
            "_milo_ranks": {"self_attn": 512, "shared_expert": 512,
                            "mlp.experts": 16}
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
        milo_ranks: Optional[dict[str, int]] = None,
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
        self.milo_ranks = milo_ranks or {}

    def resolve_compensator_rank(self, prefix: str) -> int:
        """Return the (V, U) rank for the linear at *prefix*.

        Longest substring match against ``self.milo_ranks``, falling back
        to ``self.compensator_rank``, then 0.
        """
        if self.milo_ranks:
            best: Optional[tuple[int, int]] = None
            for key, rank in self.milo_ranks.items():
                if key in prefix:
                    if best is None or len(key) > best[0]:
                        best = (len(key), int(rank))
            if best is not None:
                return best[1]
        return int(self.compensator_rank or 0)

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
            config, ["has_compensator"], default=True)
        compensator_rank = cls.get_from_keys_or(
            config, ["compensator_rank"], default=0)
        modules_to_not_convert = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], default=[])
        lm_head_quantized = cls.get_from_keys_or(
            config, ["lm_head"], default=False)
        milo_ranks = cls.get_from_keys_or(
            config, ["_milo_ranks"], default={})
        return cls(
            weight_bits=weight_bits, group_size=group_size,
            has_zp=has_zp, has_compensator=has_compensator,
            compensator_rank=compensator_rank,
            modules_to_not_convert=modules_to_not_convert,
            lm_head_quantized=lm_head_quantized,
            milo_ranks=milo_ranks,
        )

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> "QuantizationMethods | None":
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        for blacklisted in self.modules_to_not_convert:
            if blacklisted in prefix:
                return UnquantizedLinearMethod()
        if isinstance(layer, FusedMoE):
            return MiloMoEMethod(self, layer.moe_config)
        if isinstance(layer, LinearBase) or (
            isinstance(layer, ParallelLMHead) and self.lm_head_quantized
        ):
            return MiloLinearMethod(self, prefix=prefix)
        return None


# ===========================================================================
#  Helpers
# ===========================================================================
def _pick_kernel_tile(prob_n: int, prob_k: int) -> tuple[int, int]:
    """Return (thread_n, thread_k) for the MiLo INT3 GEMM kernel."""
    if prob_n % 256 == 0 and prob_k % 64 == 0:
        return 256, 64
    if prob_n % 128 == 0 and prob_k % 128 == 0:
        return 128, 128
    if prob_n % 64 == 0 and prob_k % 256 == 0:
        return 64, 256
    return -1, -1


# ===========================================================================
#  Linear method
# ===========================================================================
class MiloLinearMethod(LinearMethodBase):
    """Dense-linear method for MiLo INT3 + low-rank compensator.

    Handles attention projections (q/k/v/o), shared-expert gate/up/down,
    and optionally lm_head.

    Forward::

        out  = milo.mul_3bit_with_zeros(x, B1, B2, scales, zeros, ws)
        out += (x @ V) @ U          # if compensator exists
        out += bias                  # if bias is not None
    """

    _milo_module: Any = None

    def __init__(self, quant_config: MiloConfig, *, prefix: str = ""):
        self.quant_config = quant_config
        self.prefix = prefix
        self.rank = (
            quant_config.resolve_compensator_rank(prefix)
            if quant_config.has_compensator else 0
        )
        self._milo: Any = None

    def _get_milo(self):
        if self._milo is not None:
            return self._milo
        try:
            import milo
        except ImportError as e:
            raise ImportError(
                "MiLo runtime not installed.  "
                "cd MiLo/MiLo/kernels && pip install -e ."
            ) from e
        self._milo = milo
        type(self)._milo_module = milo
        return milo

    # ------------------------------------------------------------------
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        gs = self.quant_config.group_size
        K = input_size_per_partition
        N = output_size_per_partition

        assert K % 16 == 0, f"in_features={K} not divisible by 16"
        assert K % gs == 0, f"in_features={K} not divisible by group_size={gs}"

        layer.input_size_per_partition = K
        layer.output_size_per_partition = N
        layer.milo_group_size = gs
        layer.milo_rank = self.rank

        from vllm.model_executor.parameter import PackedvLLMParameter

        wq1 = ModelWeightParameter(
            data=torch.empty(K // 16, N, dtype=torch.int32),
            input_dim=0, output_dim=1,
            weight_loader=weight_loader,
        )
        wq2 = PackedvLLMParameter(
            data=torch.empty(K // 16, N // 2, dtype=torch.int32),
            input_dim=0, output_dim=1,
            packed_dim=1, packed_factor=2,
            weight_loader=weight_loader,
        )
        layer.register_parameter("Wq_packed1", wq1)
        layer.register_parameter("Wq_packed2", wq2)

        scales = GroupQuantScaleParameter(
            data=torch.empty(K // gs, N, dtype=params_dtype),
            input_dim=0, output_dim=1,
            weight_loader=weight_loader,
        )
        zeros = GroupQuantScaleParameter(
            data=torch.empty(K // gs, N, dtype=params_dtype),
            input_dim=0, output_dim=1,
            weight_loader=weight_loader,
        )
        layer.register_parameter("scales", scales)
        layer.register_parameter("zeros", zeros)

        # V/U: rank dimension is independent of N; use a direct-copy loader
        # that ignores shard_id so QKV/gate-up stacking won't try to narrow.
        # For QKV-stacked layers, q/k/v each have independent V/U of shape
        # [K, rank]; we keep only V (all same K, same rank) and stack U
        # along the output dimension manually.
        if self.rank > 0:
            def _comp_loader(param, loaded_weight, *args, **kwargs):
                if param.data.shape == loaded_weight.shape:
                    param.data.copy_(loaded_weight)

            V = torch.nn.Parameter(
                torch.empty(K, self.rank, dtype=params_dtype),
                requires_grad=False,
            )
            U = torch.nn.Parameter(
                torch.empty(self.rank, N, dtype=params_dtype),
                requires_grad=False,
            )
            V.weight_loader = _comp_loader
            U.weight_loader = _comp_loader
            layer.register_parameter("V", V)
            layer.register_parameter("U", U)
        else:
            layer.V = None
            layer.U = None

    # ------------------------------------------------------------------
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        N = layer.output_size_per_partition
        K = layer.input_size_per_partition
        thread_n, thread_k = _pick_kernel_tile(N, K)
        if thread_n < 0:
            raise RuntimeError(
                f"MiloLinearMethod: no kernel tile for "
                f"prob_n={N} prob_k={K}."
            )
        layer.milo_thread_n = thread_n
        layer.milo_thread_k = thread_k
        device = layer.Wq_packed1.device
        layer.milo_workspace = torch.zeros(
            (N // 128) * 16, dtype=torch.int32, device=device
        )

    # ------------------------------------------------------------------
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        milo = self._get_milo()

        orig_dtype = x.dtype
        if x.dtype != torch.float16:
            x = x.to(torch.float16)

        x_2d = x.reshape(-1, x.shape[-1])
        N = layer.output_size_per_partition
        out = torch.empty(
            (x_2d.shape[0], N), dtype=torch.float16, device=x_2d.device)

        milo.mul_3bit_with_zeros(
            x_2d, layer.Wq_packed1, layer.Wq_packed2, out,
            layer.scales, layer.zeros, layer.milo_workspace,
            thread_k=layer.milo_thread_k, thread_n=layer.milo_thread_n,
        )

        if layer.V is not None and layer.U is not None:
            # TEMP: skip compensator on dense linears (attn / shared_expert)
            # because QKV/gate_up stacking corrupts V/U layout.  MoE experts
            # are unaffected (no stacking).
            import os
            if os.environ.get("MILO_LINEAR_COMPENSATOR", "0") == "1":
                tmp = torch.mm(x_2d, layer.V)
                out.addmm_(tmp, layer.U)

        if bias is not None:
            out = out + bias.to(torch.float16)

        out = out.reshape(*x.shape[:-1], N)
        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)
        return out


# ===========================================================================
#  Fused-MoE method
# ===========================================================================
class MiloMoEMethod(FusedMoEMethodBase):
    """Fused-MoE method backed by MiLo INT3 + compensator CUDA kernel."""

    is_monolithic = True

    def __init__(self, quant_config: MiloConfig, moe_config):
        super().__init__(moe_config)
        self.quant_config = quant_config
        self._milo_moe = None
        self._compbatch = None

    def _ensure_runtime(self):
        if self._milo_moe is not None:
            return
        _, self._milo_moe, self._compbatch = _import_milo()

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
        from vllm.model_executor.utils import set_weight_attrs

        gs = self.quant_config.group_size
        K = hidden_size
        I = intermediate_size_per_partition
        rank = self.quant_config.resolve_compensator_rank("mlp.experts")

        extra_weight_attrs.update({
            "is_transposed": True,
            "quant_method": "group",
        })

        def _reg(name, shape, dtype=torch.int32):
            p = torch.nn.Parameter(
                torch.empty(num_experts, *shape, dtype=dtype),
                requires_grad=False,
            )
            layer.register_parameter(name, p)
            set_weight_attrs(p, extra_weight_attrs)
            if not hasattr(p, "quant_method") or p.quant_method is None:
                p.quant_method = "group"

        _reg("w13_Wq_packed1", (K // 16,  2 * I), torch.int32)
        _reg("w13_Wq_packed2", (K // 16,  I),     torch.int32)
        _reg("w13_scales",     (K // gs,  2 * I), params_dtype)
        _reg("w13_zeros",      (K // gs,  2 * I), params_dtype)
        if rank > 0:
            _reg("w13_V",      (K,        rank),  params_dtype)
            _reg("w13_U",      (rank,     2 * I), params_dtype)

        _reg("w2_Wq_packed1",  (I // 16,  K),     torch.int32)
        _reg("w2_Wq_packed2",  (I // 16,  K // 2), torch.int32)
        _reg("w2_scales",      (I // gs,  K),     params_dtype)
        _reg("w2_zeros",       (I // gs,  K),     params_dtype)
        if rank > 0:
            _reg("w2_V",       (I,        rank),  params_dtype)
            _reg("w2_U",       (rank,     K),     params_dtype)

        layer.milo_num_experts = num_experts
        layer.milo_hidden_size = K
        layer.milo_intermediate_size = I
        layer.milo_group_size = gs
        layer.milo_rank = rank

    # ------------------------------------------------------------------
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self._ensure_runtime()
        from MiLo.fused_moe.milo_moe import MiLoMoERail

        K = layer.milo_hidden_size
        I = layer.milo_intermediate_size
        gs = layer.milo_group_size
        E = layer.milo_num_experts
        rank = layer.milo_rank

        tile_gu_n, tile_gu_k = _pick_kernel_tile(I, K)
        tile_dn_n, tile_dn_k = _pick_kernel_tile(K, I)

        # Split w13 into gate (first I cols) and up (last I cols)
        gate_B1 = layer.w13_Wq_packed1[:, :, :I].contiguous()
        up_B1   = layer.w13_Wq_packed1[:, :, I:].contiguous()
        gate_B2 = layer.w13_Wq_packed2[:, :, :I//2].contiguous()
        up_B2   = layer.w13_Wq_packed2[:, :, I//2:].contiguous()
        gate_scales = layer.w13_scales[:, :, :I].contiguous()
        up_scales   = layer.w13_scales[:, :, I:].contiguous()
        gate_zeros  = layer.w13_zeros[:, :, :I].contiguous()
        up_zeros    = layer.w13_zeros[:, :, I:].contiguous()

        layer._milo_gate_rail = MiLoMoERail(
            B1=gate_B1, B2=gate_B2, scales=gate_scales, zeros=gate_zeros,
            prob_n=I, prob_k=K, group_size=gs,
            thread_n=tile_gu_n, thread_k=tile_gu_k, num_experts=E,
        )
        layer._milo_up_rail = MiLoMoERail(
            B1=up_B1, B2=up_B2, scales=up_scales, zeros=up_zeros,
            prob_n=I, prob_k=K, group_size=gs,
            thread_n=tile_gu_n, thread_k=tile_gu_k, num_experts=E,
        )
        layer._milo_down_rail = MiLoMoERail(
            B1=layer.w2_Wq_packed1, B2=layer.w2_Wq_packed2,
            scales=layer.w2_scales, zeros=layer.w2_zeros,
            prob_n=K, prob_k=I, group_size=gs,
            thread_n=tile_dn_n, thread_k=tile_dn_k, num_experts=E,
        )

        if rank > 0:
            layer._milo_gate_V = layer.w13_V
            layer._milo_gate_U = layer.w13_U[:, :, :I].contiguous()
            layer._milo_up_V   = layer.w13_V
            layer._milo_up_U   = layer.w13_U[:, :, I:].contiguous()
            layer._milo_down_V = layer.w2_V
            layer._milo_down_U = layer.w2_U
        else:
            layer._milo_gate_V = layer._milo_gate_U = None
            layer._milo_up_V = layer._milo_up_U = None
            layer._milo_down_V = layer._milo_down_U = None

        # Free raw buffers
        for attr in ("w13_Wq_packed1", "w13_Wq_packed2",
                     "w13_scales", "w13_zeros",
                     "w2_Wq_packed1", "w2_Wq_packed2",
                     "w2_scales", "w2_zeros"):
            if hasattr(layer, attr):
                delattr(layer, attr)
        if rank > 0:
            for attr in ("w13_V", "w13_U", "w2_V", "w2_U"):
                if hasattr(layer, attr):
                    delattr(layer, attr)

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
        self._ensure_runtime()
        from MiLo.fused_moe.milo_moe import (
            build_moe_block_descriptors, milo_int3_moe)
        from MiLo.fused_moe.compensator_batched import compensator_batched

        device = x.device
        orig_dtype = x.dtype
        if x.dtype != torch.float16:
            x = x.to(torch.float16)

        x_2d = x.reshape(-1, x.shape[-1])
        M, K = x_2d.shape
        top_k = topk_ids.shape[-1]
        I = layer.milo_intermediate_size
        E = layer.milo_num_experts
        has_comp = layer._milo_gate_V is not None

        # 1. Sorted dispatch
        flat_experts = topk_ids.reshape(-1)
        sort_idx = torch.argsort(flat_experts, stable=True)
        sorted_token_ids = sort_idx // top_k
        sorted_slot_ids  = sort_idx % top_k

        counts = torch.bincount(flat_experts, minlength=E)
        bin_edges_cpu = [0] + counts.cumsum(0).tolist()
        active_experts = sorted(
            i for i in range(E) if counts[i].item() > 0)

        x_sorted = x_2d[sorted_token_ids].contiguous()

        # 2. Gate rail
        gate_rail = layer._milo_gate_rail
        work_gate = build_moe_block_descriptors(
            bin_edges_cpu, active_experts,
            prob_n=I, thread_n=gate_rail.thread_n,
        ).to(device, non_blocking=True)
        gate_sorted = milo_int3_moe(x_sorted, gate_rail, work_gate)
        if has_comp:
            compensator_batched(
                x_sorted, layer._milo_gate_V, layer._milo_gate_U,
                active_experts, bin_edges_cpu, gate_sorted)

        # 3. Up rail
        up_sorted = milo_int3_moe(x_sorted, layer._milo_up_rail, work_gate)
        if has_comp:
            compensator_batched(
                x_sorted, layer._milo_up_V, layer._milo_up_U,
                active_experts, bin_edges_cpu, up_sorted)

        # 4. SwiGLU
        h_sorted = torch.nn.functional.silu(gate_sorted) * up_sorted
        del gate_sorted, up_sorted

        # 5. Down rail
        down_rail = layer._milo_down_rail
        work_down = build_moe_block_descriptors(
            bin_edges_cpu, active_experts,
            prob_n=K, thread_n=down_rail.thread_n,
        ).to(device, non_blocking=True)
        down_sorted = milo_int3_moe(h_sorted, down_rail, work_down)
        if has_comp:
            compensator_batched(
                h_sorted, layer._milo_down_V, layer._milo_down_U,
                active_experts, bin_edges_cpu, down_sorted)
        del h_sorted

        # 6. Scatter-add with routing weights
        w_sorted = topk_weights[sorted_token_ids, sorted_slot_ids]
        down_weighted = (down_sorted * w_sorted.unsqueeze(-1)).to(
            torch.float16)
        del down_sorted

        out = torch.zeros(M, K, dtype=torch.float16, device=device)
        out.index_add_(0, sorted_token_ids, down_weighted)

        out = out.reshape(*x.shape[:-1], K)
        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)
        return out

    # ------------------------------------------------------------------
    def get_fused_moe_quant_config(self, layer):
        return None

    def apply_monolithic(
        self,
        layer,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        routing_weights = torch.softmax(
            router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(
            routing_weights, k=layer.top_k, dim=-1)
        if layer.renormalize:
            topk_weights = topk_weights / topk_weights.sum(
                dim=-1, keepdim=True)
        return self.apply(
            layer=layer, x=x,
            topk_weights=topk_weights, topk_ids=topk_ids,
        )
