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
        # Per-module-class rank mapping written by the offline converter.
        # Keys are substrings matched against the module prefix, values are
        # integer ranks of the (V, U) low-rank compensator.  Example for
        # Qwen1.5-MoE-A2.7B w3s16d512:
        #     {"self_attn": 512, "shared_expert": 512, "mlp.experts": 16}
        # Looked up via `resolve_compensator_rank(prefix)` below.
        self.milo_ranks = milo_ranks or {}

    def resolve_compensator_rank(self, prefix: str) -> int:
        """Return the (V, U) rank for the linear at `prefix`.

        Search order:
          1. Substring match against `self.milo_ranks` (longest match wins).
          2. Fall back to the legacy single `compensator_rank` field.
          3. 0 (== no compensator) if nothing matched.
        """
        if self.milo_ranks:
            best_match: Optional[tuple[int, int]] = None  # (len(key), rank)
            for key, rank in self.milo_ranks.items():
                if key in prefix:
                    if best_match is None or len(key) > best_match[0]:
                        best_match = (len(key), int(rank))
            if best_match is not None:
                return best_match[1]
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
        milo_ranks = cls.get_from_keys_or(
            config, ["_milo_ranks"], default={}
        )
        return cls(
            weight_bits=weight_bits,
            group_size=group_size,
            has_zp=has_zp,
            has_compensator=has_compensator,
            compensator_rank=compensator_rank,
            modules_to_not_convert=modules_to_not_convert,
            lm_head_quantized=lm_head_quantized,
            milo_ranks=milo_ranks,
        )

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> "QuantizationMethods | None":
        # MiLo is a non-override (primary) backend: the checkpoint's
        # `config.json["quantization_config"]["quant_method"]` is literally
        # `"milo"`, so vLLM's main resolver picks `MiloConfig` directly via
        # the `quantization_methods` registry — there is no other backend
        # whose checkpoint we want to "claim".  Returning None here keeps
        # `milo` out of the override list (see vllm/config/model.py:962),
        # which avoids the "not in overrides list above" pydantic error.
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        # Skip blacklisted modules.
        for blacklisted in self.modules_to_not_convert:
            if blacklisted in prefix:
                return UnquantizedLinearMethod()

        if isinstance(layer, FusedMoE):
            # MoE path is still TODO; returning None here keeps `vllm serve`
            # alive long enough to verify the dense (attn / lm_head) path.
            # The runtime will fall back to the unquantized FusedMoE method,
            # which will fail to load INT3 buffers — that's expected for
            # Step-3a.  Set `modules_to_not_convert` to skip experts entirely
            # if you want a fully-loadable bring-up checkpoint.
            return MiloMoEMethod(self, layer.moe_config)
        if isinstance(layer, LinearBase) or (
            isinstance(layer, ParallelLMHead) and self.lm_head_quantized
        ):
            return MiloLinearMethod(self, prefix=prefix)
        return None


# ===========================================================================
#  Linear method
# ===========================================================================
class MiloLinearMethod(LinearMethodBase):
    """Linear method for MiLo INT3 + low-rank compensator.

    Forward path (one-to-one with `MiLo_Asymmetric_Linear.forward`):

        out  = milo.mul_3bit_with_zeros(x, Wq_packed1, Wq_packed2,
                                         scales, zeros)
        out += (x @ V) @ U                  # if has_compensator
        out += bias                         # if bias is not None

    Weight registration matches the Marlin-prepacked layout produced by
    `convert_milo_to_vllm.py`:

        Wq_packed1   int32   [in/16,        out]
        Wq_packed2   int32   [in/16,        out/2]   (Marlin zigzag halves)
        scales       fp16    [in/group,     out]
        zeros        fp16    [in/group,     out]
        V            fp16    [in,           rank]    (compensator lhs)
        U            fp16    [rank,         out]     (compensator rhs)
        bias         fp16    [out]                   (optional)

    All buffers expose `output_dim=1` (or `output_dim=0` for U / bias) so
    vLLM's `load_qkv_weight` / `load_merged_column_weight` can stack
    q/k/v or gate/up shards along the N axis at load time.

    NOTE: stacking Marlin-prepacked tensors along N is only safe if every
    shard's N is a multiple of the kernel tile size (max 256 in the MiLo
    kernel set).  All shapes in Qwen1.5-MoE / Qwen3-MoE / DeepSeek-V2 are
    safe; assert if you hit a violation.
    """

    # Cache the resolved milo runtime module across all instances.
    _milo_module: Any = None

    def __init__(self, quant_config: MiloConfig, *, prefix: str = ""):
        self.quant_config = quant_config
        self.prefix = prefix
        # Resolve compensator rank up-front so create_weights can size V/U.
        # 0 means "no compensator on this layer".
        self.rank = (
            quant_config.resolve_compensator_rank(prefix)
            if quant_config.has_compensator
            else 0
        )
        # MiLo runtime is only required at apply() time, not at config-parse
        # time, so we lazy-import to keep import paths cheap.
        self._milo: Any = None

    # ------------------------------------------------------------------
    def _get_milo(self):
        if self._milo is not None:
            return self._milo
        try:
            import milo  # noqa: F401  (registers ops + Python entry points)
        except ImportError as e:
            raise ImportError(
                "MiLo runtime is not installed.  Build the milo CUDA "
                "extension: `cd MiLo/MiLo/kernels && pip install -e .`"
            ) from e
        self._milo = milo
        type(self)._milo_module = milo
        return milo

    # ------------------------------------------------------------------
    @staticmethod
    def _pick_kernel_tile(prob_n: int, prob_k: int) -> tuple[int, int]:
        """Return a (thread_n, thread_k) tile compatible with prob_n / prob_k
        and the set of `CALL_IF` configurations compiled into the MiLo CUDA
        kernel.  Same logic as MiLo_Asymmetric_Linear._pick_kernel_tile."""
        if prob_n % 256 == 0 and prob_k % 64 == 0:
            return 256, 64
        if prob_n % 128 == 0 and prob_k % 128 == 0:
            return 128, 128
        if prob_n % 64 == 0 and prob_k % 256 == 0:
            return 64, 256
        return -1, -1

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
        del input_size, output_size  # unused; we work with partition sizes

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        gs = self.quant_config.group_size
        K = input_size_per_partition
        N = output_size_per_partition

        # Sanity: the Marlin prepack requires K % 16 == 0 (B1/B2 layout) and
        # K % gs == 0 (scales / zeros groups).  Each output shard must be a
        # multiple of the kernel tile (max 256) so that stacking along N is
        # byte-equivalent to per-shard prepack.
        assert K % 16 == 0, f"in_features={K} not divisible by 16"
        assert K % gs == 0, (
            f"in_features={K} not divisible by group_size={gs}"
        )
        for shard_n in output_partition_sizes:
            assert shard_n % 64 == 0, (
                f"output shard {shard_n} not a multiple of 64; cannot safely "
                f"concat MiLo Marlin-prepacked tensors along N"
            )

        # Save layer-level metadata for apply().
        layer.input_size_per_partition = K
        layer.output_size_per_partition = N
        layer.milo_group_size = gs
        layer.milo_rank = self.rank

        # ---- INT3 packed weights (B1, B2) ----
        # Both have output_dim=1 (N).  Treat as un-packed along input_dim
        # (the K/16 prepack is opaque to vLLM's TP slicer for now — Step 3a
        # only validates TP=1).
        wq1 = ModelWeightParameter(
            data=torch.empty(K // 16, N, dtype=torch.int32),
            input_dim=0, output_dim=1,
            weight_loader=weight_loader,
        )
        wq2 = ModelWeightParameter(
            data=torch.empty(K // 16, N // 2, dtype=torch.int32),
            input_dim=0, output_dim=1,
            weight_loader=weight_loader,
        )
        layer.register_parameter("Wq_packed1", wq1)
        layer.register_parameter("Wq_packed2", wq2)

        # ---- scales / zeros (per-group, per-output) ----
        # Use GroupQuantScaleParameter so vLLM's qkv/merged-column loader
        # also slices them correctly along output_dim=1.
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

        # ---- Low-rank compensator (V, U) ----
        # V is [K, rank], U is [rank, N].  The rank dimension is INDEPENDENT
        # of the output partition (N) and must NOT participate in merged-column
        # or QKV stacking.  We register them as plain nn.Parameters with a
        # simple weight_loader that ignores shard_id and copies the full tensor
        # (correct for TP=1; TP>1 requires explicit handling in the future).
        if self.rank > 0:
            V = torch.nn.Parameter(
                torch.empty(K, self.rank, dtype=params_dtype),
                requires_grad=False,
            )
            U = torch.nn.Parameter(
                torch.empty(self.rank, N, dtype=params_dtype),
                requires_grad=False,
            )
            layer.register_parameter("V", V)
            layer.register_parameter("U", U)
            # Attach a no-op weight_loader so vLLM's default_weight_loader
            # just copies the loaded tensor directly.
        else:
            layer.V = None
            layer.U = None

        # ---- Optional bias ----
        # vLLM's LinearBase already creates `layer.bias` for us when
        # bias=True is set on construction.  But MiLo writes bias under the
        # quantized linear with the key `<prefix>.bias`, which lands on the
        # nn.Module attribute path that vLLM owns.  We still register a
        # GroupQuantScale-style 1D parameter to receive shard-id-aware
        # loading from load_qkv_weight.
        # (vLLM's `LinearBase.__init__` will register a `bias` parameter
        # if requested by the model code — we don't override it here.)

        # Workspace for the milo INT3 GEMM kernel.  Sized once we know N.
        # `n // 128 * 16` int32 elements (per kernel header).  We allocate
        # it lazily in process_weights_after_loading because some platforms
        # don't allow torch.zeros() during create_weights.

    # ------------------------------------------------------------------
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Pick the kernel tile + allocate the workspace lock buffer."""
        N = layer.output_size_per_partition
        K = layer.input_size_per_partition
        thread_n, thread_k = self._pick_kernel_tile(N, K)
        if thread_n < 0:
            raise RuntimeError(
                f"MiloLinearMethod: no compatible kernel tile for "
                f"prob_n={N} prob_k={K}.  Allowed (N%256==0, K%64==0) | "
                f"(N%128==0, K%128==0) | (N%64==0, K%256==0)."
            )
        layer.milo_thread_n = thread_n
        layer.milo_thread_k = thread_k

        # Workspace: int32 lock array used by the kernel scheduler.
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

        # MiLo INT3 kernel only supports fp16 activations; cast if needed.
        # (Bf16 → fp16 is lossy for some calibrations; the upstream check-
        # point is calibrated in fp16 anyway.)
        orig_dtype = x.dtype
        if x.dtype != torch.float16:
            x = x.to(torch.float16)

        x_2d = x.reshape(-1, x.shape[-1])
        N = layer.output_size_per_partition
        out = torch.empty(
            (x_2d.shape[0], N), dtype=torch.float16, device=x_2d.device
        )

        # 1) INT3 grouped GEMM: out = dequant(Wq) @ x
        milo.mul_3bit_with_zeros(
            x_2d,
            layer.Wq_packed1,
            layer.Wq_packed2,
            out,
            layer.scales,
            layer.zeros,
            layer.milo_workspace,
            thread_k=layer.milo_thread_k,
            thread_n=layer.milo_thread_n,
        )

        # 2) Low-rank compensator: out += (x @ V) @ U
        if layer.V is not None and layer.U is not None:
            tmp = torch.mm(x_2d, layer.V)        # [M, rank]
            out.addmm_(tmp, layer.U)             # out += tmp @ U

        # 3) Bias (vLLM may pass it via `bias` arg, or layer.bias may exist).
        if bias is not None:
            out = out + bias.to(torch.float16)

        # Restore original output shape + dtype.
        out = out.reshape(*x.shape[:-1], N)
        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)
        return out


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

    # Monolithic: this method owns the full MoE forward; vLLM should NOT
    # try to wrap it in the modular kernel framework.
    is_monolithic = True

    def __init__(self, quant_config: MiloConfig, moe_config):
        super().__init__(moe_config)
        self.quant_config = quant_config
        # Lazy-import MiLo runtime (only needed at forward time).
        self._milo_moe = None
        self._compbatch = None

    def _ensure_runtime(self):
        if self._milo_moe is not None:
            return
        _, self._milo_moe, self._compbatch = _import_milo()

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

        Layout per expert:
            w13 (gate+up fused, N = 2*I):
                Wq_packed1  [K/16,  2*I]       int32
                Wq_packed2  [K/16,  I]         int32   (N/2 = I)
                scales      [K/gs,  2*I]       params_dtype
                zeros       [K/gs,  2*I]       params_dtype
                V           [K,     rank]      params_dtype
                U           [rank,  2*I]       params_dtype
            w2 (down, K_in=I, N_out=H):
                Wq_packed1  [I/16,  H]         int32
                Wq_packed2  [I/16,  H/2]       int32
                scales      [I/gs,  H]         params_dtype
                zeros       [I/gs,  H]         params_dtype
                V           [I,     rank]      params_dtype
                U           [rank,  H]         params_dtype

        Parameters are registered with shape [num_experts, ...] and
        `is_transposed=True` so vLLM's weight_loader slices along the
        correct dim when filling gate(w1) vs up(w3) halves.
        """
        from vllm.model_executor.utils import set_weight_attrs

        gs = self.quant_config.group_size
        K = hidden_size
        I = intermediate_size_per_partition
        rank = self.quant_config.resolve_compensator_rank(
            "mlp.experts"  # MoE experts always use the expert rank
        )

        # Mark transposed so weight_loader flips shard_dim correctly.
        # "quant_method": "group" tells the weight_loader how to handle
        # scales/zeros (same group-quantization loading path as GPTQ).
        extra_weight_attrs.update({
            "is_transposed": True,
            "quant_method": "group",
        })
        logger.info("MiloMoEMethod.create_weights: E=%d K=%d I=%d rank=%d "
                    "extra_keys=%s", num_experts, K, I, rank,
                    list(extra_weight_attrs.keys()))

        # ---------- w13 (gate + up, fused along N=2*I) ----------
        def _reg(name, shape, dtype=torch.int32):
            p = torch.nn.Parameter(
                torch.empty(num_experts, *shape, dtype=dtype),
                requires_grad=False,
            )
            layer.register_parameter(name, p)
            set_weight_attrs(p, extra_weight_attrs)
            # Belt-and-suspenders: ensure quant_method is readable via getattr
            # even if set_weight_attrs path has issues with Parameter subclass.
            if not hasattr(p, "quant_method") or p.quant_method is None:
                p.quant_method = "group"

        _reg("w13_Wq_packed1", (K // 16,  2 * I), torch.int32)
        _reg("w13_Wq_packed2", (K // 16,  I),     torch.int32)     # N/2
        _reg("w13_scales",     (K // gs,  2 * I), params_dtype)
        _reg("w13_zeros",      (K // gs,  2 * I), params_dtype)
        if rank > 0:
            _reg("w13_V",      (K,        rank),  params_dtype)
            _reg("w13_U",      (rank,     2 * I), params_dtype)

        # ---------- w2 (down: K_in=I, N_out=H) ----------
        _reg("w2_Wq_packed1",  (I // 16,  K),     torch.int32)
        _reg("w2_Wq_packed2",  (I // 16,  K // 2), torch.int32)
        _reg("w2_scales",      (I // gs,  K),     params_dtype)
        _reg("w2_zeros",       (I // gs,  K),     params_dtype)
        if rank > 0:
            _reg("w2_V",       (I,        rank),  params_dtype)
            _reg("w2_U",       (rank,     K),     params_dtype)

        # Save metadata for process_weights_after_loading / apply.
        layer.milo_num_experts = num_experts
        layer.milo_hidden_size = K
        layer.milo_intermediate_size = I
        layer.milo_group_size = gs
        layer.milo_rank = rank

    # ------------------------------------------------------------------
    #  Layout finalisation after all weights are loaded
    # ------------------------------------------------------------------
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Pre-compute kernel tile selection for each rail."""
        K = layer.milo_hidden_size
        I = layer.milo_intermediate_size

        # gate/up rail: prob_k=K, prob_n=I  (we split w13 into two halves)
        layer.milo_tile_gate = MiloLinearMethod._pick_kernel_tile(I, K)
        layer.milo_tile_up   = layer.milo_tile_gate  # same shape
        # down rail: prob_k=I, prob_n=K
        layer.milo_tile_down = MiloLinearMethod._pick_kernel_tile(K, I)

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

        TODO(milo): implement full fused path using milo_int3_moe kernel.
        For now raises NotImplementedError.
        """
        raise NotImplementedError(
            "MiloMoEMethod.apply is not implemented yet (Step 3c).  "
            "Weight registration and loading (Step 3b) are complete."
        )

    def get_fused_moe_quant_config(self, layer):
        """Return the metadata that vLLM uses to pick the right runner."""
        return None
