# SPDX-License-Identifier: MIT
"""Import seam for SGLang kernel helpers that moved in v0.5.17.

v0.5.17 split ``sglang.srt.layers.attention.utils`` and
``sglang.srt.layers.quantization.fp8_kernel`` apart, scattering the helpers
into ``sglang.kernels.ops.*``. The symbols themselves are unchanged, so each
one is resolved from its v0.5.17 home first and its v0.5.15 home second and the
plugin loads unmodified on either.

Keep this file a flat lookup table, not an abstraction: it exists so the port
is greppable and can be deleted once the oldest supported SGLang is v0.5.17.
"""

from __future__ import annotations

import importlib
from typing import Any


def _resolve(symbol: str, *modules: str) -> Any:
    """First ``modules`` entry that exposes ``symbol``."""
    tried = []
    for module in modules:
        try:
            return getattr(importlib.import_module(module), symbol)
        except (ImportError, AttributeError) as exc:
            tried.append(f"{module} ({type(exc).__name__})")
    raise ImportError(
        f"SGLang does not expose {symbol!r} in any known location; tried "
        + ", ".join(tried)
    )


_ATTENTION_UTILS_V0515 = "sglang.srt.layers.attention.utils"
_FP8_KERNEL_V0515 = "sglang.srt.layers.quantization.fp8_kernel"

create_flashinfer_kv_indices_triton = _resolve(
    "create_flashinfer_kv_indices_triton",
    "sglang.kernels.ops.kvcache.kv_indices",
    _ATTENTION_UTILS_V0515,
)
launch_reshape_and_cache_flash = _resolve(
    "launch_reshape_and_cache_flash",
    "sglang.kernels.ops.kvcache.cache_ops",
    _ATTENTION_UTILS_V0515,
)
pad_sequence_with_mask = _resolve(
    "pad_sequence_with_mask",
    "sglang.kernels.ops.attention.pad",
    _ATTENTION_UTILS_V0515,
)
concat_and_cast_mha_k_triton = _resolve(
    "concat_and_cast_mha_k_triton",
    "sglang.kernels.ops.kvcache.cache_ops",
    _ATTENTION_UTILS_V0515,
)
per_tensor_quant_mla_fp8 = _resolve(
    "per_tensor_quant_mla_fp8",
    "sglang.kernels.ops.quantization.fp8_kernel",
    _FP8_KERNEL_V0515,
)
per_token_group_quant_mla_deep_gemm_masked_fp8 = _resolve(
    "per_token_group_quant_mla_deep_gemm_masked_fp8",
    "sglang.kernels.ops.quantization.fp8_kernel",
    _FP8_KERNEL_V0515,
)

__all__ = [
    "concat_and_cast_mha_k_triton",
    "create_flashinfer_kv_indices_triton",
    "launch_reshape_and_cache_flash",
    "pad_sequence_with_mask",
    "per_tensor_quant_mla_fp8",
    "per_token_group_quant_mla_deep_gemm_masked_fp8",
]
