# SPDX-License-Identifier: MIT
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_decode_update_kernel(
    A_log,
    a,
    dt_bias,
    q,
    k,
    v,
    b,
    out,
    state,
    state_indices,
    scale: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HEADS_PER_V: tl.constexpr,
):
    i_k = tl.program_id(0)
    i_v = tl.program_id(1)
    i_nh = tl.program_id(2)
    i_n = i_nh // HV
    i_hv = i_nh - i_n * HV
    i_h = i_hv // HEADS_PER_V

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    state_idx = tl.load(state_indices + i_n).to(tl.int64)
    if state_idx < 0:
        return

    state_base = ((state_idx * HV + i_hv) * K) * V
    state_offsets = state_base + o_k[:, None] * V + o_v[None, :]
    h = tl.load(
        state + state_offsets,
        mask=mask_h,
        other=0.0,
        cache_modifier=".cg",
    ).to(tl.float32)

    q_offsets = (i_n * H + i_h) * K + o_k
    k_offsets = (i_n * H + i_h) * K + o_k
    v_offsets = (i_n * HV + i_hv) * V + o_v
    q_vec = tl.load(
        q + q_offsets,
        mask=mask_k,
        other=0.0,
        cache_modifier=".ca",
    ).to(tl.float32)
    k_vec = tl.load(
        k + k_offsets,
        mask=mask_k,
        other=0.0,
        cache_modifier=".ca",
    ).to(tl.float32)
    v_vec = tl.load(
        v + v_offsets,
        mask=mask_v,
        other=0.0,
        cache_modifier=".ca",
    ).to(tl.float32)

    x = tl.load(a + i_n * HV + i_hv).to(tl.float32) + tl.load(
        dt_bias + i_hv
    ).to(tl.float32)
    softplus_x = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
    gate = -tl.exp(tl.load(A_log + i_hv).to(tl.float32)) * softplus_x
    beta = tl.sigmoid(tl.load(b + i_n * HV + i_hv).to(tl.float32))

    q_vec = q_vec * tl.rsqrt(tl.sum(q_vec * q_vec, axis=0) + 1.0e-6)
    k_vec = k_vec * tl.rsqrt(tl.sum(k_vec * k_vec, axis=0) + 1.0e-6)
    q_vec = q_vec * scale

    h = h * tl.exp(gate)
    v_vec = (v_vec - tl.sum(h * k_vec[:, None], axis=0)) * beta
    h = h + k_vec[:, None] * v_vec[None, :]
    out_vec = tl.sum(h * q_vec[:, None], axis=0)

    out_offsets = (i_n * HV + i_hv) * V + o_v
    tl.store(out + out_offsets, out_vec.to(out.dtype.element_ty), mask=mask_v)
    tl.store(
        state + state_offsets,
        h.to(state.dtype.element_ty),
        mask=mask_h,
        cache_modifier=".cg",
    )


@triton.jit
def _gdn_decode_update_k128_v64_kernel(
    A_log,
    a,
    dt_bias,
    q,
    k,
    v,
    b,
    out,
    state,
    state_indices,
    scale: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HEADS_PER_V: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n = i_nh // HV
    i_hv = i_nh - i_n * HV
    i_h = i_hv // HEADS_PER_V

    state_idx = tl.load(state_indices + i_n).to(tl.int64)
    if state_idx < 0:
        return

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    qk_base = (i_n * H + i_h) * K
    nh_off = i_n * HV + i_hv
    state_offsets = ((state_idx * HV + i_hv) * K + o_k[:, None]) * V + o_v[None, :]
    v_offsets = nh_off * V + o_v
    out_offsets = nh_off * V + o_v

    # Issue the large state load before scalar math so LLVM can overlap memory
    # latency with the independent q/k/v/scalar loads.
    h_raw = tl.load(
        state + state_offsets,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )
    q_raw = tl.load(q + qk_base + o_k, cache_modifier=".ca")
    k_raw = tl.load(k + qk_base + o_k, cache_modifier=".ca")
    v_raw = tl.load(v + v_offsets, cache_modifier=".ca")
    a_val = tl.load(a + nh_off)
    dt_val = tl.load(dt_bias + i_hv)
    A_log_val = tl.load(A_log + i_hv)
    b_val = tl.load(b + nh_off)

    h = h_raw.to(tl.float32)
    q_vec = q_raw.to(tl.float32)
    k_vec = k_raw.to(tl.float32)
    v_vec = v_raw.to(tl.float32)

    x = a_val.to(tl.float32) + dt_val.to(tl.float32)
    softplus_x = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
    gate = -tl.exp(A_log_val.to(tl.float32)) * softplus_x
    beta = tl.sigmoid(b_val.to(tl.float32))
    exp_gate = tl.exp(gate)

    q_norm = tl.rsqrt(tl.sum(q_vec * q_vec, axis=0) + 1.0e-6) * scale
    k_norm = tl.rsqrt(tl.sum(k_vec * k_vec, axis=0) + 1.0e-6)
    q_vec = q_vec * q_norm
    k_vec = k_vec * k_norm

    h = h * exp_gate
    delta = tl.sum(h * k_vec[:, None], axis=0)
    v_vec = (v_vec - delta) * beta
    h = h + k_vec[:, None] * v_vec[None, :]
    out_vec = tl.sum(h * q_vec[:, None], axis=0)

    tl.store(out + out_offsets, out_vec.to(out.dtype.element_ty), cache_modifier=".cg")
    tl.store(state + state_offsets, h.to(state.dtype.element_ty), cache_modifier=".cg")


@triton.jit
def _gdn_decode_update_k128_v128_seq2_kernel(
    A_log,
    a,
    dt_bias,
    q,
    k,
    v,
    b,
    out,
    state,
    state_indices,
    scale: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HEADS_PER_V: tl.constexpr,
):
    i_nh = tl.program_id(0)
    i_n = i_nh // HV
    i_hv = i_nh - i_n * HV
    i_h = i_hv // HEADS_PER_V

    state_idx = tl.load(state_indices + i_n).to(tl.int64)
    if state_idx < 0:
        return

    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)

    qk_base = (i_n * H + i_h) * K
    nh_off = i_n * HV + i_hv
    state_base = ((state_idx * HV + i_hv) * K) * V
    v_base = nh_off * V

    # Q/K/scalar values are shared by the two 64-wide V halves. Processing the
    # halves sequentially removes duplicate q/k/scalar loads without holding a
    # 128-wide state tile live, avoiding the register-pressure regression of a
    # monolithic BV=128 program.
    q_raw = tl.load(q + qk_base + o_k, cache_modifier=".ca")
    k_raw = tl.load(k + qk_base + o_k, cache_modifier=".ca")
    a_val = tl.load(a + nh_off)
    dt_val = tl.load(dt_bias + i_hv)
    A_log_val = tl.load(A_log + i_hv)
    b_val = tl.load(b + nh_off)

    q_vec = q_raw.to(tl.float32)
    k_vec = k_raw.to(tl.float32)

    x = a_val.to(tl.float32) + dt_val.to(tl.float32)
    softplus_x = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
    gate = -tl.exp(A_log_val.to(tl.float32)) * softplus_x
    beta = tl.sigmoid(b_val.to(tl.float32))
    exp_gate = tl.exp(gate)

    q_norm = tl.rsqrt(tl.sum(q_vec * q_vec, axis=0) + 1.0e-6) * scale
    k_norm = tl.rsqrt(tl.sum(k_vec * k_vec, axis=0) + 1.0e-6)
    q_vec = q_vec * q_norm
    k_vec = k_vec * k_norm

    state_offsets0 = state_base + o_k[:, None] * V + o_v[None, :]
    h0 = tl.load(
        state + state_offsets0,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    ).to(tl.float32)
    v0 = tl.load(v + v_base + o_v, cache_modifier=".ca").to(tl.float32)
    h0 = h0 * exp_gate
    delta0 = tl.sum(h0 * k_vec[:, None], axis=0)
    v0 = (v0 - delta0) * beta
    h0 = h0 + k_vec[:, None] * v0[None, :]
    out0 = tl.sum(h0 * q_vec[:, None], axis=0)
    tl.store(out + v_base + o_v, out0.to(out.dtype.element_ty))
    tl.store(state + state_offsets0, h0.to(state.dtype.element_ty), cache_modifier=".cg")

    o_v1 = o_v + BV
    state_offsets1 = state_base + o_k[:, None] * V + o_v1[None, :]
    h1 = tl.load(
        state + state_offsets1,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    ).to(tl.float32)
    v1 = tl.load(v + v_base + o_v1, cache_modifier=".ca").to(tl.float32)
    h1 = h1 * exp_gate
    delta1 = tl.sum(h1 * k_vec[:, None], axis=0)
    v1 = (v1 - delta1) * beta
    h1 = h1 + k_vec[:, None] * v1[None, :]
    out1 = tl.sum(h1 * q_vec[:, None], axis=0)
    tl.store(out + v_base + o_v1, out1.to(out.dtype.element_ty))
    tl.store(state + state_offsets1, h1.to(state.dtype.element_ty), cache_modifier=".cg")


def gdn_decode_update_fast(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor | None = None,
    beta: float = 1.0,
    threshold: float = 20.0,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
) -> torch.Tensor:
    _, n_tokens, n_heads, head_k = q.shape
    _, _, n_value_heads, head_v = v.shape
    out = torch.empty_like(v).squeeze(0) if o is None else o.squeeze(0)

    if beta != 1.0 or threshold != 20.0:
        raise ValueError("gdn_decode_update_fast supports beta=1.0 and threshold=20.0")
    if not use_qk_l2norm_in_kernel:
        raise ValueError("gdn_decode_update_fast requires use_qk_l2norm_in_kernel=True")
    if is_kda:
        raise ValueError("gdn_decode_update_fast does not support KDA gating")
    if initial_state is None or ssm_state_indices is None:
        raise ValueError("gdn_decode_update_fast requires initial_state and ssm_state_indices")
    if not inplace_final_state:
        raise ValueError("gdn_decode_update_fast only supports inplace_final_state=True")
    if cu_seqlens is None:
        raise ValueError("gdn_decode_update_fast requires cu_seqlens for decode")
    if num_accepted_tokens is not None:
        raise ValueError("gdn_decode_update_fast does not support spec decoding")
    if scale is None:
        scale = head_k**-0.5

    bv = 64
    bk = triton.next_power_of_2(head_k)
    grid = (triton.cdiv(head_k, bk), triton.cdiv(head_v, bv), n_tokens * n_value_heads)
    _gdn_decode_update_kernel[grid](
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        out,
        initial_state,
        ssm_state_indices,
        scale,
        n_heads,
        n_value_heads,
        head_k,
        head_v,
        bk,
        bv,
        n_value_heads // n_heads,
        num_warps=4,
        num_stages=1,
    )
    return out.unsqueeze(0)
