# SPDX-License-Identifier: MIT
# Contract tests for the per-chunk recurrent states returned by
# `chunk_gated_delta_rule(..., return_intermediate_states=True)`.
#
# The mamba radix cache's `extra_buffer` strategy snapshots GDN state at chunk
# boundaries so a cached prefix can be resumed. SGLang's
# `MambaAttnBackendBase._track_mamba_state_extend` does a plain gather-copy,
#
#     ssm_states[track_ssm_h_dst] = h.squeeze(0)[track_ssm_h_src]
#
# with `track_ssm_h_src` derived from `cumsum(cdiv(extend_seq_lens, 64))`. That
# copy is layout-agnostic, so it is only correct if `h` is in the *same*
# convention as the state pool. The pool's convention is whatever ATOM writes
# to and reads from it -- i.e. `final_state`. For Qwen3.5 the GDN key and value
# head dims are both 128, so a transposed `h` would have an identical shape and
# would corrupt state silently. These tests pin the contract down by value, not
# by shape.

import pytest
import torch

pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from atom.model_ops.fla_ops import chunk_gated_delta_rule
from atom.model_ops.fla_ops.index import prepare_chunk_offsets

CHUNK = 64
# Qwen3.5-397B-A17B GDN geometry, scaled down in head count only.
NUM_K_HEADS = 2
NUM_V_HEADS = 8
HEAD_K_DIM = 128
HEAD_V_DIM = 128
DTYPE = torch.bfloat16


def _inputs(seq_lens, seed=0):
    """Flattened varlen GDN inputs, one row per token."""
    torch.manual_seed(seed)
    total = sum(seq_lens)
    dev = "cuda"
    q = torch.randn(1, total, NUM_K_HEADS, HEAD_K_DIM, device=dev, dtype=DTYPE)
    k = torch.randn(1, total, NUM_K_HEADS, HEAD_K_DIM, device=dev, dtype=DTYPE)
    v = torch.randn(1, total, NUM_V_HEADS, HEAD_V_DIM, device=dev, dtype=DTYPE)
    # g is a log-space forget gate; keep it mildly negative like the real model.
    g = -torch.rand(1, total, NUM_V_HEADS, device=dev, dtype=torch.float32)
    beta = torch.rand(1, total, NUM_V_HEADS, device=dev, dtype=DTYPE).sigmoid()
    cu = torch.tensor(
        [0] + list(torch.tensor(seq_lens).cumsum(0)), device=dev, dtype=torch.long
    )
    return q, k, v, g, beta, cu


def _run(seq_lens, seed=0, want_h=False):
    q, k, v, g, beta, cu = _inputs(seq_lens, seed)
    n = len(seq_lens)
    init = torch.zeros(
        n, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, device="cuda", dtype=torch.float32
    )
    out = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=init,
        output_final_state=True,
        cu_seqlens=cu,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
        return_intermediate_states=want_h,
    )
    return out


def test_default_return_is_unchanged():
    """Existing callers (vLLM / rtpllm plugins) must keep seeing a 2-tuple."""
    out = _run([128])
    assert len(out) == 2


def test_intermediate_states_shape_and_offsets():
    seq_lens = [128, 192]
    _, _, h = _run(seq_lens, want_h=True)

    cu = torch.tensor([0] + list(torch.tensor(seq_lens).cumsum(0)), device="cuda")
    offsets = prepare_chunk_offsets(cu, CHUNK)
    total_chunks = int(offsets[-1])

    assert h.shape[0] == 1, "varlen inputs are flattened into batch 1"
    assert h.shape[1] == total_chunks == (128 // CHUNK + 192 // CHUNK)
    # Per-chunk slot must match a state-pool slot exactly, else the
    # gather-copy in _track_mamba_state_extend would broadcast or fail.
    assert h.shape[2:] == (NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM)


@pytest.mark.parametrize("seq_lens", [[128, 192], [256]])
def test_intermediate_state_equals_truncated_final_state(seq_lens):
    """`h[offset_i + j]` == the final state of sequence i truncated to j chunks.

    This is the property `extra_buffer` relies on: a snapshot taken at a chunk
    boundary must be a state the model could have arrived at by processing only
    the prefix up to that boundary. It simultaneously pins down (a) the chunk
    indexing, (b) that h[j] is the state *entering* chunk j, and (c) that h
    shares `final_state`'s layout -- a transposed h fails this by value.
    """
    _, _, h = _run(seq_lens, want_h=True)
    h = h.squeeze(0)

    cu = torch.tensor([0] + list(torch.tensor(seq_lens).cumsum(0)), device="cuda")
    offsets = prepare_chunk_offsets(cu, CHUNK)

    for i, seq_len in enumerate(seq_lens):
        num_chunks = seq_len // CHUNK
        # j == 0 is the (zero) initial state; j == num_chunks is final_state and
        # has no h row. The interesting boundaries are strictly in between.
        for j in range(1, num_chunks):
            # Re-run sequence i alone, truncated to j complete chunks. Same seed
            # and same lengths => byte-identical tokens for the prefix.
            q, k, v, g, beta, _ = _inputs(seq_lens, seed=0)
            start = int(cu[i])
            stop = start + j * CHUNK
            trunc_cu = torch.tensor([0, j * CHUNK], device="cuda", dtype=torch.long)
            init = torch.zeros(
                1,
                NUM_V_HEADS,
                HEAD_K_DIM,
                HEAD_V_DIM,
                device="cuda",
                dtype=torch.float32,
            )
            _, final_state = chunk_gated_delta_rule(
                q=q[:, start:stop],
                k=k[:, start:stop],
                v=v[:, start:stop],
                g=g[:, start:stop],
                beta=beta[:, start:stop],
                initial_state=init,
                output_final_state=True,
                cu_seqlens=trunc_cu,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )

            got = h[int(offsets[i]) + j].float()
            want = final_state[0].float()
            torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)


def test_transposed_layout_would_be_caught():
    """Guard the guard: the value check above must actually reject a transpose.

    K == V == 128 for Qwen3.5, so shape checks cannot see this. If this test
    ever passes with the transpose applied, the assertion above is vacuous.
    """
    _, _, h = _run([128], want_h=True)
    row = h.squeeze(0)[1].float()
    transposed = row.transpose(-1, -2)
    assert not torch.allclose(row, transposed, rtol=2e-2, atol=2e-2)
