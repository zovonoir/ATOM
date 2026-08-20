# Known issue: MoE forward is not deterministic

**Status:** open, not caused by ATOM's plugin code. Root cause is below ATOM,
in the MoE kernels. Recorded here with a standalone reproducer so it can be
filed against aiter without needing this stack.

**Reproduce:** `python3 repro_moe_nondeterminism.py` (needs only torch + aiter,
no model, no server). `probe_determinism.py` is the server-level probe used for
the bisection.

---

## Temperature-0 nondeterminism: bisection

Identical greedy requests, sent sequentially at concurrency 1, return different
text. Batch composition is fixed and nothing is cached between them, so the
output should be a function of the input alone.

Measured with `probe_determinism.py`, which asks for **logprobs** rather than
comparing text. Text only moves when a token is near-tied, so it conflates "the
numerics changed" with "the numerics changed enough to flip an argmax".
`--mode prefill` requests one token with `logprob_start_len=0`, so the reported
input logprobs come from a single prefill forward with no decode involved.

Stack: v3 image (sglang v0.5.17 + ATOM `atom-plugin-dflash-enable`), 8x MI308X
(gfx942), TP8, radix cache off, DFLASH off.

## Bisection

| model | MoE | dtype | prefill logprobs over 5 runs |
|---|---|---|---|
| Qwen3.5-27B | dense | bf16 | **identical, delta exactly 0** |
| Qwen3.5-35B-A3B | MoE | bf16 | 5 distinct, delta up to 2.1 nats |
| Qwen3.5-397B-A17B | MoE | fp8 | 5 distinct, delta up to 1.6 nats |

Dense is deterministic and MoE is not, at every prompt length tested (8, 128,
2048 tokens). An 8-token prompt is already affected, so this is not an
accumulation effect over sequence length or depth. The bf16 MoE model shows it
too, so quantization is not the cause.

## Kernel-level repro

`repro_moe_nondeterminism.py` calls `aiter.fused_moe` five times on the same
tensors, back to back, on one GPU. No model, no server. Routing is computed
once and passed in, so expert selection cannot be the variable -- any
difference is downstream of it.

| tokens | bit-identical to run 0 | elements differing | max relative |
|---:|---|---|---:|
| 1 | 0/4 | 1021/2048 (49.9%) | 0.87% |
| 8 | 0/4 | 2946/16384 (18.0%) | 0.54% |
| 64 | 0/4 | 1317/131072 (1.0%) | 0.47% |
| 512 | 0/4 | 18475/1048576 (1.8%) | 0.52% |

A single token through the MoE already gives half its outputs differently on a
repeat call. Per-layer perturbation of this size, compounded over the model's
depth, is consistent with the nat-scale logprob spread seen end to end.

## What has been ruled out

- **Quantization** -- bf16 MoE shows it.
- **Sequence length / depth accumulation** -- 8-token prompts show it.
- **Speculative decoding** -- present with DFLASH off.
- **Radix cache** -- present with the cache disabled.
- **SGLang version** -- present on the delivered v2 image (v0.5.15) as well,
  so it is not a regression from the v0.5.17 port. Note the v2 evidence is
  text-level (10 identical requests, 6/4 split into two distinct answers), not
  logprob-level; the probe was only run on the v3 stack.
- **`AITER_MOE_SMALL_BATCH`** -- setting it to 0 changes nothing.
- **`ATOM_USE_TRITON_MOE=1`** -- changes nothing, and cannot: `moe.py` already
  defaults `use_triton` to True on `gfx94*`, so the Triton MoE was the path in
  use all along. So the production path on MI308X is ATOM's Triton MoE, not
  `aiter.fused_moe`. Both are confirmed affected: the server test above
  exercises the Triton path, and the standalone repro below exercises
  `aiter.fused_moe` directly.

## Not yet established

- Forcing the aiter path (`ATOM_USE_TRITON_MOE=0`) to compare the two
  implementations head to head: the server failed to start on that config
  (memory check), so this control is **untested**.
- Which step inside the MoE is responsible. The `moe_sorting_opus` kernel that
  both paths call is the obvious next place to look: if it groups tokens per
  expert in an order that varies run to run, the downstream GEMM accumulates in
  a different order and the output moves by a few ULP -- which matches the
  observed magnitude.
- Whether `sglang`'s own MoE (without the ATOM plugin) is affected. That would
  say whether this is an ATOM/aiter issue or reaches upstream.

## Impact

Any expectation of byte-reproducible output at temperature 0 does not hold on
this stack for MoE models. Concretely this makes single-run accuracy comparisons
unreliable: the same GSM8K config over the same 300 questions moved by up to 10
points between runs (see `ACCURACY.md`), which is far larger than any regression
worth detecting.
