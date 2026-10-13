#!/usr/bin/env python3
"""Acceptance check for GDN prefix caching: a cache hit must not change output.

Not a pytest unit test -- it drives a running SGLang server, hence the `check_`
prefix so collection skips it. Run it against a server started with
`--mamba-radix-cache-strategy extra_buffer --page-size 64`.

The failure mode this guards against is silent. A hybrid GDN model's radix node
carries both KV pages and a recurrent state snapshot; if the snapshot is stale
or was never taken, a prefix hit resumes from the wrong state and the model
keeps generating fluent-but-wrong text. Accuracy benchmarks blur that into
noise, so each case compares a warm arm (resumed from a snapshot) against a
cold arm (no prefix to hit) on the same token ids, by the teacher-forced
log-probs of those tokens rather than by generated text:

* Generated text is a poor oracle. Greedy decoding turns any numerical
  difference into a different token at the first near-tie, and both arms
  legitimately differ numerically: the warm arm prefills only the suffix, in
  a different batch shape, and MoE targets (Qwen3.5-397B-A17B) are not even
  run-to-run deterministic at temperature 0.
* Log-probs are compared against a noise level measured on the same server,
  so the verdict means the same thing on a deterministic 9B and on a
  non-deterministic MoE.

There are two snapshot paths and each gets its own case:

* **Prompt prefix** (prefill-side snapshot):

    warm: [flush] -> P (populates the tree) -> P+S+T (hits the P prefix)
    cold: [flush] -> P+S+T (twice; their gap is the noise level)

  T is a 200-token greedy continuation, so the score covers generated-style
  text and not just the question.

* **Generated prefix** (decode-side snapshot, `--decode-snapshot`). A request's
  own output becomes the next request's prefix, so the state at the hit was
  produced by the recurrent *decode* kernel while the cold arm recomputes it
  with the *prefill* kernel. The noise level therefore also includes that
  kernel gap, measured without any cache: the first request's own decode
  log-probs versus the cold prefill's. Token ids are passed verbatim so the
  radix match cannot break on re-templating, and the output is free-form
  prose -- a predictable list scores near-zero log-probs whatever the state.

A verdict is only reported when it means something:

* **The warm arm must actually hit**, and for the decode case hit past the
  prompt with enough generated tokens left to score. Otherwise both arms are
  the same computation; the case is reported VACUOUS.
* **Anything not PASS exits non-zero.**

Usage: check_gdn_prefix_reuse.py [--port 31000] [--prefix-tokens 2048 2080]
                                 [--decode-snapshot]
"""

import argparse
import json
import sys
import time
import urllib.request

PORT = 31000

# Warm may exceed the measured noise level by this factor plus an absolute
# slack (mean |dlogprob| over the scored tokens). The slack covers the
# batch-shape gap between a suffix-only and a full prefill, which cold/cold
# cannot see. Qwen3.5-397B-A17B-FP8 TP4, SGLang v0.5.20: correct snapshots
# scored 0.006-0.017; a missing prefill-side snapshot 0.081-0.143; a missing
# decode-side snapshot 0.156-0.438.
NOISE_FACTOR = 2.0
NOISE_SLACK = 0.03


def post(path, payload=None, port=PORT, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else b""
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


def flush(port):
    # Right after startup, warmup requests can still be in flight and
    # /flush_cache is refused; retry instead of treating that as an error.
    for _ in range(60):
        try:
            post("/flush_cache", {}, port=port)
            return
        except Exception:  # noqa: BLE001
            time.sleep(1)
    raise RuntimeError("/flush_cache never succeeded")


def generate(port, ids, max_new_tokens=1, logprob_start=None):
    payload = {
        "input_ids": ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
    }
    if logprob_start is not None:
        payload.update(return_logprob=True, logprob_start_len=logprob_start)
    return post("/generate", payload, port=port)


def tokenize(text, port):
    return post("/tokenize", {"prompt": text}, port=port)["tokens"]


def cached(res):
    return res["meta_info"].get("cached_tokens") or 0


def input_logprobs(res, n):
    """Log-probs of the `n` tokens after `logprob_start`."""
    return [x[0] for x in res["meta_info"]["input_token_logprobs"]][1 : n + 1]


def mean_gap(a, b):
    return sum(abs(x - y) for x, y in zip(a, b, strict=True)) / len(a)


def build_prefix(approx_tokens):
    """Deterministic, non-repeating filler so the radix key is a real prefix."""
    words = []
    n = 0
    i = 0
    while n < approx_tokens:
        words.append(f"Item {i} records value {(i * 7919) % 10007}.")
        n += 9
        i += 1
    return " ".join(words)


def report(status, label, *details):
    print(f"[{status.upper()}] {label}")
    for line in details:
        print(f"    {line}")
    return status


def verdict(label, warm_gap, noise, stats):
    if warm_gap <= NOISE_FACTOR * noise + NOISE_SLACK:
        return report("pass", label, stats)
    return report("fail", label, stats)


def cold_pair(port, ids, start, n):
    colds = []
    for _ in range(2):
        flush(port)
        colds.append(input_logprobs(generate(port, ids, logprob_start=start), n))
    return colds


def check_prompt_prefix(port, approx):
    """Reuse a prompt prefix -- the prefill-side snapshot path."""
    label = f"prompt prefix ~{approx} tokens"
    p_ids = tokenize(build_prefix(approx), port)
    s_ids = tokenize(
        "\n\nList the first twenty items above with their values, one per line:\n",
        port,
    )
    flush(port)
    t_ids = generate(port, p_ids + s_ids, max_new_tokens=200)["output_ids"]
    full = p_ids + s_ids + t_ids
    # Scoring starts inside P, so the match is capped there and lands on the
    # snapshot P left behind rather than on anything past it.
    start = len(p_ids) - 1
    n = len(full) - start - 1

    flush(port)
    # Populating with P alone puts a snapshot at or below len(P), which every
    # longer request sharing P is guaranteed to reach.
    generate(port, p_ids)
    warm = generate(port, full, logprob_start=start)
    if not cached(warm):
        return report("vacuous", label, "warm arm did not hit the prefix")
    w = input_logprobs(warm, n)
    c0, c1 = cold_pair(port, full, start, n)

    noise = mean_gap(c0, c1)
    warm_gap = mean_gap(w, c0)
    stats = (
        f"hit {cached(warm)} of {len(full)}, {n} scored tokens, mean |dlogprob|: "
        f"warm {warm_gap:.4f}, cold/cold {noise:.4f}"
    )
    return verdict(label, warm_gap, noise, stats)


def check_decode_snapshot(port, generated=700, min_scored=160):
    """Reuse a request's *generated* tokens -- the decode-side snapshot path."""
    label = "generated prefix (decode-side snapshot)"
    p_ids = tokenize(
        build_prefix(1024)
        + "\n\nWrite a long, detailed essay about the patterns in these values.\n",
        port,
    )
    q_ids = tokenize("\nIn one sentence, the main pattern is", port)

    # Where the decode-side snapshot lands is SGLang's choice
    # (mamba_track_interval); ask the cache instead of assuming it, and keep at
    # least `min_scored` generated tokens after it.
    for attempt in range(3):
        if attempt:
            generated += min_scored
        flush(port)
        r1 = generate(
            port, p_ids, max_new_tokens=generated, logprob_start=len(p_ids) - 1
        )
        out = r1["output_ids"]
        r2 = p_ids + out + q_ids
        hit = cached(generate(port, r2))
        if hit > len(p_ids) and len(p_ids) + len(out) - hit >= min_scored:
            break
    else:
        return report(
            "vacuous",
            label,
            f"hit {hit} does not reach past the {len(p_ids)}-token prompt "
            f"with {min_scored} generated tokens left to score",
        )
    decode_lp = [x[0] for x in r1["meta_info"].get("output_token_logprobs") or []]
    if len(decode_lp) != len(out) or any(v is None for v in decode_lp):
        return report("inconclusive", label, "server returned no decode log-probs")

    # The probe above inserted r2's own snapshot past the generated tokens;
    # capping the match just after the hit makes the warm arm resume from the
    # decode-side snapshot again.
    start = hit + 8
    n = len(p_ids) + len(out) - start - 1
    warm = generate(port, r2, logprob_start=start)
    if cached(warm) != hit:
        return report("vacuous", label, f"warm arm hit {cached(warm)}, expected {hit}")
    w = input_logprobs(warm, n)
    c0, c1 = cold_pair(port, r2, start, n)

    first = start + 1 - len(p_ids)
    kernel_gap = mean_gap(decode_lp[first : first + n], c0)
    cold_gap = mean_gap(c0, c1)
    warm_gap = mean_gap(w, c0)
    stats = (
        f"hit {hit} of {len(r2)}, {n} scored tokens, mean |dlogprob|: "
        f"warm {warm_gap:.4f}, decode/prefill {kernel_gap:.4f}, "
        f"cold/cold {cold_gap:.4f}"
    )
    return verdict(label, warm_gap, max(kernel_gap, cold_gap), stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument(
        "--prefix-tokens",
        type=int,
        nargs="+",
        default=[2048, 2080],
        help="approx prefix lengths; include one that is not a multiple of 64",
    )
    ap.add_argument(
        "--decode-snapshot",
        action="store_true",
        help="also reuse generated tokens as a prefix (decode-side snapshot)",
    )
    args = ap.parse_args()

    results = [check_prompt_prefix(args.port, a) for a in args.prefix_tokens]
    if args.decode_snapshot:
        results.append(check_decode_snapshot(args.port))

    passed = results.count("pass")
    print(
        f"\nRESULT: {passed} pass, {results.count('fail')} fail, "
        f"{results.count('inconclusive')} inconclusive, "
        f"{results.count('vacuous')} vacuous"
    )
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
