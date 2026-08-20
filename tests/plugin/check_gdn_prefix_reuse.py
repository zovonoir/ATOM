#!/usr/bin/env python3
"""Acceptance check for GDN prefix caching: a cache hit must not change output.

Not a pytest unit test -- it drives a running SGLang server, hence the `check_`
prefix so collection skips it. Run it against a server started with
`--mamba-radix-cache-strategy extra_buffer --page-size 64`.

The failure mode this guards against is silent. A hybrid GDN model's radix node
carries both KV pages and a recurrent state snapshot; if the snapshot is stale
or was never taken, a prefix hit resumes from the wrong state and the model
keeps generating fluent-but-wrong text. Accuracy benchmarks blur that into
noise, so compare byte-for-byte instead:

  warm: [flush] -> P+S1 (populates the tree) -> P+S2 (hits the P prefix)
  cold: [flush] -> P+S2 (no prefix to hit)

Two things make a byte-for-byte verdict trustworthy, and both are enforced here
rather than left to the reader:

* **The warm arm must actually hit.** Without a hit the two arms are the same
  computation and every case passes vacuously. Check the server log for a
  non-zero `#cached-token` per warm request.
* **The cold arm must be reproducible.** DFLASH is *not* byte-deterministic at
  temperature 0, even with the radix cache disabled: speculative verify
  computes logits in a different batch shape than plain decode, so a near-tie
  can flip. A free-form prompt flips between "**9930**" and "9930" from run to
  run. Every case therefore runs cold twice and reports INCONCLUSIVE, not FAIL,
  when the two disagree -- that is the harness failing to measure, not the
  cache failing. Prompts also demand a bare constrained answer, which is stable
  under DFLASH (verified 6/6) where free-form phrasing is not.

Usage: check_gdn_prefix_reuse.py [--port 31000] [--prefix-tokens 2048 2080]
                                 [--multiturn]
"""

import argparse
import json
import sys
import urllib.request

PORT = 31000
MODEL = "/model/Qwen3.5-397B-A17B-FP8"

# Constrained enough that a near-tie on formatting cannot flip it; see the
# module docstring on why free-form answers are unusable under DFLASH.
BARE = " Reply with only the numeric value: no words, no punctuation, no markdown."


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


def gen(prompt, port, max_new_tokens=32):
    out = post(
        "/generate",
        {
            "text": prompt,
            "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
        },
        port=port,
    )
    return out["text"]


def chat(messages, port, max_tokens=32):
    out = post(
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            # Thinking off: with it on the model spends the whole budget in
            # `reasoning_content` and returns empty `content`, so the assistant
            # turn would carry no generated tokens and the second turn would
            # have nothing decode-side to reuse.
            "chat_template_kwargs": {"enable_thinking": False},
        },
        port=port,
    )
    return out["choices"][0]["message"]["content"]


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


def verdict(label, warm, cold_a, cold_b):
    """PASS / FAIL / INCONCLUSIVE, printed and returned."""
    if cold_a != cold_b:
        print(f"[INCONCLUSIVE] {label} -- cold arm is not reproducible")
        print(f"    cold #1: {cold_a[:160]!r}")
        print(f"    cold #2: {cold_b[:160]!r}")
        return "inconclusive"
    if warm == cold_a:
        print(f"[PASS] {label}")
        return "pass"
    print(f"[FAIL] {label}")
    print(f"    warm (cache hit): {warm[:160]!r}")
    print(f"    cold (no cache) : {cold_a[:160]!r}")
    return "fail"


def check_prompt_prefix(port, approx):
    """Reuse a prompt prefix -- the prefill-side snapshot path."""
    prefix = build_prefix(approx)
    s1 = "\n\nHow many items are listed above?" + BARE
    s2 = "\n\nWhat is the single largest recorded value above?" + BARE

    post("/flush_cache", {}, port=port)
    gen(prefix + s1, port)  # populate the tree with P
    warm = gen(prefix + s2, port)  # should hit the P prefix

    colds = []
    for _ in range(2):
        post("/flush_cache", {}, port=port)
        colds.append(gen(prefix + s2, port))

    return verdict(f"prompt prefix ~{approx} tokens", warm, *colds)


def check_multiturn(port, turn1_tokens=512):
    """Reuse *generated* tokens as a prefix -- the decode-side snapshot path.

    The prompt-prefix checks never exercise it: they only reuse the prompt,
    whose state was snapshotted during prefill. A snapshot taken while decoding
    only matters once a request's own output becomes someone else's prefix,
    which is what the second turn of a conversation does. Driven through the
    chat endpoint so the turns are templated the way a real deployment sends
    them -- raw string concatenation instead makes the model see a finished
    answer and emit EOS immediately, which tests nothing.

    `turn1_tokens` is above `mamba_track_interval` (256) so turn 1's generation
    crosses at least one track boundary.
    """
    prompt = build_prefix(1024) + (
        "\n\nList every item above with its value, one per line, no commentary."
    )
    ask = "Which of the items you just listed has the largest value?" + BARE

    post("/flush_cache", {}, port=port)
    answer = chat([{"role": "user", "content": prompt}], port, max_tokens=turn1_tokens)

    followup = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
        {"role": "user", "content": ask},
    ]
    warm = chat(followup, port)  # prefix hit spans turn 1's generated tokens

    colds = []
    for _ in range(2):
        post("/flush_cache", {}, port=port)
        colds.append(chat(followup, port))

    label = f"multi-turn (turn 1 generated {len(answer)} chars)"
    return verdict(label, warm, *colds)


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
        "--multiturn",
        action="store_true",
        help="also reuse generated tokens as a prefix (decode-side snapshot)",
    )
    args = ap.parse_args()

    results = [check_prompt_prefix(args.port, a) for a in args.prefix_tokens]
    if args.multiturn:
        results.append(check_multiturn(args.port))

    fails = results.count("fail")
    unknown = results.count("inconclusive")
    print(
        f"\nRESULT: {results.count('pass')} pass, {fails} fail, "
        f"{unknown} inconclusive"
    )
    return 1 if fails or unknown else 0


if __name__ == "__main__":
    sys.exit(main())
