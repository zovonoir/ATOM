#!/usr/bin/env python3
"""Acceptance check for GDN prefix caching: a cache hit must not change output.

Not a pytest unit test -- it drives a running SGLang server, hence the
`check_` prefix so collection skips it. Run it against a server started with
`--mamba-radix-cache-strategy extra_buffer --page-size 64`.

The failure mode this guards against is silent. A hybrid GDN model's radix node
carries both KV pages and a recurrent state snapshot; if the snapshot is stale
or was never taken, a prefix hit resumes from the wrong state and the model
keeps generating fluent-but-wrong text. Accuracy benchmarks blur that into
noise, so compare byte-for-byte instead:

  warm: [flush] -> P+S1 (populates the tree) -> P+S2 (hits the P prefix)
  cold: [flush] -> P+S2 (no prefix to hit)

Confirm the warm arm really hit: the server log must show a non-zero
`#cached-token` per warm request. Without a hit the two arms are the same
computation and the check passes vacuously.

At temperature 0 the two P+S2 answers must be identical. Prefix lengths are
chosen to land both on and off a 64-token chunk boundary, since the snapshot
source differs between the two (`final_state` vs an interior `h` row).

Usage: check_gdn_prefix_reuse.py [--port 31000] [--prefix-tokens 2048 2080]
"""

import argparse
import json
import sys
import urllib.request

PORT = 31000


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


def gen(prompt, port, max_new_tokens=64):
    out = post(
        "/generate",
        {
            "text": prompt,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
            },
        },
        port=port,
    )
    return out["text"]


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
    args = ap.parse_args()

    s1 = "\n\nQuestion A: how many items are listed above? Answer briefly."
    s2 = "\n\nQuestion B: name the single largest recorded value. Answer briefly."

    failures = 0
    for approx in args.prefix_tokens:
        prefix = build_prefix(approx)

        post("/flush_cache", {}, port=args.port)
        gen(prefix + s1, args.port)  # populate the tree with P
        warm = gen(prefix + s2, args.port)  # should hit the P prefix

        post("/flush_cache", {}, port=args.port)
        cold = gen(prefix + s2, args.port)  # no prefix to hit

        ok = warm == cold
        failures += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] prefix~{approx} tokens")
        if not ok:
            print(f"    warm (cache hit): {warm[:220]!r}")
            print(f"    cold (no cache) : {cold[:220]!r}")

    print("\nRESULT:", "all prefixes match" if not failures else f"{failures} mismatch")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
