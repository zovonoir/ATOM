#!/usr/bin/env python3
"""Locate the source of run-to-run nondeterminism at temperature 0.

Text-level comparison is a poor probe: it only moves when a token is near-tied,
so it conflates "the numerics changed" with "the numerics changed enough to
flip an argmax". This asks the server for logprobs instead, which move whenever
the numerics move at all.

`--mode prefill` requests one token with `logprob_start_len=0`, so the reported
input_token_logprobs come from a single prefill forward with no decode step
involved. If those differ across identical requests, prefill alone is
nondeterministic.

`--mode decode` generates N tokens and compares the output logprobs, which
carry prefill and decode together.

Usage: probe_determinism.py [--mode prefill|decode] [--lengths 8 32 128 512]
                            [--repeats 5] [--port 31000]
"""

import argparse
import json
import sys
import urllib.request


def post(payload, port, timeout=600):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def build_prompt(approx_tokens):
    words, n, i = [], 0, 0
    while n < approx_tokens:
        words.append(f"Item {i} records value {(i * 7919) % 10007}.")
        n += 9
        i += 1
    return " ".join(words)


def prefill_logprobs(prompt, port):
    out = post(
        {
            "text": prompt,
            "sampling_params": {"temperature": 0, "max_new_tokens": 1},
            "return_logprob": True,
            "logprob_start_len": 0,
        },
        port,
    )
    return [x[0] for x in out["meta_info"]["input_token_logprobs"] if x[0] is not None]


def decode_logprobs(prompt, port, n=8):
    out = post(
        {
            "text": prompt,
            "sampling_params": {"temperature": 0, "max_new_tokens": n},
            "return_logprob": True,
        },
        port,
    )
    return [x[0] for x in out["meta_info"]["output_token_logprobs"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=31000)
    ap.add_argument("--mode", choices=["prefill", "decode"], default="prefill")
    ap.add_argument("--lengths", type=int, nargs="+", default=[8, 32, 128, 512, 2048])
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    print(f"mode={args.mode} repeats={args.repeats}")
    print(f"{'approx tokens':>14} {'distinct':>9} {'max abs delta':>14} {'first delta at':>15}")
    for approx in args.lengths:
        prompt = build_prompt(approx)
        runs = []
        for _ in range(args.repeats):
            if args.mode == "prefill":
                runs.append(prefill_logprobs(prompt, args.port))
            else:
                runs.append(decode_logprobs(prompt, args.port))

        distinct = len({tuple(r) for r in runs})
        base = runs[0]
        worst, first_idx = 0.0, None
        for r in runs[1:]:
            for i, (a, b) in enumerate(zip(base, r)):
                d = abs(a - b)
                if d > 0 and first_idx is None:
                    first_idx = i
                worst = max(worst, d)
        pos = "-" if first_idx is None else str(first_idx)
        print(f"{approx:>14} {distinct:>9} {worst:>14.3e} {pos:>15}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
