"""Converts the Lichess evaluation database (CC0) into training positions.

Reads lichess_db_eval.jsonl (decompressed, on stdin) and writes "fen,score" lines: the score is
the deepest Stockfish evaluation of the position in centipawns, from White's point of view.
Positions with a forced mate are left out (their scores are not on the centipawn scale).

    curl -s https://database.lichess.org/lichess_db_eval.jsonl.zst | zstd -dc |
        python -m rookery_train.fetch_evals --limit 12000000 > data/evals.csv
"""
from __future__ import annotations

import argparse
import json
import sys


def best(record: dict) -> tuple[int, int] | None:
    """(depth, centipawns) of the deepest evaluation, or None if it is a mate."""
    top = max(record["evals"], key=lambda e: e["depth"], default=None)
    if top is None or not top["pvs"]:
        return None
    pv = top["pvs"][0]
    if "cp" not in pv:
        return None
    return top["depth"], int(pv["cp"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=12_000_000)
    ap.add_argument("--min-depth", type=int, default=20)
    ap.add_argument("--skip", type=int, default=0, help="positions to pass over first (for a held-out test set)")
    args = ap.parse_args()
    out = sys.stdout
    kept = 0
    for line in sys.stdin:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        found = best(record)
        if found is None or found[0] < args.min_depth:
            continue
        if args.skip:
            args.skip -= 1
            continue
        out.write(f"{record['fen']},{found[1]}\n")
        kept += 1
        if kept % 1_000_000 == 0:
            print(f"{kept:,} positions", file=sys.stderr, flush=True)
        if kept >= args.limit:
            break


if __name__ == "__main__":
    main()
