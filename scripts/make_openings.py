"""Builds a set of balanced opening positions for engine matches.

Each candidate is the position after a few random legal moves from the start (Rookery's
`randomfen`); Stockfish then evaluates it, and only positions within +-50 centipawns are kept,
so that games start from fair, varied positions. Every opening is played twice in a match, once
with each colour.

    python scripts/make_openings.py --count 300 --out results/openings.epd
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = ".exe" if sys.platform == "win32" else ""


class Uci:
    def __init__(self, path: Path):
        self.p = subprocess.Popen([str(path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.send("uci")
        self.wait("uciok")

    def send(self, line: str) -> None:
        self.p.stdin.write(line + "\n")

    def wait(self, prefix: str) -> list[str]:
        lines = []
        for line in self.p.stdout:
            lines.append(line.strip())
            if line.startswith(prefix):
                return lines
        raise RuntimeError("engine closed")

    def score(self, fen: str, depth: int) -> int | None:
        self.send(f"position fen {fen}")
        self.send(f"go depth {depth}")
        last = None
        for line in self.wait("bestmove"):
            words = line.split()
            if "score" in words:
                i = words.index("score")
                last = int(words[i + 2]) if words[i + 1] == "cp" else None
        return last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--plies", type=int, default=8)
    ap.add_argument("--depth", type=int, default=14)
    ap.add_argument("--rookery", type=Path, default=ROOT / "engine" / "build" / f"rookery{EXE}")
    ap.add_argument("--stockfish", type=Path, default=next((ROOT / "third_party" / "stockfish").rglob(f"stockfish*{EXE}"), None))
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "openings.epd")
    args = ap.parse_args()
    sf = Uci(args.stockfish)
    kept, seen = [], set()
    seed = 0
    while len(kept) < args.count:
        seed += 1
        fen = subprocess.run([str(args.rookery), "randomfen", str(args.plies), str(seed)], capture_output=True,
                             text=True).stdout.strip().splitlines()[-1]
        board = " ".join(fen.split()[:4])
        if board in seen:
            continue
        seen.add(board)
        cp = sf.score(fen, args.depth)
        if cp is not None and abs(cp) <= 50:
            kept.append(board)
    sf.send("quit")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(kept) + "\n")
    print(f"{len(kept)} openings from {seed} candidates -> {args.out}")


if __name__ == "__main__":
    main()
