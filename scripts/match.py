"""Plays engine matches with cutechess-cli and records the results.

    python scripts/match.py nnue-vs-classical [--games 400]
    python scripts/match.py stockfish --levels 1500 1800 2100 2400 [--games 100]

Every opening of results/openings.epd is played twice, once with each colour, at 10 s + 0.1 s per
game. Games are adjudicated only when both engines agree the result is clear: a draw after move 40
when both report |score| <= 10 cp for 8 moves, a resignation when both report >= 800 cp for 3.

Stockfish's UCI_Elo is converted to a skill level that, according to its source, "covers CCRL
Blitz Elo from 1320 to 3190, approximately"; Rookery's rating estimate inherits that calibration.

Writes results/<match>.json and the games to results/games/<match>.pgn.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = ".exe" if sys.platform == "win32" else ""
CUTECHESS = next((ROOT / "third_party" / "cutechess").rglob(f"cutechess-cli{EXE}"), None)
STOCKFISH = next((ROOT / "third_party" / "stockfish").rglob(f"stockfish*{EXE}"), None)
ROOKERY = ROOT / "engine" / "build-native" / f"rookery{EXE}"


def run_match(name: str, first: list[str], second: list[str], games: int, concurrency: int, tc: str) -> dict:
    pgn = ROOT / "results" / "games" / f"{name}.pgn"
    pgn.parent.mkdir(parents=True, exist_ok=True)
    pgn.unlink(missing_ok=True)
    cmd = [str(CUTECHESS), "-engine", *first, "-engine", *second,
           "-each", "proto=uci", f"tc={tc}", "option.Hash=64",
           "-openings", f"file={ROOT / 'results' / 'openings.epd'}", "format=epd", "order=random", "policy=round",
           "-repeat", "-games", "2", "-rounds", str(games // 2), "-concurrency", str(concurrency),
           "-draw", "movenumber=40", "movecount=8", "score=10",
           "-resign", "movecount=3", "score=800", "twosided=true",
           "-srand", "21", "-recover", "-pgnout", str(pgn)]
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    score = re.findall(r"Score of .*?: (\d+) - (\d+) - (\d+)", out)
    elo = re.findall(r"Elo difference: (-?[\d.inf]+) \+/- ([\d.nan]+)", out)
    if not score:
        raise RuntimeError(out[-2000:])
    wins, losses, draws = map(int, score[-1])
    n = wins + losses + draws
    points = (wins + draws / 2) / n
    return {"games": n, "wins": wins, "losses": losses, "draws": draws, "score": round(points, 3),
            "elo_difference": float(elo[-1][0]) if elo else None,
            "elo_error_95": float(elo[-1][1]) if elo and elo[-1][1] != "nan" else None,
            "time_control": tc, "minutes": round((time.time() - t0) / 60, 1), "pgn": str(pgn.relative_to(ROOT))}


def rookery(name: str, nnue: bool) -> list[str]:
    return [f"name={name}", f"cmd={ROOKERY}", f"dir={ROOKERY.parent}", f"option.UseNNUE={'true' if nnue else 'false'}"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("match", choices=["nnue-vs-classical", "stockfish"])
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--levels", type=int, nargs="*", default=[1500, 1800, 2100, 2400])
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--tc", default="10+0.1")
    args = ap.parse_args()
    results = {}
    if args.match == "nnue-vs-classical":
        results = run_match("nnue-vs-classical", rookery("Rookery NNUE", True), rookery("Rookery classical", False),
                            args.games, args.concurrency, args.tc)
        print(json.dumps(results, indent=1), flush=True)
    else:
        for level in args.levels:
            sf = [f"name=Stockfish {level}", f"cmd={STOCKFISH}", f"dir={STOCKFISH.parent}", "option.Threads=1",
                  "option.UCI_LimitStrength=true", f"option.UCI_Elo={level}"]
            r = run_match(f"stockfish-{level}", rookery("Rookery NNUE", True), sf, args.games, args.concurrency, args.tc)
            # Rookery's rating implied by this match alone.
            if r["elo_difference"] is not None and math.isfinite(r["elo_difference"]):
                r["implied_rating"] = round(level + r["elo_difference"])
            results[f"UCI_Elo {level}"] = r
            print(level, json.dumps(r), flush=True)
    name = "nnue-vs-classical" if args.match == "nnue-vs-classical" else "stockfish"
    (ROOT / "results" / f"{name}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
