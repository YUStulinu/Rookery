"""Trains the network on Lichess positions and exports it for the engine.

The positions come from the engine (`rookery positions`), already filtered to quiet ones, with the
Stockfish score from the side to move's view. The loss compares win probabilities,
sigmoid(score / 400), as the classical tuner does.

    python -m rookery_train.train_nnue data/positions.bin [--hidden 256 --epochs 12]

writes nets/rookery.nnue, nets/check.json (positions with the scores the engine must give) and
results/nnue-training.json.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from rookery_train.nnue import INPUTS, SCALE, Net, quantize, quantized_eval, save

ROOT = Path(__file__).resolve().parents[2]
RECORD = np.dtype([("stm", "u1"), ("n", "u1"), ("score", "<i2"), ("slots", "<u2", 32)])

CHECK_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 b - - 0 1",
    "8/8/4k3/3p4/3P4/4K3/8/8 w - - 0 1",
    "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R2QKB1R w KQ - 1 8",
]


def features(records: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Feature indices (batch, 32) from the side to move's and the other side's view."""
    slots = records["slots"].astype(np.int64)
    empty = slots == 0xFFFF
    piece, sq = slots // 64, slots % 64
    color = piece // 6
    kind = piece % 6
    white = np.where(color == 0, 0, 384) + kind * 64 + sq
    black = np.where(color == 1, 0, 384) + kind * 64 + (sq ^ 56)
    white[empty] = INPUTS
    black[empty] = INPUTS
    stm_black = (records["stm"] == 1)[:, None]
    return np.where(stm_black, black, white), np.where(stm_black, white, black)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("positions", type=Path)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, default=ROOT / "nets" / "rookery.nnue")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(21)
    data = np.fromfile(args.positions, dtype=RECORD)
    n = len(data)
    rng = np.random.default_rng(21)
    order = rng.permutation(n)
    valid_idx, train_idx = order[: n // 50], order[n // 50:]
    print(f"{n:,} positions: {len(train_idx):,} for training, {len(valid_idx):,} for validation", flush=True)

    net = Net(args.hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    steps = args.epochs * math.ceil(len(train_idx) / args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=args.lr / 30)

    def run(idx: np.ndarray, train: bool) -> float:
        total, count = 0.0, 0
        for b in range(0, len(idx), args.batch):
            rec = data[np.sort(idx[b:b + args.batch])]
            us, them = features(rec)
            us = torch.from_numpy(us).to(device)
            them = torch.from_numpy(them).to(device)
            target = torch.sigmoid(torch.from_numpy(rec["score"].astype(np.float32)).to(device) / SCALE)
            with torch.set_grad_enabled(train):
                pred = torch.sigmoid(net(us, them))
                loss = torch.mean((pred - target) ** 2)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                net.clip()
            total += loss.item() * len(rec)
            count += len(rec)
        return total / count

    t0 = time.time()
    history = []
    for epoch in range(args.epochs):
        rng.shuffle(train_idx)
        net.train()
        tr = run(train_idx, True)
        net.eval()
        va = run(valid_idx, False)
        history.append({"epoch": epoch + 1, "train": round(tr, 6), "valid": round(va, 6)})
        print(f"epoch {epoch + 1}: train {tr:.5f} valid {va:.5f} ({time.time() - t0:.0f}s)", flush=True)

    q = quantize(net)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save(q, args.out)
    check = [{"fen": fen, "eval": quantized_eval(q, fen)} for fen in CHECK_FENS]
    (args.out.parent / "check.json").write_text(json.dumps(check, indent=2))
    out = {"positions": n, "validation_positions": len(valid_idx), "hidden": args.hidden, "epochs": history,
           "seconds": round(time.time() - t0), "file": str(args.out.relative_to(ROOT)),
           "file_bytes": args.out.stat().st_size, "check": check}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "nnue-training.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(check, indent=1))


if __name__ == "__main__":
    main()
