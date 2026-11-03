"""Tunes the classical evaluation's weights on Lichess positions (Texel-style tuning).

The classical evaluation is linear in its features (engine/src/eval.h), with a middlegame and an
endgame weight per feature blended by game phase. The engine writes each position's feature
counts (`rookery features`), so the tuner fits exactly what the engine computes. The loss compares
win probabilities, sigmoid(score / 400), as the network trainer does.

    python -m rookery_train.tune_classical data/features.bin [--epochs 8]

writes engine/src/eval_weights.h and results/classical-tuning.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
NUM_FEATURES = 401
MAX_PHASE = 24
MATERIAL_MG = [82, 337, 365, 477, 1025, 0]
MATERIAL_EG = [94, 281, 297, 512, 936, 0]


def read(path: Path) -> dict[str, np.ndarray]:
    """Parses the engine's feature file into CSR arrays: position i's features are
    feats[offsets[i]:offsets[i + 1]] with counts[...]."""
    raw = np.fromfile(path, dtype=np.uint8)
    # Records have variable length, so finding where each starts takes one sequential pass.
    lengths = []
    i, n = 0, len(raw)
    while i < n:
        count = int(raw[i + 2]) | (int(raw[i + 3]) << 8)
        lengths.append(count)
        i += 8 + 3 * count
    lengths = np.array(lengths, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(8 + 3 * lengths)[:-1]])
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    # Byte position of every feature entry.
    pos = np.repeat(starts + 8, lengths) + 3 * (np.arange(offsets[-1]) - np.repeat(offsets[:-1], lengths))
    feats = raw[pos].astype(np.int64) | (raw[pos + 1].astype(np.int64) << 8)
    counts = raw[pos + 2].view(np.int8).astype(np.float32)
    scores = raw[(starts[:, None] + np.arange(4, 8)).ravel()].reshape(-1, 4).copy().view(np.int32).ravel()
    return {"phase": raw[starts].astype(np.float32), "stm": raw[starts + 1].astype(np.int8),
            "score": scores.astype(np.float32), "offsets": offsets, "lengths": lengths, "feats": feats, "counts": counts}


def batches(data: dict, rows: np.ndarray, size: int, device: str):
    for b in range(0, len(rows), size):
        idx = rows[b:b + size]
        lengths = data["lengths"][idx]
        row_of = np.repeat(np.arange(len(idx)), lengths)
        within = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
        flat = np.repeat(data["offsets"][idx], lengths) + within
        yield (torch.from_numpy(row_of).to(device),
               torch.from_numpy(data["feats"][flat]).to(device),
               torch.from_numpy(data["counts"][flat]).to(device),
               torch.from_numpy(data["phase"][idx]).to(device),
               torch.from_numpy(np.where(data["stm"][idx] == 0, 1.0, -1.0).astype(np.float32)).to(device),
               torch.from_numpy(data["score"][idx]).to(device))


def predict(mg, eg, tempo, row_of, feat, cnt, phase, sign, n):
    """White's score in centipawns, as the engine computes it."""
    contrib = cnt * (mg[feat] * phase[row_of] + eg[feat] * (MAX_PHASE - phase[row_of])) / MAX_PHASE
    white = torch.zeros(n, device=mg.device).index_add_(0, row_of, contrib)
    return white + tempo * sign


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("features", type=Path)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--lr", type=float, default=2.0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    data = read(args.features)
    n = len(data["score"])
    print(f"{n:,} positions read in {time.time() - t0:.0f}s", flush=True)

    rng = np.random.default_rng(21)
    order = rng.permutation(n)
    valid, train = order[: n // 20], order[n // 20:]

    init_mg = [MATERIAL_MG[i // 64] for i in range(384)] + [0] * (NUM_FEATURES - 384)
    init_eg = [MATERIAL_EG[i // 64] for i in range(384)] + [0] * (NUM_FEATURES - 384)
    mg = torch.tensor(init_mg, dtype=torch.float32, device=device, requires_grad=True)
    eg = torch.tensor(init_eg, dtype=torch.float32, device=device, requires_grad=True)
    tempo = torch.tensor(10.0, device=device, requires_grad=True)
    opt = torch.optim.Adam([mg, eg, tempo], lr=args.lr)

    def loss_on(rows, train_step: bool) -> float:
        total, count = 0.0, 0
        for row_of, feat, cnt, phase, sign, score in batches(data, rows, args.batch, device):
            pred = predict(mg, eg, tempo, row_of, feat, cnt, phase, sign, len(score))
            loss = torch.mean((torch.sigmoid(pred / 400) - torch.sigmoid(score / 400)) ** 2)
            if train_step:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total += loss.item() * len(score)
            count += len(score)
        return total / count

    with torch.no_grad():
        start_loss = loss_on(valid, False)
    print(f"validation loss with material only: {start_loss:.5f}", flush=True)
    history = []
    for epoch in range(args.epochs):
        rng.shuffle(train)
        tr = loss_on(train, True)
        with torch.no_grad():
            va = loss_on(valid, False)
        history.append({"epoch": epoch + 1, "train": round(tr, 6), "valid": round(va, 6)})
        print(f"epoch {epoch + 1}: train {tr:.5f} valid {va:.5f} ({time.time() - t0:.0f}s)", flush=True)

    w_mg = [int(round(x)) for x in mg.detach().cpu().tolist()]
    w_eg = [int(round(x)) for x in eg.detach().cpu().tolist()]

    def arr(v):
        return ",\n    ".join(", ".join(f"{x}" for x in v[i:i + 16]) for i in range(0, len(v), 16))

    (ROOT / "engine" / "src" / "eval_weights.h").write_text(
        "// The classical evaluation's weights, tuned on Lichess positions by\n"
        "// trainer/rookery_train/tune_classical.py. Do not edit by hand.\n#pragma once\n\n"
        "namespace rookery::eval {\n\n"
        f"constexpr int TEMPO = {int(round(tempo.item()))};\n"
        f"constexpr int WEIGHTS_MG[{NUM_FEATURES}] = {{\n    {arr(w_mg)}}};\n"
        f"constexpr int WEIGHTS_EG[{NUM_FEATURES}] = {{\n    {arr(w_eg)}}};\n\n"
        "}  // namespace rookery::eval\n", encoding="utf-8")
    names = ["mobility knight", "mobility bishop", "mobility rook", "mobility queen", "bishop pair"] + \
            [f"passed pawn rank {r}" for r in range(2, 8)] + \
            ["doubled pawn", "isolated pawn", "rook open file", "rook half-open file", "king pawn shield", "king attacks"]
    out = {"positions": n, "validation_positions": len(valid), "loss_material_only": round(start_loss, 6),
           "epochs": history, "tempo": int(round(tempo.item())),
           "terms (mg, eg)": {name: [w_mg[384 + i], w_eg[384 + i]] for i, name in enumerate(names)},
           "average piece values (mg, eg)": {p: [round(float(np.mean(w_mg[t * 64:(t + 1) * 64]))),
                                                round(float(np.mean(w_eg[t * 64:(t + 1) * 64])))]
                                            for t, p in enumerate(["pawn", "knight", "bishop", "rook", "queen"])}}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "classical-tuning.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["terms (mg, eg)"]), json.dumps(out["average piece values (mg, eg)"]))


if __name__ == "__main__":
    main()
