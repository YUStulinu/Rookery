"""The network and its file format, shared by training and export.

Inputs: 768 per perspective, own/their piece type times square, mirrored for Black, exactly as
engine/src/nnue.h. The engine's integer arithmetic is reproduced in `quantized_eval`, so the
exported file can be checked position by position against the engine.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch

INPUTS = 768
QA, QB, SCALE = 255, 64, 400
PIECES = "PNBRQKpnbrqk"


def feature(perspective: int, piece: int, square: int) -> int:
    """perspective 0 = White, 1 = Black; piece 0..11 as in PIECES; square a1 = 0."""
    own = 0 if piece // 6 == perspective else 1
    sq = square if perspective == 0 else square ^ 56
    return own * 384 + (piece % 6) * 64 + sq


def parse_fen(fen: str) -> tuple[list[tuple[int, int]], int]:
    """(piece, square) pairs and the side to move (0 White, 1 Black)."""
    placement, side = fen.split()[:2]
    pieces = []
    rank, file = 7, 0
    for c in placement:
        if c == "/":
            rank, file = rank - 1, 0
        elif c.isdigit():
            file += int(c)
        else:
            pieces.append((PIECES.index(c), rank * 8 + file))
            file += 1
    return pieces, 0 if side == "w" else 1


class Net(torch.nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.hidden = hidden
        # Index INPUTS is padding: positions have fewer than 32 pieces.
        self.ft = torch.nn.EmbeddingBag(INPUTS + 1, hidden, mode="sum", padding_idx=INPUTS)
        self.ft_bias = torch.nn.Parameter(torch.zeros(hidden))
        self.out = torch.nn.Linear(2 * hidden, 1)
        torch.nn.init.normal_(self.ft.weight, std=0.1)
        with torch.no_grad():
            self.ft.weight[INPUTS].zero_()

    def forward(self, us: torch.Tensor, them: torch.Tensor) -> torch.Tensor:
        """us, them: (batch, 32) feature indices for the side to move and the other side.
        Returns the evaluation in units of SCALE centipawns."""
        a = torch.clamp(self.ft(us) + self.ft_bias, 0, 1) ** 2
        b = torch.clamp(self.ft(them) + self.ft_bias, 0, 1) ** 2
        return self.out(torch.cat([a, b], dim=1)).squeeze(1)

    def clip(self) -> None:
        """Keeps weights inside what int16 can hold after quantisation."""
        with torch.no_grad():
            self.ft.weight.clamp_(-1.98, 1.98)
            self.out.weight.clamp_(-1.98, 1.98)


def quantize(net: Net) -> dict[str, np.ndarray]:
    ft = net.ft.weight.detach().cpu().numpy()[:INPUTS]
    return {
        "ft_weights": np.round(ft * QA).astype(np.int16),
        "ft_bias": np.round(net.ft_bias.detach().cpu().numpy() * QA).astype(np.int16),
        "out_weights": np.round(net.out.weight.detach().cpu().numpy().ravel() * QB).astype(np.int16),
        "out_bias": np.int32(round(float(net.out.bias.item()) * QA * QB)),
    }


def save(q: dict[str, np.ndarray], path: Path) -> None:
    hidden = q["ft_bias"].shape[0]
    with open(path, "wb") as f:
        f.write(b"RKNN")
        f.write(struct.pack("<ii", 1, hidden))
        f.write(q["ft_weights"].astype("<i2").tobytes())
        f.write(q["ft_bias"].astype("<i2").tobytes())
        f.write(q["out_weights"].astype("<i2").tobytes())
        f.write(struct.pack("<i", int(q["out_bias"])))


def _trunc_div(a: int, b: int) -> int:
    """Integer division rounding toward zero, as in C++."""
    return a // b if a >= 0 else -((-a) // b)


def quantized_eval(q: dict[str, np.ndarray], fen: str) -> int:
    """The engine's evaluation of a position, computed the way engine/src/nnue.cpp does."""
    pieces, stm = parse_fen(fen)
    acc = []
    for persp in (0, 1):
        v = q["ft_bias"].astype(np.int64).copy()
        for piece, sq in pieces:
            v += q["ft_weights"][feature(persp, piece, sq)].astype(np.int64)
        acc.append(v.astype(np.int16).astype(np.int64))  # the engine keeps int16
    hidden = len(acc[0])
    total = 0
    for half, persp in enumerate((stm, 1 - stm)):
        c = np.clip(acc[persp], 0, QA)
        w = q["out_weights"][half * hidden:(half + 1) * hidden].astype(np.int64)
        total += int(np.sum(c * c * w))
    total = _trunc_div(total, QA) + int(q["out_bias"])
    cp = _trunc_div(total * SCALE, QA * QB)
    return max(-31743, min(31743, cp))
