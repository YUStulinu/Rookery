import numpy as np
import torch

from rookery_train import nnue
from rookery_train.train_nnue import RECORD, features


def test_feature_mirrors_for_black():
    # A white pawn on e2 for White is a "their" pawn on e7 for Black.
    assert nnue.feature(0, 0, 12) == 0 * 384 + 0 * 64 + 12
    assert nnue.feature(1, 0, 12) == 1 * 384 + 0 * 64 + (12 ^ 56)
    # A black king on e8 for Black is an "own" king on e1.
    assert nnue.feature(1, 11, 60) == 0 * 384 + 5 * 64 + 4


def test_parse_fen():
    pieces, stm = nnue.parse_fen("4k3/8/8/8/8/8/4P3/4K3 b - - 0 1")
    assert stm == 1
    assert sorted(pieces) == [(0, 12), (5, 4), (11, 60)]


def test_batch_features_match_the_scalar_definition():
    # One record the way the engine writes it: Black to move, three pieces.
    rec = np.zeros(1, dtype=RECORD)
    rec["stm"] = 1
    rec["n"] = 3
    rec["slots"] = 0xFFFF
    rec["slots"][0, :3] = [0 * 64 + 12, 5 * 64 + 4, 11 * 64 + 60]
    us, them = features(rec)
    expect_us = sorted(nnue.feature(1, p, s) for p, s in [(0, 12), (5, 4), (11, 60)])
    expect_them = sorted(nnue.feature(0, p, s) for p, s in [(0, 12), (5, 4), (11, 60)])
    assert sorted(us[0][:3].tolist()) == expect_us
    assert sorted(them[0][:3].tolist()) == expect_them
    assert (us[0][3:] == nnue.INPUTS).all()


def test_quantized_eval_follows_the_float_network():
    torch.manual_seed(0)
    net = nnue.Net(32)
    with torch.no_grad():
        net.ft.weight.mul_(3)
    q = nnue.quantize(net)
    fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    pieces, stm = nnue.parse_fen(fen)
    pad = lambda xs: torch.tensor([xs + [nnue.INPUTS] * (32 - len(xs))])
    us = pad([nnue.feature(stm, p, s) for p, s in pieces])
    them = pad([nnue.feature(1 - stm, p, s) for p, s in pieces])
    float_cp = net(us, them).item() * nnue.SCALE
    # Quantisation (output weights in steps of 1/64) costs a few percent.
    assert abs(nnue.quantized_eval(q, fen) - float_cp) <= 0.06 * abs(float_cp) + 3


def test_saved_file_layout(tmp_path):
    net = nnue.Net(16)
    q = nnue.quantize(net)
    path = tmp_path / "n.nnue"
    nnue.save(q, path)
    data = path.read_bytes()
    assert data[:4] == b"RKNN"
    assert len(data) == 4 + 8 + 2 * (768 * 16 + 16 + 2 * 16) + 4
