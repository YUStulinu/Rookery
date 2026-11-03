import struct

import numpy as np

from rookery_train.fetch_evals import best
from rookery_train.tune_classical import batches, read


def test_deepest_evaluation_wins_and_mates_are_dropped():
    rec = {"evals": [{"depth": 20, "pvs": [{"cp": 30}]}, {"depth": 36, "pvs": [{"cp": -12}, {"cp": -40}]}]}
    assert best(rec) == (36, -12)
    assert best({"evals": [{"depth": 40, "pvs": [{"mate": 3}]}]}) is None
    assert best({"evals": []}) is None


def test_feature_file_round_trip(tmp_path):
    # Two positions in the engine's format: phase, side, count, score, then (index, count) pairs.
    def record(phase, stm, score, feats):
        out = struct.pack("<BBHi", phase, stm, len(feats), score)
        for f, c in feats:
            out += struct.pack("<Hb", f, c)
        return out

    path = tmp_path / "f.bin"
    path.write_bytes(record(24, 0, 35, [(3, 1), (400, -2)]) + record(5, 1, -120, [(70, 1)]))
    data = read(path)
    assert data["phase"].tolist() == [24, 5]
    assert data["stm"].tolist() == [0, 1]
    assert data["score"].tolist() == [35, -120]
    assert data["feats"].tolist() == [3, 400, 70]
    assert data["counts"].tolist() == [1, -2, 1]
    row_of, feat, cnt, phase, sign, score = next(batches(data, np.array([1, 0]), 10, "cpu"))
    assert row_of.tolist() == [0, 1, 1]
    assert feat.tolist() == [70, 3, 400]
    assert sign.tolist() == [-1.0, 1.0]
