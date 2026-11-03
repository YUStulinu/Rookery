// Evaluation: the classical hand-built terms (with weights tuned on Lichess positions) and the
// neural network (NNUE). Both score from the side to move's point of view, in centipawns.
#pragma once

#include <utility>
#include <vector>

#include "types.h"

namespace rookery {

class Position;

namespace eval {

// The classical evaluation is linear in these features, each with a middlegame and an endgame
// weight blended by game phase:
//   0..383   piece-square: piece type * 64 + square (from the piece owner's side)
//   384..387 mobility of knights, bishops, rooks, queens (safe squares)
//   388      bishop pair
//   389..394 passed pawn on relative rank 2..7
//   395      doubled pawn, 396 isolated pawn
//   397      rook on an open file, 398 rook on a half-open file
//   399      pawn shield in front of the king, 400 enemy attacks next to the king
constexpr int NUM_FEATURES = 401;
constexpr int MAX_PHASE = 24;

// What the tuner needs: the phase, the side to move, and each feature's count (White minus Black).
struct Trace {
    int phase = 0;
    Color stm = WHITE;
    std::vector<std::pair<int, int>> features;
};

Value classical(const Position& pos);
void trace(const Position& pos, Trace& t);

}  // namespace eval
}  // namespace rookery
