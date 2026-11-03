#include "eval.h"

#include <algorithm>

#include "attacks.h"
#include "eval_weights.h"
#include "position.h"

namespace rookery::eval {

namespace {

constexpr int PHASE_WEIGHT[6] = {0, 1, 1, 2, 4, 0};

// Adds tuned weights: the evaluation itself.
struct Score {
    int mg = 0, eg = 0;
    void add(int feature, Color c, int count) {
        int sign = c == WHITE ? count : -count;
        mg += sign * WEIGHTS_MG[feature];
        eg += sign * WEIGHTS_EG[feature];
    }
};

// Records feature counts: what the tuner fits the weights to.
struct Recorder {
    int counts[NUM_FEATURES] = {};
    void add(int feature, Color c, int count) { counts[feature] += c == WHITE ? count : -count; }
};

constexpr Bitboard file_bb(int f) { return FILE_A << f; }

Bitboard adjacent_files(int f) {
    return (f > 0 ? file_bb(f - 1) : 0) | (f < 7 ? file_bb(f + 1) : 0);
}

// Squares in front of a pawn on its own and the adjacent files, up to the last rank.
Bitboard passed_span(Color c, Square s) {
    Bitboard files = file_bb(file_of(s)) | adjacent_files(file_of(s));
    Bitboard ahead = 0;
    if (c == WHITE)
        for (int r = rank_of(s) + 1; r < 8; ++r) ahead |= RANK_1 << (8 * r);
    else
        for (int r = rank_of(s) - 1; r >= 0; --r) ahead |= RANK_1 << (8 * r);
    return files & ahead;
}

template <typename Sink>
void features(const Position& pos, Sink& sink) {
    Bitboard occ = pos.occupied();
    for (Color c : {WHITE, BLACK}) {
        Color them = ~c;
        Bitboard own_pawns = pos.pieces(c, PAWN), their_pawns = pos.pieces(them, PAWN);
        Bitboard their_pawn_attacks = 0;
        for (Bitboard b = their_pawns; b;) their_pawn_attacks |= attacks::pawn[them][pop_lsb(b)];
        Bitboard safe = ~pos.pieces(c) & ~their_pawn_attacks;

        for (PieceType t : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING}) {
            for (Bitboard b = pos.pieces(c, t); b;) {
                Square s = pop_lsb(b);
                Square rel = c == WHITE ? s : flip(s);
                sink.add(t * 64 + rel, c, 1);
                if (t >= KNIGHT && t <= QUEEN) sink.add(384 + t - KNIGHT, c, popcount(attacks::of(t, s, occ) & safe));
                if (t == ROOK) {
                    Bitboard f = file_bb(file_of(s));
                    if (!(f & (own_pawns | their_pawns)))
                        sink.add(397, c, 1);
                    else if (!(f & own_pawns))
                        sink.add(398, c, 1);
                }
            }
        }
        if (popcount(pos.pieces(c, BISHOP)) >= 2) sink.add(388, c, 1);

        for (Bitboard b = own_pawns; b;) {
            Square s = pop_lsb(b);
            int f = file_of(s);
            if (!(passed_span(c, s) & their_pawns)) {
                int rel_rank = c == WHITE ? rank_of(s) : 7 - rank_of(s);
                sink.add(389 + rel_rank - 1, c, 1);
            }
            if (!(adjacent_files(f) & own_pawns)) sink.add(396, c, 1);
        }
        for (int f = 0; f < 8; ++f) {
            int n = popcount(own_pawns & file_bb(f));
            if (n > 1) sink.add(395, c, n - 1);
        }

        Square k = pos.king_square(c);
        Bitboard shield_zone = attacks::king[k] | (c == WHITE ? shift_north(attacks::king[k]) : shift_south(attacks::king[k]));
        Bitboard front = passed_span(c, k) & shield_zone;
        sink.add(399, c, popcount(front & own_pawns));
        int pressure = 0;
        for (PieceType t : {KNIGHT, BISHOP, ROOK, QUEEN}) {
            for (Bitboard b = pos.pieces(them, t); b;) {
                Square s = pop_lsb(b);
                pressure += popcount(attacks::of(t, s, occ) & attacks::king[k]);
            }
        }
        sink.add(400, c, pressure);
    }
}

int phase(const Position& pos) {
    int p = 0;
    for (PieceType t : {KNIGHT, BISHOP, ROOK, QUEEN}) p += PHASE_WEIGHT[t] * popcount(pos.pieces(t));
    return std::min(p, MAX_PHASE);
}

}  // namespace

Value classical(const Position& pos) {
    Score s;
    features(pos, s);
    int ph = phase(pos);
    int white = (s.mg * ph + s.eg * (MAX_PHASE - ph)) / MAX_PHASE;
    int v = (pos.side_to_move() == WHITE ? white : -white) + TEMPO;
    return std::clamp(v, -VALUE_MATE_IN_MAX + 1, VALUE_MATE_IN_MAX - 1);
}

void trace(const Position& pos, Trace& t) {
    Recorder r;
    features(pos, r);
    t.phase = phase(pos);
    t.stm = pos.side_to_move();
    t.features.clear();
    for (int i = 0; i < NUM_FEATURES; ++i)
        if (r.counts[i]) t.features.emplace_back(i, r.counts[i]);
}

}  // namespace rookery::eval
