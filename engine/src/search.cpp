#include "search.h"

#include <algorithm>
#include <cmath>

#include "attacks.h"
#include "eval.h"
#include "movegen.h"

namespace rookery {

namespace {

constexpr int PIECE_VALUE[7] = {100, 320, 330, 500, 900, 20000, 0};

int lmr_table[64][64];

struct LmrInit {
    LmrInit() {
        for (int d = 1; d < 64; ++d)
            for (int m = 1; m < 64; ++m) lmr_table[d][m] = int(0.75 + std::log(d) * std::log(m) / 2.25);
    }
} lmr_init;

PieceType captured_type(const Position& pos, Move m) {
    if (m.flags() == EN_PASSANT) return PAWN;
    Piece p = pos.piece_on(m.to());
    return p == NO_PIECE ? NO_PIECE_TYPE : type_of(p);
}

}  // namespace

bool see_ge(const Position& pos, Move m, int threshold) {
    if (m.is_castle() || m.is_promotion() || m.flags() == EN_PASSANT) return threshold <= 0;
    Square from = m.from(), to = m.to();
    int swap = PIECE_VALUE[captured_type(pos, m)] - threshold;
    if (swap < 0) return false;
    swap = PIECE_VALUE[type_of(pos.piece_on(from))] - swap;
    if (swap <= 0) return true;

    Bitboard occ = pos.occupied() ^ bb(from) ^ bb(to);
    Color stm = pos.side_to_move();
    Bitboard attackers = pos.attackers_to(to, occ);
    Bitboard diag = pos.pieces(BISHOP) | pos.pieces(QUEEN);
    Bitboard straight = pos.pieces(ROOK) | pos.pieces(QUEEN);
    int result = 1;
    while (true) {
        stm = ~stm;
        attackers &= occ;
        Bitboard mine = attackers & pos.pieces(stm);
        if (!mine) break;
        result ^= 1;
        // The least valuable attacker takes next; sliders behind it join in.
        PieceType t = PAWN;
        Bitboard b = 0;
        for (t = PAWN; t <= KING; t = PieceType(t + 1))
            if ((b = mine & pos.pieces(stm, t))) break;
        if (t == KING) return (attackers & pos.pieces(~stm)) ? result ^ 1 : result;
        swap = PIECE_VALUE[t] - swap;
        if (swap < result) break;
        occ ^= bb(lsb(b));
        if (t == PAWN || t == BISHOP || t == QUEEN) attackers |= attacks::bishop(to, occ) & diag;
        if (t == ROOK || t == QUEEN) attackers |= attacks::rook(to, occ) & straight;
    }
    return result;
}

void Search::set_network(std::shared_ptr<const nnue::Network> net) {
    nnue_ = net ? std::make_unique<nnue::Evaluator>(std::move(net)) : nullptr;
}

void Search::clear() {
    for (auto& k : killers_) k[0] = k[1] = Move();
    for (auto& c : history_)
        for (auto& f : c)
            for (int& h : f) h = 0;
}

Value Search::evaluate(const Position& pos) {
    return nnue_ ? nnue_->evaluate(pos) : eval::classical(pos);
}

bool Search::should_stop() {
    if (stop_) return true;
    if (node_limit_ && nodes_ >= node_limit_) return stop_ = true;
    if ((nodes_ & 2047) == 0 && hard_ms_) {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_).count();
        if (ms >= hard_ms_) stop_ = true;
    }
    return stop_;
}

Move Search::think(Position& pos, const Limits& limits, const std::function<void(const SearchInfo&)>& on_info) {
    start_ = std::chrono::steady_clock::now();
    stop_ = false;
    nodes_ = 0;
    node_limit_ = limits.nodes;
    tt_.new_search();
    if (nnue_) nnue_->reset();

    // Time: a soft limit checked between iterations and a hard one checked during the search.
    soft_ms_ = hard_ms_ = 0;
    Color us = pos.side_to_move();
    if (limits.movetime) {
        soft_ms_ = hard_ms_ = std::max(1, limits.movetime - 10);
    } else if (!limits.infinite && limits.time[us]) {
        int left = limits.time[us];
        int moves = limits.movestogo ? std::min(limits.movestogo, 40) : 30;
        std::int64_t base = left / moves + limits.inc[us] * 3 / 4;
        soft_ms_ = std::max<std::int64_t>(1, std::min<std::int64_t>(base, left / 2) - 10);
        hard_ms_ = std::max<std::int64_t>(1, std::min<std::int64_t>(base * 3, left * 3 / 5) - 10);
    }

    MoveList root_moves;
    generate(pos, root_moves);
    Move fallback;
    for (Move m : root_moves)
        if (pos.legal(m)) {
            fallback = m;
            break;
        }
    if (fallback.is_null()) return Move();

    Move best = fallback;
    Value score = 0;
    for (int depth = 1; depth <= limits.depth; ++depth) {
        seldepth_ = 0;
        Value alpha = -VALUE_INF, beta = VALUE_INF, delta = 25;
        if (depth >= 5) {
            alpha = std::max(score - delta, -VALUE_INF);
            beta = std::min(score + delta, VALUE_INF);
        }
        Value v;
        // Aspiration windows: search a narrow window around the last score, widening on failure.
        while (true) {
            v = negamax(pos, alpha, beta, depth, 0, false);
            if (stop_) break;
            if (v <= alpha) {
                beta = (alpha + beta) / 2;
                alpha = std::max(v - delta, -VALUE_INF);
            } else if (v >= beta) {
                beta = std::min(v + delta, VALUE_INF);
            } else {
                break;
            }
            delta += delta / 2;
        }
        if (stop_ && depth > 1) break;  // an interrupted iteration is not trusted
        score = v;
        if (pv_len_[0] > 0) best = pv_[0][0];
        last_score_ = score;
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_).count();
        if (on_info) {
            SearchInfo info{depth, seldepth_, score, nodes_, ms, tt_.hashfull(), ""};
            for (int i = 0; i < pv_len_[0]; ++i) info.pv += (i ? " " : "") + move_to_uci(pv_[0][i]);
            on_info(info);
        }
        if (soft_ms_ && ms >= soft_ms_) break;
        // Pruning can hide a shorter mate, so a mate found is no reason to stop early; but a mate
        // in one is the shortest there is.
        if (score == VALUE_MATE - 1) break;
    }
    return best;
}

void Search::update_quiet_history(Color c, Move best, Move* tried, int n_tried, int depth, int ply) {
    int bonus = std::min(16 * depth * depth, 1200);
    auto apply = [&](Move m, int b) {
        int& h = history_[c][m.from()][m.to()];
        h += b - h * std::abs(b) / 16384;  // the gravity term keeps values bounded
    };
    apply(best, bonus);
    for (int i = 0; i < n_tried; ++i)
        if (!(tried[i] == best)) apply(tried[i], -bonus);
    if (!(killers_[ply][0] == best)) {
        killers_[ply][1] = killers_[ply][0];
        killers_[ply][0] = best;
    }
}

Value Search::quiescence(Position& pos, Value alpha, Value beta, int ply) {
    ++nodes_;
    pv_len_[ply] = 0;
    if (should_stop()) return 0;
    seldepth_ = std::max(seldepth_, ply);
    if (ply >= MAX_PLY - 1) return evaluate(pos);
    if (pos.is_draw(ply)) return 0;

    bool found;
    TTEntry* e = tt_.probe(pos.key(), found);
    if (found) {
        Value v = score_from_tt(e->score, ply);
        if (e->bound == BOUND_EXACT || (e->bound == BOUND_LOWER && v >= beta) || (e->bound == BOUND_UPPER && v <= alpha))
            return v;
    }

    bool in_check = pos.in_check();
    Value stand = in_check ? -VALUE_MATE + ply : evaluate(pos);
    if (!in_check) {
        if (stand >= beta) return stand;
        alpha = std::max(alpha, stand);
    }

    MoveList list;
    generate(pos, list, in_check ? GenType::ALL : GenType::NOISY);
    int scores[256];
    for (int i = 0; i < list.size; ++i) {
        Move m = list.moves[i];
        PieceType cap = captured_type(pos, m);
        scores[i] = (cap == NO_PIECE_TYPE ? 0 : PIECE_VALUE[cap] * 10) - PIECE_VALUE[type_of(pos.piece_on(m.from()))] / 10 +
                    (m.is_promotion() ? 8000 : 0);
    }
    Value best = stand;
    int legal = 0;
    for (int i = 0; i < list.size; ++i) {
        int j = int(std::max_element(scores + i, scores + list.size) - scores);
        std::swap(list.moves[i], list.moves[j]);
        std::swap(scores[i], scores[j]);
        Move m = list.moves[i];
        if (!pos.legal(m)) continue;
        ++legal;
        // Captures that lose material cannot raise the score; skip them (unless escaping check).
        if (!in_check && !see_ge(pos, m, 0)) continue;
        pos.make(m);
        Value v = -quiescence(pos, -beta, -alpha, ply + 1);
        pos.unmake();
        if (stop_) return 0;
        if (v > best) {
            best = v;
            if (v > alpha) {
                alpha = v;
                pv_[ply][0] = m;
                for (int k = 0; k < pv_len_[ply + 1]; ++k) pv_[ply][k + 1] = pv_[ply + 1][k];
                pv_len_[ply] = pv_len_[ply + 1] + 1;
                if (v >= beta) break;
            }
        }
    }
    if (in_check && legal == 0) return -VALUE_MATE + ply;
    return best;
}

Value Search::negamax(Position& pos, Value alpha, Value beta, int depth, int ply, bool cut_node) {
    pv_len_[ply] = 0;
    bool in_check = pos.in_check();
    if (in_check) ++depth;  // check extension
    if (depth <= 0) return quiescence(pos, alpha, beta, ply);
    ++nodes_;
    if (should_stop()) return 0;
    seldepth_ = std::max(seldepth_, ply);
    bool root = ply == 0;
    bool pv_node = beta - alpha > 1;

    if (!root) {
        if (pos.is_draw(ply)) return 0;
        if (ply >= MAX_PLY - 1) return evaluate(pos);
        // Mate distance pruning: no line can beat a mate already found nearer the root.
        alpha = std::max(alpha, -VALUE_MATE + ply);
        beta = std::min(beta, VALUE_MATE - ply - 1);
        if (alpha >= beta) return alpha;
    }

    bool found;
    TTEntry* e = tt_.probe(pos.key(), found);
    Move tt_move = found ? e->move : Move();
    if (found && !pv_node && e->depth >= depth) {
        Value v = score_from_tt(e->score, ply);
        if (e->bound == BOUND_EXACT || (e->bound == BOUND_LOWER && v >= beta) || (e->bound == BOUND_UPPER && v <= alpha))
            return v;
    }

    Value static_eval = in_check ? VALUE_NONE : (found ? e->eval : evaluate(pos));
    static_eval_[ply] = static_eval;
    bool improving = !in_check && ply >= 2 && static_eval_[ply - 2] != VALUE_NONE && static_eval > static_eval_[ply - 2];

    if (!pv_node && !in_check) {
        // Reverse futility: far enough above beta that a shallow search will not fall below it.
        if (depth <= 6 && static_eval - (80 - 20 * improving) * depth >= beta && static_eval < VALUE_MATE_IN_MAX)
            return static_eval;
        // Null move: if passing still beats beta, a real move will too (not in pawn endings,
        // where zugzwang is common).
        if (depth >= 3 && static_eval >= beta && pos.has_non_pawn_material(pos.side_to_move()) &&
            !pos.state().move.is_null()) {
            int r = 3 + depth / 4 + std::min((static_eval - beta) / 200, 3);
            pos.make_null();
            Value v = -negamax(pos, -beta, -beta + 1, depth - r, ply + 1, !cut_node);
            pos.unmake_null();
            if (stop_) return 0;
            if (v >= beta) return v >= VALUE_MATE_IN_MAX ? beta : v;
        }
    }
    // Without a move from the table, a shallower search ordering is worth less than its cost.
    if (tt_move.is_null() && depth >= 4 && (pv_node || cut_node)) --depth;

    MoveList list;
    generate(pos, list);
    int scores[256];
    Color us = pos.side_to_move();
    for (int i = 0; i < list.size; ++i) {
        Move m = list.moves[i];
        if (m == tt_move) {
            scores[i] = 1 << 30;
        } else if (m.is_capture() || m.is_promotion()) {
            PieceType cap = captured_type(pos, m);
            int mvv = (cap == NO_PIECE_TYPE ? 0 : PIECE_VALUE[cap] * 10) - PIECE_VALUE[type_of(pos.piece_on(m.from()))] / 10;
            scores[i] = (see_ge(pos, m, 0) ? (1 << 28) : -(1 << 20)) + mvv + (m.is_promotion() ? 8000 : 0);
        } else if (m == killers_[ply][0]) {
            scores[i] = 1 << 27;
        } else if (m == killers_[ply][1]) {
            scores[i] = (1 << 27) - 1;
        } else {
            scores[i] = history_[us][m.from()][m.to()];
        }
    }

    Value best = -VALUE_INF;
    Move best_move;
    Move quiets[256];
    int n_quiets = 0;
    int legal = 0;
    Value old_alpha = alpha;
    for (int i = 0; i < list.size; ++i) {
        int j = int(std::max_element(scores + i, scores + list.size) - scores);
        std::swap(list.moves[i], list.moves[j]);
        std::swap(scores[i], scores[j]);
        Move m = list.moves[i];
        if (!pos.legal(m)) continue;
        ++legal;
        bool quiet = !m.is_capture() && !m.is_promotion();

        // Quiet moves that late move pruning or futility would skip; checking moves are kept.
        bool prunable = false;
        if (!root && best > -VALUE_MATE_IN_MAX && !in_check) {
            // Late move pruning: deep in a quiet move list at low depth, the rest rarely matter.
            if (quiet && depth <= 5 && legal > 3 + depth * depth / (2 - improving)) prunable = true;
            // Futility: a quiet move cannot lift a static eval this far below alpha.
            if (quiet && depth <= 6 && static_eval + 100 + 90 * depth <= alpha) prunable = true;
            // Captures that lose much material at low depth.
            if (!quiet && depth <= 6 && !see_ge(pos, m, -90 * depth)) continue;
        }

        pos.make(m);
        bool gives_check = pos.in_check();
        if (prunable && !gives_check) {
            pos.unmake();
            continue;
        }
        Value v;
        if (legal == 1) {
            v = -negamax(pos, -beta, -alpha, depth - 1, ply + 1, false);
        } else {
            // Late move reductions: search later quiet moves shallower, and again if they surprise.
            int r = 0;
            if (depth >= 3 && quiet && !gives_check) {
                r = lmr_table[std::min(depth, 63)][std::min(legal, 63)];
                r += !pv_node + cut_node - improving;
                r -= history_[us][m.from()][m.to()] / 8000;
                r = std::clamp(r, 0, depth - 2);
            }
            v = -negamax(pos, -alpha - 1, -alpha, depth - 1 - r, ply + 1, true);
            if (v > alpha && r > 0) v = -negamax(pos, -alpha - 1, -alpha, depth - 1, ply + 1, !cut_node);
            if (v > alpha && v < beta) v = -negamax(pos, -beta, -alpha, depth - 1, ply + 1, false);
        }
        pos.unmake();
        if (stop_) return 0;

        if (v > best) {
            best = v;
            if (v > alpha) {
                alpha = v;
                best_move = m;
                pv_[ply][0] = m;
                for (int k = 0; k < pv_len_[ply + 1]; ++k) pv_[ply][k + 1] = pv_[ply + 1][k];
                pv_len_[ply] = pv_len_[ply + 1] + 1;
                if (v >= beta) {
                    if (quiet) update_quiet_history(us, m, quiets, n_quiets, depth, ply);
                    break;
                }
            }
        }
        if (quiet && n_quiets < 256) quiets[n_quiets++] = m;
    }

    if (legal == 0) return in_check ? -VALUE_MATE + ply : 0;

    Bound bound = best >= beta ? BOUND_LOWER : alpha > old_alpha ? BOUND_EXACT : BOUND_UPPER;
    tt_.store(e, pos.key(), best_move, score_to_tt(best, ply), static_eval == VALUE_NONE ? 0 : static_eval, depth, bound);
    return best;
}

}  // namespace rookery
