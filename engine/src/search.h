// The search: iterative deepening over a principal-variation alpha-beta search with a
// quiescence search, transposition table, null-move pruning, late-move reductions and pruning,
// and killer and history move ordering.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "nnue.h"
#include "position.h"
#include "tt.h"

namespace rookery {

constexpr int MAX_PLY = 128;

struct Limits {
    int depth = MAX_PLY - 1;
    std::uint64_t nodes = 0;  // 0: no limit
    int movetime = 0;         // milliseconds; 0: none
    int time[2] = {0, 0};     // remaining clock per colour, milliseconds
    int inc[2] = {0, 0};
    int movestogo = 0;
    bool infinite = false;
};

struct SearchInfo {
    int depth = 0;
    int seldepth = 0;
    Value score = 0;
    std::uint64_t nodes = 0;
    std::int64_t millis = 0;
    int hashfull = 0;
    std::string pv;
};

// Static exchange evaluation: whether a capture sequence on the move's square gains at least
// `threshold` for the side moving.
bool see_ge(const Position& pos, Move m, int threshold);

class Search {
public:
    Search(TranspositionTable& tt) : tt_(tt) {}

    // With a network, positions are evaluated by it; without, by the classical evaluation.
    void set_network(std::shared_ptr<const nnue::Network> net);
    bool uses_network() const { return nnue_ != nullptr; }

    // Searches until a limit is reached or stop() is called; returns the best move.
    Move think(Position& pos, const Limits& limits, const std::function<void(const SearchInfo&)>& on_info = {});
    void stop() { stop_ = true; }
    void clear();  // forget history and killers (a new game)
    std::uint64_t nodes() const { return nodes_; }
    Value last_score() const { return last_score_; }

    Value evaluate(const Position& pos);

private:
    Value negamax(Position& pos, Value alpha, Value beta, int depth, int ply, bool cut_node);
    Value quiescence(Position& pos, Value alpha, Value beta, int ply);
    bool should_stop();
    void update_quiet_history(Color c, Move best, Move* tried, int n_tried, int depth, int ply);

    TranspositionTable& tt_;
    std::unique_ptr<nnue::Evaluator> nnue_;
    std::atomic<bool> stop_{false};
    std::uint64_t nodes_ = 0;
    std::uint64_t node_limit_ = 0;
    int seldepth_ = 0;
    Value last_score_ = 0;
    std::chrono::steady_clock::time_point start_;
    std::int64_t soft_ms_ = 0, hard_ms_ = 0;

    Move pv_[MAX_PLY][MAX_PLY];
    int pv_len_[MAX_PLY] = {};
    Move killers_[MAX_PLY][2];
    int history_[2][64][64] = {};
    Value static_eval_[MAX_PLY] = {};
};

}  // namespace rookery
