// The transposition table: what the search learned about positions it has already met.
#pragma once

#include <cstdint>
#include <vector>

#include "types.h"

namespace rookery {

enum Bound : std::uint8_t { BOUND_NONE, BOUND_UPPER, BOUND_LOWER, BOUND_EXACT };

struct TTEntry {
    Key key = 0;
    Move move;
    std::int16_t score = 0;
    std::int16_t eval = 0;
    std::uint8_t depth = 0;
    std::uint8_t bound = BOUND_NONE;
    std::uint8_t generation = 0;
};

class TranspositionTable {
public:
    explicit TranspositionTable(std::size_t mb = 64) { resize(mb); }

    void resize(std::size_t mb) {
        std::size_t n = mb * 1024 * 1024 / sizeof(TTEntry);
        table_.assign(n ? n : 1, TTEntry{});
    }
    void clear() { table_.assign(table_.size(), TTEntry{}); }
    void new_search() { ++generation_; }

    TTEntry* probe(Key key, bool& found) {
        TTEntry* e = &table_[key % table_.size()];
        found = e->key == key && e->bound != BOUND_NONE;
        return e;
    }

    // Keeps the entry for the same position unless the new result is clearly shallower, and
    // replaces entries from older searches or of lesser depth.
    void store(TTEntry* e, Key key, Move move, Value score, Value eval, int depth, Bound bound) {
        bool same = e->key == key;
        if (!same || bound == BOUND_EXACT || depth + 2 >= e->depth || e->generation != generation_) {
            if (!move.is_null() || !same) e->move = move;
            e->key = key;
            e->score = std::int16_t(score);
            e->eval = std::int16_t(eval);
            e->depth = std::uint8_t(depth < 0 ? 0 : depth);
            e->bound = bound;
            e->generation = generation_;
        }
    }

    // Permille of entries written in the current search (for "info hashfull").
    int hashfull() const {
        int used = 0;
        for (std::size_t i = 0; i < 1000 && i < table_.size(); ++i)
            used += table_[i].bound != BOUND_NONE && table_[i].generation == generation_;
        return used;
    }

private:
    std::vector<TTEntry> table_;
    std::uint8_t generation_ = 0;
};

// Mate scores are stored relative to the entry's position, not the root.
inline Value score_to_tt(Value v, int ply) {
    return v >= VALUE_MATE_IN_MAX ? v + ply : v <= -VALUE_MATE_IN_MAX ? v - ply : v;
}
inline Value score_from_tt(Value v, int ply) {
    return v >= VALUE_MATE_IN_MAX ? v - ply : v <= -VALUE_MATE_IN_MAX ? v + ply : v;
}

}  // namespace rookery
