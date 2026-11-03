// Pseudo-legal move generation; Position::legal filters the moves that leave the king in check.
#pragma once

#include "types.h"

namespace rookery {

class Position;

struct MoveList {
    Move moves[256];
    int size = 0;
    void add(Move m) { moves[size++] = m; }
    Move* begin() { return moves; }
    Move* end() { return moves + size; }
};

enum class GenType { ALL, NOISY };

// NOISY: captures and queen promotions, for the quiescence search.
void generate(const Position& pos, MoveList& list, GenType type = GenType::ALL);

// Counts the leaf nodes of the legal move tree to the given depth (for checking the generator).
std::uint64_t perft(Position& pos, int depth);

}  // namespace rookery
