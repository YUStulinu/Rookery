// A chess position: bitboards, a mailbox, and a stack of states for unmaking moves.
#pragma once

#include <string>
#include <vector>

#include "types.h"

namespace rookery {

// The pieces a move changed, for updating the network's accumulators incrementally:
// each entry moves a piece from `from` to `to` (NO_SQUARE for removed or added).
struct DirtyPiece {
    int count = 0;
    Piece piece[3];
    Square from[3];
    Square to[3];
};

struct StateInfo {
    Key key = 0;
    int castling = 0;
    Square ep = NO_SQUARE;
    int halfmove = 0;
    Piece captured = NO_PIECE;
    Move move;
    DirtyPiece dirty;
};

class Position {
public:
    static constexpr const char* START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    Position() { set_fen(START_FEN); }
    explicit Position(const std::string& fen) { set_fen(fen); }

    // Returns false (and leaves the start position) if the FEN cannot be read.
    bool set_fen(const std::string& fen);
    std::string fen() const;

    Color side_to_move() const { return stm_; }
    Piece piece_on(Square s) const { return board_[s]; }
    Bitboard pieces(Color c) const { return colors_[c]; }
    Bitboard pieces(Color c, PieceType t) const { return bitboards_[make_piece(c, t)]; }
    Bitboard pieces(PieceType t) const { return bitboards_[t] | bitboards_[t + 6]; }
    Bitboard occupied() const { return colors_[WHITE] | colors_[BLACK]; }
    Square king_square(Color c) const { return lsb(pieces(c, KING)); }

    const StateInfo& state() const { return states_.back(); }
    const StateInfo& state_at(std::size_t i) const { return states_[i]; }
    std::size_t state_count() const { return states_.size(); }
    Key key() const { return state().key; }
    int castling() const { return state().castling; }
    Square ep_square() const { return state().ep; }
    int halfmove_clock() const { return state().halfmove; }
    int fullmove() const { return fullmove_; }

    Bitboard attackers_to(Square s, Bitboard occupied) const;
    bool attacked(Square s, Color by) const;
    Bitboard checkers() const;
    bool in_check() const { return checkers() != 0; }

    // Whether a pseudo-legal move leaves the own king safe.
    bool legal(Move m) const;
    void make(Move m);
    void unmake();
    void make_null();
    void unmake_null();

    // Draw by the fifty-move rule, repetition (once inside the search, twice before it) or
    // insufficient material. `ply` is the distance from the search root.
    bool is_draw(int ply) const;
    bool has_non_pawn_material(Color c) const;

    // Parses a move in UCI notation ("e2e4", "e7e8q") into one of this position's legal moves.
    Move parse_uci(const std::string& uci) const;

    Key compute_key() const;

private:
    void put(Piece p, Square s);
    void remove(Square s);
    void move_piece(Square from, Square to);

    Bitboard bitboards_[12] = {};
    Bitboard colors_[2] = {};
    Piece board_[64];
    Color stm_ = WHITE;
    int fullmove_ = 1;
    std::vector<StateInfo> states_;
};

namespace zobrist {
void init();
}

}  // namespace rookery
