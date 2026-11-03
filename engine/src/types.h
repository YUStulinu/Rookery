// Basic chess types: squares, pieces, moves and bitboards.
#pragma once

#include <bit>
#include <cstdint>
#include <string>

namespace rookery {

using Bitboard = std::uint64_t;
using Key = std::uint64_t;

enum Color : int { WHITE, BLACK };
constexpr Color operator~(Color c) { return Color(c ^ 1); }

enum PieceType : int { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, NO_PIECE_TYPE };

// Piece = colour * 6 + type; NO_PIECE marks an empty square.
enum Piece : int { NO_PIECE = 12 };
constexpr Piece make_piece(Color c, PieceType t) { return Piece(c * 6 + t); }
constexpr Color color_of(Piece p) { return Color(p / 6); }
constexpr PieceType type_of(Piece p) { return PieceType(p % 6); }

// a1 = 0, b1 = 1, ..., h8 = 63.
using Square = int;
constexpr Square NO_SQUARE = 64;
constexpr int file_of(Square s) { return s & 7; }
constexpr int rank_of(Square s) { return s >> 3; }
constexpr Square make_square(int file, int rank) { return rank * 8 + file; }
// The same square seen from Black's side of the board.
constexpr Square flip(Square s) { return s ^ 56; }

constexpr Bitboard bb(Square s) { return Bitboard(1) << s; }
inline int popcount(Bitboard b) { return std::popcount(b); }
inline Square lsb(Bitboard b) { return std::countr_zero(b); }
inline Square pop_lsb(Bitboard& b) {
    Square s = lsb(b);
    b &= b - 1;
    return s;
}

constexpr Bitboard FILE_A = 0x0101010101010101ULL;
constexpr Bitboard FILE_H = FILE_A << 7;
constexpr Bitboard RANK_1 = 0xFFULL;
constexpr Bitboard RANK_2 = RANK_1 << 8;
constexpr Bitboard RANK_7 = RANK_1 << 48;
constexpr Bitboard RANK_8 = RANK_1 << 56;

enum Direction : int { NORTH = 8, SOUTH = -8, EAST = 1, WEST = -1 };

constexpr Bitboard shift_north(Bitboard b) { return b << 8; }
constexpr Bitboard shift_south(Bitboard b) { return b >> 8; }
constexpr Bitboard shift_east(Bitboard b) { return (b & ~FILE_H) << 1; }
constexpr Bitboard shift_west(Bitboard b) { return (b & ~FILE_A) >> 1; }

// Castling rights as four bits.
enum Castling : int { WHITE_OO = 1, WHITE_OOO = 2, BLACK_OO = 4, BLACK_OOO = 8 };

// A move in 16 bits: from (6), to (6), flags (4).
enum MoveFlag : int {
    QUIET = 0,
    DOUBLE_PUSH = 1,
    KING_CASTLE = 2,
    QUEEN_CASTLE = 3,
    CAPTURE = 4,
    EN_PASSANT = 5,
    // Promotions: 8 + piece (knight 0 .. queen 3), +4 for a capture.
    PROMO = 8,
    PROMO_CAPTURE = 12,
};

struct Move {
    std::uint16_t data = 0;

    constexpr Move() = default;
    constexpr Move(Square from, Square to, int flags) : data(std::uint16_t(from | (to << 6) | (flags << 12))) {}
    constexpr Square from() const { return data & 63; }
    constexpr Square to() const { return (data >> 6) & 63; }
    constexpr int flags() const { return data >> 12; }
    constexpr bool is_capture() const { return flags() & CAPTURE; }
    constexpr bool is_promotion() const { return flags() & PROMO; }
    constexpr PieceType promotion() const { return PieceType(KNIGHT + (flags() & 3)); }
    constexpr bool is_castle() const { return flags() == KING_CASTLE || flags() == QUEEN_CASTLE; }
    constexpr bool is_null() const { return data == 0; }
    constexpr bool operator==(const Move& o) const = default;
};

std::string square_name(Square s);
std::string move_to_uci(Move m);

using Value = int;
constexpr Value VALUE_INF = 32001;
constexpr Value VALUE_MATE = 32000;
constexpr Value VALUE_MATE_IN_MAX = VALUE_MATE - 256;
constexpr Value VALUE_NONE = 32002;

}  // namespace rookery
