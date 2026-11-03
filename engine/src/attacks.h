// Attack tables: leapers by lookup, sliders by magic bitboards.
#pragma once

#include "types.h"

namespace rookery::attacks {

void init();

extern Bitboard pawn[2][64];
extern Bitboard knight[64];
extern Bitboard king[64];
// Squares strictly between two aligned squares (0 if not aligned), and the full line through them.
extern Bitboard between[64][64];
extern Bitboard line[64][64];

Bitboard bishop(Square s, Bitboard occupied);
Bitboard rook(Square s, Bitboard occupied);
inline Bitboard queen(Square s, Bitboard occupied) { return bishop(s, occupied) | rook(s, occupied); }

// Attacks by a piece of the given type (not pawns) from s.
Bitboard of(PieceType t, Square s, Bitboard occupied);

}  // namespace rookery::attacks
