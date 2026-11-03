#include "movegen.h"

#include "attacks.h"
#include "position.h"

namespace rookery {

namespace {

void add_promotions(MoveList& list, Square from, Square to, bool capture, GenType type) {
    int base = capture ? PROMO_CAPTURE : PROMO;
    list.add(Move(from, to, base + 3));  // queen
    if (type == GenType::ALL)
        for (int p = 0; p < 3; ++p) list.add(Move(from, to, base + p));
}

void pawn_moves(const Position& pos, MoveList& list, GenType type) {
    Color us = pos.side_to_move();
    Bitboard pawns = pos.pieces(us, PAWN);
    Bitboard empty = ~pos.occupied();
    Bitboard enemies = pos.pieces(~us);
    Bitboard promo_rank = us == WHITE ? RANK_8 : RANK_1;
    Bitboard third_rank = us == WHITE ? (RANK_1 << 16) : (RANK_1 << 40);
    int up = us == WHITE ? NORTH : SOUTH;
    auto forward = [&](Bitboard b) { return us == WHITE ? shift_north(b) : shift_south(b); };

    Bitboard single = forward(pawns) & empty;
    Bitboard promos = single & promo_rank;
    for (Bitboard b = promos; b;) {
        Square to = pop_lsb(b);
        add_promotions(list, to - up, to, false, type);
    }
    if (type == GenType::ALL) {
        for (Bitboard b = single & ~promo_rank; b;) {
            Square to = pop_lsb(b);
            list.add(Move(to - up, to, QUIET));
        }
        for (Bitboard b = forward(single & third_rank) & empty; b;) {
            Square to = pop_lsb(b);
            list.add(Move(to - 2 * up, to, DOUBLE_PUSH));
        }
    }
    for (Bitboard b = pawns; b;) {
        Square from = pop_lsb(b);
        Bitboard caps = attacks::pawn[us][from] & enemies;
        for (Bitboard c = caps; c;) {
            Square to = pop_lsb(c);
            if (bb(to) & promo_rank)
                add_promotions(list, from, to, true, type);
            else
                list.add(Move(from, to, CAPTURE));
        }
        Square ep = pos.ep_square();
        if (ep != NO_SQUARE && (attacks::pawn[us][from] & bb(ep))) list.add(Move(from, ep, EN_PASSANT));
    }
}

void piece_moves(const Position& pos, MoveList& list, GenType type) {
    Color us = pos.side_to_move();
    Bitboard occ = pos.occupied();
    Bitboard targets = type == GenType::NOISY ? pos.pieces(~us) : ~pos.pieces(us);
    for (PieceType t : {KNIGHT, BISHOP, ROOK, QUEEN, KING}) {
        for (Bitboard b = pos.pieces(us, t); b;) {
            Square from = pop_lsb(b);
            for (Bitboard a = attacks::of(t, from, occ) & targets; a;) {
                Square to = pop_lsb(a);
                list.add(Move(from, to, pos.piece_on(to) != NO_PIECE ? CAPTURE : QUIET));
            }
        }
    }
}

void castling_moves(const Position& pos, MoveList& list) {
    Color us = pos.side_to_move();
    int rights = pos.castling();
    Square k = us == WHITE ? make_square(4, 0) : make_square(4, 7);
    int oo = us == WHITE ? WHITE_OO : BLACK_OO;
    int ooo = us == WHITE ? WHITE_OOO : BLACK_OOO;
    if (!(rights & (oo | ooo)) || pos.piece_on(k) != make_piece(us, KING) || pos.attacked(k, ~us)) return;
    Bitboard occ = pos.occupied();
    if ((rights & oo) && !(occ & (bb(k + 1) | bb(k + 2))) && pos.piece_on(k + 3) == make_piece(us, ROOK) &&
        !pos.attacked(k + 1, ~us) && !pos.attacked(k + 2, ~us))
        list.add(Move(k, k + 2, KING_CASTLE));
    if ((rights & ooo) && !(occ & (bb(k - 1) | bb(k - 2) | bb(k - 3))) && pos.piece_on(k - 4) == make_piece(us, ROOK) &&
        !pos.attacked(k - 1, ~us) && !pos.attacked(k - 2, ~us))
        list.add(Move(k, k - 2, QUEEN_CASTLE));
}

}  // namespace

void generate(const Position& pos, MoveList& list, GenType type) {
    list.size = 0;
    pawn_moves(pos, list, type);
    piece_moves(pos, list, type);
    if (type == GenType::ALL) castling_moves(pos, list);
}

std::uint64_t perft(Position& pos, int depth) {
    MoveList list;
    generate(pos, list);
    std::uint64_t nodes = 0;
    for (Move m : list) {
        if (!pos.legal(m)) continue;
        if (depth <= 1) {
            ++nodes;
            continue;
        }
        pos.make(m);
        nodes += perft(pos, depth - 1);
        pos.unmake();
    }
    return nodes;
}

}  // namespace rookery
