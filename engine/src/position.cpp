#include "position.h"

#include <sstream>

#include "attacks.h"
#include "movegen.h"

namespace rookery {

namespace zobrist {
Key piece_square[12][64];
Key castling[16];
Key ep_file[8];
Key side;

void init() {
    std::uint64_t s = 0x2545F4914F6CDD1DULL;
    auto next = [&] {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return s * 2685821657736338717ULL;
    };
    for (auto& p : piece_square)
        for (auto& k : p) k = next();
    for (auto& k : castling) k = next();
    for (auto& k : ep_file) k = next();
    side = next();
}
}  // namespace zobrist

namespace {
const std::string PIECE_CHARS = "PNBRQKpnbrqk";

// Castling rights that survive a move touching each square (a rook or king moving or captured).
int castling_mask[64];

void init_castling_mask() {
    for (int& m : castling_mask) m = 15;
    castling_mask[make_square(4, 0)] &= ~(WHITE_OO | WHITE_OOO);
    castling_mask[make_square(7, 0)] &= ~WHITE_OO;
    castling_mask[make_square(0, 0)] &= ~WHITE_OOO;
    castling_mask[make_square(4, 7)] &= ~(BLACK_OO | BLACK_OOO);
    castling_mask[make_square(7, 7)] &= ~BLACK_OO;
    castling_mask[make_square(0, 7)] &= ~BLACK_OOO;
}
}  // namespace

std::string square_name(Square s) {
    return {char('a' + file_of(s)), char('1' + rank_of(s))};
}

std::string move_to_uci(Move m) {
    if (m.is_null()) return "0000";
    std::string s = square_name(m.from()) + square_name(m.to());
    if (m.is_promotion()) s += "nbrq"[m.promotion() - KNIGHT];
    return s;
}

void Position::put(Piece p, Square s) {
    board_[s] = p;
    bitboards_[p] |= bb(s);
    colors_[color_of(p)] |= bb(s);
}

void Position::remove(Square s) {
    Piece p = board_[s];
    bitboards_[p] &= ~bb(s);
    colors_[color_of(p)] &= ~bb(s);
    board_[s] = NO_PIECE;
}

void Position::move_piece(Square from, Square to) {
    Piece p = board_[from];
    Bitboard fromto = bb(from) | bb(to);
    bitboards_[p] ^= fromto;
    colors_[color_of(p)] ^= fromto;
    board_[to] = p;
    board_[from] = NO_PIECE;
}

bool Position::set_fen(const std::string& fen) {
    static bool once = [] {
        attacks::init();
        zobrist::init();
        init_castling_mask();
        return true;
    }();
    (void)once;

    for (auto& b : bitboards_) b = 0;
    colors_[0] = colors_[1] = 0;
    for (auto& p : board_) p = NO_PIECE;
    states_.assign(1, StateInfo{});
    StateInfo& st = states_.back();

    std::istringstream in(fen);
    std::string placement, side, castle, ep;
    int halfmove = 0, fullmove = 1;
    in >> placement >> side >> castle >> ep;
    if (!(in >> halfmove)) halfmove = 0;
    if (!(in >> fullmove)) fullmove = 1;

    int rank = 7, file = 0;
    for (char c : placement) {
        if (c == '/') {
            --rank;
            file = 0;
        } else if (c >= '1' && c <= '8') {
            file += c - '0';
        } else {
            auto idx = PIECE_CHARS.find(c);
            if (idx == std::string::npos || file > 7 || rank < 0) return set_fen(START_FEN);
            put(Piece(idx), make_square(file, rank));
            ++file;
        }
    }
    if (popcount(pieces(WHITE, KING)) != 1 || popcount(pieces(BLACK, KING)) != 1) return set_fen(START_FEN);
    stm_ = side == "b" ? BLACK : WHITE;
    for (char c : castle) {
        if (c == 'K') st.castling |= WHITE_OO;
        if (c == 'Q') st.castling |= WHITE_OOO;
        if (c == 'k') st.castling |= BLACK_OO;
        if (c == 'q') st.castling |= BLACK_OOO;
    }
    if (ep.size() == 2 && ep[0] >= 'a' && ep[0] <= 'h' && (ep[1] == '3' || ep[1] == '6'))
        st.ep = make_square(ep[0] - 'a', ep[1] - '1');
    st.halfmove = halfmove;
    fullmove_ = fullmove;
    st.key = compute_key();
    return true;
}

std::string Position::fen() const {
    std::ostringstream out;
    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            Piece p = board_[make_square(file, rank)];
            if (p == NO_PIECE) {
                ++empty;
                continue;
            }
            if (empty) out << empty;
            empty = 0;
            out << PIECE_CHARS[p];
        }
        if (empty) out << empty;
        if (rank) out << '/';
    }
    out << (stm_ == WHITE ? " w " : " b ");
    int c = castling();
    if (!c) out << '-';
    if (c & WHITE_OO) out << 'K';
    if (c & WHITE_OOO) out << 'Q';
    if (c & BLACK_OO) out << 'k';
    if (c & BLACK_OOO) out << 'q';
    out << ' ' << (ep_square() == NO_SQUARE ? "-" : square_name(ep_square()));
    out << ' ' << halfmove_clock() << ' ' << fullmove_;
    return out.str();
}

// The en passant square only changes the hash when a pawn can actually capture there, so that
// positions that are the same for play are the same for repetition.
static bool ep_capturable(const Position& pos, Square ep) {
    if (ep == NO_SQUARE) return false;
    Color us = pos.side_to_move();
    return attacks::pawn[~us][ep] & pos.pieces(us, PAWN);
}

Key Position::compute_key() const {
    Key k = 0;
    for (Square s = 0; s < 64; ++s)
        if (board_[s] != NO_PIECE) k ^= zobrist::piece_square[board_[s]][s];
    k ^= zobrist::castling[castling()];
    if (ep_capturable(*this, ep_square())) k ^= zobrist::ep_file[file_of(ep_square())];
    if (stm_ == BLACK) k ^= zobrist::side;
    return k;
}

Bitboard Position::attackers_to(Square s, Bitboard occ) const {
    return (attacks::pawn[WHITE][s] & pieces(BLACK, PAWN)) | (attacks::pawn[BLACK][s] & pieces(WHITE, PAWN)) |
           (attacks::knight[s] & pieces(KNIGHT)) | (attacks::king[s] & pieces(KING)) |
           (attacks::bishop(s, occ) & (pieces(BISHOP) | pieces(QUEEN))) |
           (attacks::rook(s, occ) & (pieces(ROOK) | pieces(QUEEN)));
}

bool Position::attacked(Square s, Color by) const {
    return attackers_to(s, occupied()) & colors_[by];
}

Bitboard Position::checkers() const {
    return attackers_to(king_square(stm_), occupied()) & colors_[~stm_];
}

bool Position::legal(Move m) const {
    if (m.is_castle()) return true;  // checked when generated
    Color us = stm_;
    Square from = m.from(), to = m.to();
    Square ksq = type_of(board_[from]) == KING ? to : king_square(us);
    Bitboard occ = (occupied() ^ bb(from)) | bb(to);
    Bitboard removed = bb(to);
    if (m.flags() == EN_PASSANT) {
        Square cap = us == WHITE ? to - 8 : to + 8;
        occ ^= bb(cap);
        removed |= bb(cap);
    }
    return !(attackers_to(ksq, occ) & colors_[~us] & ~removed);
}

void Position::make(Move m) {
    StateInfo next;
    const StateInfo& prev = states_.back();
    next.key = prev.key ^ zobrist::side ^ zobrist::castling[prev.castling];
    if (ep_capturable(*this, prev.ep)) next.key ^= zobrist::ep_file[file_of(prev.ep)];
    next.castling = prev.castling;
    next.halfmove = prev.halfmove + 1;
    next.move = m;

    Color us = stm_;
    Square from = m.from(), to = m.to();
    Piece p = board_[from];
    DirtyPiece& d = next.dirty;

    if (m.is_castle()) {
        Square rook_from = m.flags() == KING_CASTLE ? to + 1 : to - 2;
        Square rook_to = m.flags() == KING_CASTLE ? to - 1 : to + 1;
        Piece rook = board_[rook_from];
        move_piece(from, to);
        move_piece(rook_from, rook_to);
        next.key ^= zobrist::piece_square[p][from] ^ zobrist::piece_square[p][to] ^
                    zobrist::piece_square[rook][rook_from] ^ zobrist::piece_square[rook][rook_to];
        d = {2, {p, rook}, {from, rook_from}, {to, rook_to}};
    } else {
        if (m.is_capture()) {
            Square cap = m.flags() == EN_PASSANT ? (us == WHITE ? to - 8 : to + 8) : to;
            next.captured = board_[cap];
            next.key ^= zobrist::piece_square[next.captured][cap];
            remove(cap);
            next.halfmove = 0;
            d.piece[d.count] = next.captured;
            d.from[d.count] = cap;
            d.to[d.count++] = NO_SQUARE;
        }
        move_piece(from, to);
        next.key ^= zobrist::piece_square[p][from] ^ zobrist::piece_square[p][to];
        if (type_of(p) == PAWN) {
            next.halfmove = 0;
            if (m.is_promotion()) {
                Piece promo = make_piece(us, m.promotion());
                remove(to);
                put(promo, to);
                next.key ^= zobrist::piece_square[p][to] ^ zobrist::piece_square[promo][to];
                // The pawn leaves the board and the new piece appears.
                d.piece[d.count] = p;
                d.from[d.count] = from;
                d.to[d.count++] = NO_SQUARE;
                d.piece[d.count] = promo;
                d.from[d.count] = NO_SQUARE;
                d.to[d.count++] = to;
            } else {
                d.piece[d.count] = p;
                d.from[d.count] = from;
                d.to[d.count++] = to;
            }
            if (m.flags() == DOUBLE_PUSH) next.ep = us == WHITE ? from + 8 : from - 8;
        } else {
            d.piece[d.count] = p;
            d.from[d.count] = from;
            d.to[d.count++] = to;
        }
    }
    next.castling &= castling_mask[from] & castling_mask[to];
    next.key ^= zobrist::castling[next.castling];
    stm_ = ~stm_;
    if (stm_ == WHITE) ++fullmove_;
    states_.push_back(next);
    if (ep_capturable(*this, next.ep)) states_.back().key ^= zobrist::ep_file[file_of(next.ep)];
}

void Position::unmake() {
    const StateInfo st = states_.back();
    states_.pop_back();
    stm_ = ~stm_;
    if (stm_ == BLACK) --fullmove_;
    Move m = st.move;
    Square from = m.from(), to = m.to();
    if (m.is_castle()) {
        Square rook_from = m.flags() == KING_CASTLE ? to + 1 : to - 2;
        Square rook_to = m.flags() == KING_CASTLE ? to - 1 : to + 1;
        move_piece(to, from);
        move_piece(rook_to, rook_from);
        return;
    }
    if (m.is_promotion()) {
        remove(to);
        put(make_piece(stm_, PAWN), to);
    }
    move_piece(to, from);
    if (st.captured != NO_PIECE) {
        Square cap = m.flags() == EN_PASSANT ? (stm_ == WHITE ? to - 8 : to + 8) : to;
        put(st.captured, cap);
    }
}

void Position::make_null() {
    StateInfo next;
    const StateInfo& prev = states_.back();
    next.key = prev.key ^ zobrist::side;
    if (ep_capturable(*this, prev.ep)) next.key ^= zobrist::ep_file[file_of(prev.ep)];
    next.castling = prev.castling;
    next.halfmove = prev.halfmove + 1;
    stm_ = ~stm_;
    states_.push_back(next);
}

void Position::unmake_null() {
    states_.pop_back();
    stm_ = ~stm_;
}

bool Position::has_non_pawn_material(Color c) const {
    return pieces(c) & ~pieces(c, PAWN) & ~pieces(c, KING);
}

bool Position::is_draw(int ply) const {
    const StateInfo& st = states_.back();
    if (st.halfmove >= 100) {
        if (!in_check()) return true;
        // Checkmate on the hundredth half-move still wins.
        MoveList list;
        generate(*this, list);
        for (Move m : list)
            if (legal(m)) return true;
        return false;
    }
    // Repetition: positions with the same side to move, back to the last irreversible move.
    int count = 0;
    int n = int(states_.size());
    for (int i = n - 3; i >= 0 && i >= n - 1 - st.halfmove; i -= 2) {
        if (states_[i].key == st.key) {
            // Inside the search one repetition is enough; before the root it takes two.
            if (i >= n - 1 - ply) return true;
            if (++count == 2) return true;
        }
    }
    // Insufficient material: bare kings, or a single minor piece.
    if (pieces(PAWN) | pieces(ROOK) | pieces(QUEEN)) return false;
    return popcount(pieces(KNIGHT) | pieces(BISHOP)) <= 1;
}

Move Position::parse_uci(const std::string& uci) const {
    MoveList list;
    generate(*this, list);
    for (Move m : list)
        if (move_to_uci(m) == uci && legal(m)) return m;
    return Move();
}

}  // namespace rookery
