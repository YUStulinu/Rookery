#include "attacks.h"

#include <vector>

namespace rookery::attacks {

Bitboard pawn[2][64];
Bitboard knight[64];
Bitboard king[64];
Bitboard between[64][64];
Bitboard line[64][64];

namespace {

struct Magic {
    Bitboard mask;
    Bitboard magic;
    int shift;
    Bitboard* table;
};

Magic bishop_magics[64];
Magic rook_magics[64];
std::vector<Bitboard> bishop_table(0x1480);
std::vector<Bitboard> rook_table(0x19000);

// Attacks of a slider walking in the given directions until it hits a piece (included).
Bitboard slide(Square s, Bitboard occupied, const int (*dirs)[2]) {
    Bitboard result = 0;
    for (int d = 0; d < 4; ++d) {
        int f = file_of(s), r = rank_of(s);
        while (true) {
            f += dirs[d][0];
            r += dirs[d][1];
            if (f < 0 || f > 7 || r < 0 || r > 7) break;
            result |= bb(make_square(f, r));
            if (occupied & bb(make_square(f, r))) break;
        }
    }
    return result;
}

constexpr int BISHOP_DIRS[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
constexpr int ROOK_DIRS[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

// A small deterministic generator (xorshift64*) for finding magics, so start-up is repeatable.
struct Prng {
    std::uint64_t s;
    std::uint64_t next() {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return s * 2685821657736338717ULL;
    }
    // Numbers with few bits set make good magic candidates.
    std::uint64_t sparse() { return next() & next() & next(); }
};

void init_magics(Magic* magics, Bitboard* table, const int (*dirs)[2]) {
    Prng rng{0x9E3779B97F4A7C15ULL};
    Bitboard* next_table = table;
    std::vector<Bitboard> occupancy(4096), reference(4096);
    std::vector<int> epoch(4096, 0);
    int attempt = 0;
    for (Square s = 0; s < 64; ++s) {
        // Edges do not matter for blocking unless the piece is on them.
        Bitboard edges = ((RANK_1 | RANK_8) & ~(RANK_1 << (8 * rank_of(s)))) |
                         ((FILE_A | FILE_H) & ~(FILE_A << file_of(s)));
        Magic& m = magics[s];
        m.mask = slide(s, 0, dirs) & ~edges;
        int bits = popcount(m.mask);
        m.shift = 64 - bits;
        m.table = next_table;
        int size = 0;
        // Enumerate every subset of the mask (Carry-Rippler).
        Bitboard b = 0;
        do {
            occupancy[size] = b;
            reference[size] = slide(s, b, dirs);
            ++size;
            b = (b - m.mask) & m.mask;
        } while (b);
        for (;;) {
            m.magic = rng.sparse();
            if (popcount((m.magic * m.mask) >> 56) < 6) continue;
            ++attempt;
            bool ok = true;
            for (int i = 0; i < size && ok; ++i) {
                unsigned idx = unsigned((occupancy[i] * m.magic) >> m.shift);
                if (epoch[idx] < attempt) {
                    epoch[idx] = attempt;
                    m.table[idx] = reference[i];
                } else if (m.table[idx] != reference[i]) {
                    ok = false;
                }
            }
            if (ok) break;
        }
        next_table += std::size_t(1) << bits;
    }
}

}  // namespace

Bitboard bishop(Square s, Bitboard occupied) {
    const Magic& m = bishop_magics[s];
    return m.table[((occupied & m.mask) * m.magic) >> m.shift];
}

Bitboard rook(Square s, Bitboard occupied) {
    const Magic& m = rook_magics[s];
    return m.table[((occupied & m.mask) * m.magic) >> m.shift];
}

Bitboard of(PieceType t, Square s, Bitboard occupied) {
    switch (t) {
        case KNIGHT: return knight[s];
        case BISHOP: return bishop(s, occupied);
        case ROOK: return rook(s, occupied);
        case QUEEN: return queen(s, occupied);
        case KING: return king[s];
        default: return 0;
    }
}

void init() {
    static bool done = false;
    if (done) return;
    done = true;
    for (Square s = 0; s < 64; ++s) {
        Bitboard b = bb(s);
        pawn[WHITE][s] = shift_east(shift_north(b)) | shift_west(shift_north(b));
        pawn[BLACK][s] = shift_east(shift_south(b)) | shift_west(shift_south(b));
        Bitboard n = 0;
        constexpr int JUMPS[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
        for (const auto& j : JUMPS) {
            int f = file_of(s) + j[0], r = rank_of(s) + j[1];
            if (f >= 0 && f < 8 && r >= 0 && r < 8) n |= bb(make_square(f, r));
        }
        knight[s] = n;
        Bitboard k = 0;
        for (int df = -1; df <= 1; ++df)
            for (int dr = -1; dr <= 1; ++dr) {
                int f = file_of(s) + df, r = rank_of(s) + dr;
                if ((df || dr) && f >= 0 && f < 8 && r >= 0 && r < 8) k |= bb(make_square(f, r));
            }
        king[s] = k;
    }
    init_magics(bishop_magics, bishop_table.data(), BISHOP_DIRS);
    init_magics(rook_magics, rook_table.data(), ROOK_DIRS);
    for (Square a = 0; a < 64; ++a)
        for (Square b = 0; b < 64; ++b) {
            between[a][b] = line[a][b] = 0;
            if (a == b) continue;
            for (PieceType t : {BISHOP, ROOK}) {
                if (of(t, a, 0) & bb(b)) {
                    between[a][b] = of(t, a, bb(b)) & of(t, b, bb(a));
                    line[a][b] = (of(t, a, 0) & of(t, b, 0)) | bb(a) | bb(b);
                }
            }
        }
}

}  // namespace rookery::attacks
