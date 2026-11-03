// Rookery's tests: a tiny runner, no framework needed.
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include <fstream>
#include <regex>
#include <sstream>

#include "eval.h"
#include "movegen.h"
#include "nnue.h"
#include "position.h"
#include "search.h"

using namespace rookery;

namespace {
struct Test {
    const char* name;
    std::function<void()> body;
};
std::vector<Test>& registry() {
    static std::vector<Test> tests;
    return tests;
}
int failures = 0;
struct Register {
    Register(const char* name, std::function<void()> body) { registry().push_back({name, std::move(body)}); }
};
#define TEST(name)                                     \
    static void name();                                \
    static Register register_##name(#name, name);      \
    static void name()
#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (!(cond)) {                                                              \
            std::printf("  FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
            ++failures;                                                             \
        }                                                                           \
    } while (0)
}  // namespace

// Perft counts from the Chess Programming Wiki's standard positions.
TEST(perft_standard_positions) {
    struct Case {
        const char* fen;
        int depth;
        std::uint64_t nodes;
    };
    const Case cases[] = {
        {Position::START_FEN, 5, 4865609},
        {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603},
        {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 6, 11030083},
        {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 5, 15833292},
        {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 4, 2103487},
        {"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 4, 3894594},
    };
    for (const Case& c : cases) {
        Position pos(c.fen);
        std::uint64_t n = perft(pos, c.depth);
        if (n != c.nodes) std::printf("  %s depth %d: %llu, want %llu\n", c.fen, c.depth, (unsigned long long)n, (unsigned long long)c.nodes);
        CHECK(n == c.nodes);
        CHECK(pos.fen() == Position(c.fen).fen());  // make/unmake restored the position
    }
}

TEST(fen_round_trip) {
    for (const char* fen : {Position::START_FEN, "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                            "8/8/8/3pP3/8/8/8/K6k w - d6 0 3"}) {
        CHECK(Position(fen).fen() == fen);
    }
}

// The incrementally updated hash must equal one computed from scratch after any line of moves.
TEST(zobrist_incremental) {
    Position pos("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    std::function<void(int)> walk = [&](int depth) {
        CHECK(pos.key() == pos.compute_key());
        if (!depth) return;
        MoveList list;
        generate(pos, list);
        for (Move m : list) {
            if (!pos.legal(m)) continue;
            pos.make(m);
            walk(depth - 1);
            pos.unmake();
        }
    };
    walk(3);
}

TEST(draws) {
    CHECK(Position("8/8/8/4k3/8/8/2K5/6N1 w - - 0 1").is_draw(0));   // a lone knight
    CHECK(!Position("8/8/8/4k3/8/8/2K5/6R1 w - - 0 1").is_draw(0));  // a rook mates
    CHECK(Position("8/8/8/4k3/8/8/2K5/6R1 w - - 100 80").is_draw(0));
    Position pos;
    for (const char* m : {"g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"}) pos.make(pos.parse_uci(m));
    CHECK(pos.is_draw(0));  // the start position for the third time
}

namespace {
std::shared_ptr<const nnue::Network> network() {
    static auto net = nnue::load(std::string(ROOKERY_NETS_DIR) + "/rookery.nnue");
    return net;
}
}  // namespace

// The engine must give exactly the scores the trainer computed for its exported network.
TEST(nnue_matches_trainer) {
    auto net = network();
    CHECK(net != nullptr);
    if (!net) return;
    std::ifstream in(std::string(ROOKERY_NETS_DIR) + "/check.json");
    std::stringstream text;
    text << in.rdbuf();
    std::string json = text.str();
    std::regex entry(R"re("fen": "([^"]+)",\s*"eval": (-?\d+))re");
    int checked = 0;
    nnue::Evaluator ev(net);
    for (auto it = std::sregex_iterator(json.begin(), json.end(), entry); it != std::sregex_iterator(); ++it) {
        Position pos((*it)[1].str());
        int want = std::stoi((*it)[2].str());
        int got = ev.evaluate_fresh(pos);
        if (got != want) std::printf("  %s: engine %d, trainer %d\n", (*it)[1].str().c_str(), got, want);
        CHECK(got == want);
        ++checked;
    }
    CHECK(checked >= 8);
}

// The incrementally updated accumulators must equal a fresh computation after any line of moves,
// including captures, promotions, castling and en passant.
TEST(nnue_incremental) {
    auto net = network();
    if (!net) return;
    nnue::Evaluator ev(net);
    for (const char* fen : {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
                            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"}) {
        Position pos(fen);
        int compared = 0;
        std::function<void(int)> walk = [&](int depth) {
            if (ev.evaluate(pos) != ev.evaluate_fresh(pos)) {
                std::printf("  differs at %s\n", pos.fen().c_str());
                CHECK(false);
            }
            ++compared;
            if (!depth) return;
            MoveList list;
            generate(pos, list);
            for (Move m : list) {
                if (!pos.legal(m)) continue;
                pos.make(m);
                walk(depth - 1);
                pos.unmake();
            }
        };
        walk(2);
        CHECK(compared > 100);
    }
}

TEST(classical_is_symmetric) {
    // A position and its colour-flipped mirror get the same score for the side to move.
    Position a("r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R2QKB1R w KQ - 1 8");
    Position b("r2qkb1r/pp2bppp/2n1pn2/2pp4/3P4/2N1PN2/PP2BPPP/R1BQ1RK1 b kq - 1 8");
    CHECK(eval::classical(a) == eval::classical(b));
}

static Move best_move(const char* fen, int depth, bool use_net, Value* score = nullptr) {
    static TranspositionTable tt(16);
    tt.clear();
    Search s(tt);
    if (use_net) s.set_network(network());
    Position pos(fen);
    Limits l;
    l.depth = depth;
    Move m = s.think(pos, l);
    if (score) *score = s.last_score();
    return m;
}

TEST(search_finds_mates_and_tactics) {
    for (bool net : {false, true}) {
        Value v;
        // A back-rank mate in one.
        CHECK(move_to_uci(best_move("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", 6, net, &v)) == "d1d8");
        CHECK(v == VALUE_MATE - 1);
        // Mate in two: 1. Kb6 Kb8 2. Rh8# or 1. Kc7 Ka7 2. Ra1#. Pruning hides it at low depths
        // (the static evaluation is already winning), so it takes a deeper search.
        Move m = best_move("k7/8/2K5/8/8/8/8/7R w - - 0 1", 12, net, &v);
        CHECK(move_to_uci(m) == "c6b6" || move_to_uci(m) == "c6c7");
        CHECK(v == VALUE_MATE - 3);
        // Winning a hanging queen.
        CHECK(move_to_uci(best_move("rnb1kbnr/pppp1ppp/8/4p1q1/4P3/3P4/PPP2PPP/RNBQKBNR w KQkq - 1 3", 6, net)) == "c1g5");
        // Stalemate is a draw, not a loss.
        CHECK(best_move("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 4, net).is_null());
    }
}

TEST(search_respects_node_limit) {
    TranspositionTable tt(16);
    Search s(tt);
    Position pos;
    Limits l;
    l.nodes = 20000;
    Move m = s.think(pos, l);
    CHECK(!m.is_null() && pos.legal(m));
    CHECK(s.nodes() <= 20000 + 100);
}

int main() {
    for (const Test& t : registry()) {
        int before = failures;
        t.body();
        std::printf("%s %s\n", failures == before ? "ok  " : "FAIL", t.name);
    }
    std::printf("%d failure(s)\n", failures);
    return failures ? 1 : 0;
}
