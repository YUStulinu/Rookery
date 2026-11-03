// Rookery: a UCI chess engine.
//
// Besides the UCI protocol it understands a few commands of its own:
//   perft <depth>                 count the legal move tree from the current position
//   bench [depth]                 search fixed positions; prints nodes and speed
//   eval                          the classical and the network evaluation of the position
//   features <in.csv> <out.bin> [limit]
//                                 write the classical features of "fen,score" positions for the tuner
//   evaltest <in.csv>             compare both evaluations with Stockfish's on quiet held-out positions
//   randomfen <plies> <seed>      the position after that many random legal moves (for opening sets)
//   positions <in.csv> <out.bin> [limit]
//                                 write "fen,score" positions as fixed records for the network trainer
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

#include "eval.h"
#include "movegen.h"
#include "nnue.h"
#include "position.h"
#include "search.h"

using namespace rookery;

namespace {

const char* BENCH_FENS[] = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "2r3k1/pp3ppp/2n5/3p4/3P4/2P2N2/P4PPP/2R3K1 b - - 0 24",
    "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R2QKB1R w KQ - 1 8",
    "8/8/4k3/3p4/3P4/4K3/8/8 w - - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
};

// Positions where the static evaluation cannot be right are left out of training data: the side
// to move is in check, or can win material at once (a capture that gains by static exchange).
bool noisy(const Position& p) {
    if (p.in_check()) return true;
    MoveList list;
    generate(p, list, GenType::NOISY);
    for (Move m : list)
        if (p.legal(m) && see_ge(p, m, 1)) return true;
    return false;
}

class Engine {
public:
    explicit Engine(const std::string& exe_dir) : tt_(64), search_(tt_) {
        eval_file_ = (std::filesystem::path(exe_dir) / "rookery.nnue").string();
        load_network();
    }
    ~Engine() { stop(); }

    void run(std::istream& input) {
        std::string line;
        while (std::getline(input, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::istringstream in(line);
            std::string cmd;
            in >> cmd;
            if (cmd == "quit") break;
            command(cmd, in);
        }
        stop();
    }

private:
    static void say(const std::string& s) {
        std::cout << s << std::endl;  // endl flushes: GUIs read line by line
    }

    void load_network() {
        if (!use_nnue_) {
            search_.set_network(nullptr);
            return;
        }
        std::string error;
        auto net = nnue::load(eval_file_, &error);
        if (!net)
            say("info string NNUE not loaded (" + error + "); using the classical evaluation");
        else
            say("info string NNUE loaded: " + eval_file_ + " (768 -> " + std::to_string(net->hidden) + "x2 -> 1)");
        search_.set_network(net);
    }

    void stop() {
        search_.stop();
        if (worker_.joinable()) worker_.join();
    }

    void command(const std::string& cmd, std::istringstream& in) {
        if (cmd == "uci") {
            say("id name Rookery 1.0");
            say("id author YUStulinu");
            say("option name Hash type spin default 64 min 1 max 4096");
            say("option name UseNNUE type check default true");
            say("option name EvalFile type string default " + eval_file_);
            say("uciok");
        } else if (cmd == "isready") {
            say("readyok");
        } else if (cmd == "setoption") {
            stop();
            set_option(in);
        } else if (cmd == "ucinewgame") {
            stop();
            tt_.clear();
            search_.clear();
        } else if (cmd == "position") {
            stop();
            set_position(in);
        } else if (cmd == "go") {
            stop();
            go(in);
        } else if (cmd == "stop") {
            stop();
        } else if (cmd == "d") {
            say(pos_.fen());
        } else if (cmd == "eval") {
            say("classical " + std::to_string(eval::classical(pos_)) + " cp (side to move)");
            if (search_.uses_network()) say("network " + std::to_string(search_.evaluate(pos_)) + " cp (side to move)");
        } else if (cmd == "perft") {
            int depth = 1;
            in >> depth;
            auto t0 = std::chrono::steady_clock::now();
            std::uint64_t n = perft(pos_, depth);
            double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            say("nodes " + std::to_string(n) + " time " + std::to_string(int(s * 1000)) + " ms");
        } else if (cmd == "bench") {
            int depth = 10;
            in >> depth;
            bench(depth);
        } else if (cmd == "features") {
            std::string src, dst;
            long long limit = -1;
            in >> src >> dst >> limit;
            write_features(src, dst, limit);
        } else if (cmd == "evaltest") {
            std::string src;
            in >> src;
            eval_test(src);
        } else if (cmd == "randomfen") {
            int plies = 8;
            std::uint64_t seed = 1;
            in >> plies >> seed;
            random_fen(plies, seed);
        } else if (cmd == "positions") {
            std::string src, dst;
            long long limit = -1;
            in >> src >> dst >> limit;
            write_positions(src, dst, limit);
        }
    }

    void set_option(std::istringstream& in) {
        std::string word, name, value;
        in >> word;  // "name"
        while (in >> word && word != "value") name += (name.empty() ? "" : " ") + word;
        std::getline(in, value);
        if (!value.empty() && value[0] == ' ') value.erase(0, 1);
        if (name == "Hash") {
            tt_.resize(std::max(1, std::atoi(value.c_str())));
        } else if (name == "UseNNUE") {
            use_nnue_ = value == "true";
            load_network();
        } else if (name == "EvalFile") {
            eval_file_ = value;
            load_network();
        }
    }

    void set_position(std::istringstream& in) {
        std::string word, fen;
        in >> word;
        if (word == "startpos") {
            fen = Position::START_FEN;
            in >> word;
        } else if (word == "fen") {
            while (in >> word && word != "moves") fen += (fen.empty() ? "" : " ") + word;
        }
        pos_.set_fen(fen);
        if (word == "moves") {
            while (in >> word) {
                Move m = pos_.parse_uci(word);
                if (m.is_null()) {
                    say("info string illegal move " + word);
                    break;
                }
                pos_.make(m);
            }
        }
    }

    void go(std::istringstream& in) {
        Limits limits;
        std::string word;
        while (in >> word) {
            if (word == "depth") in >> limits.depth;
            else if (word == "nodes") in >> limits.nodes;
            else if (word == "movetime") in >> limits.movetime;
            else if (word == "wtime") in >> limits.time[WHITE];
            else if (word == "btime") in >> limits.time[BLACK];
            else if (word == "winc") in >> limits.inc[WHITE];
            else if (word == "binc") in >> limits.inc[BLACK];
            else if (word == "movestogo") in >> limits.movestogo;
            else if (word == "infinite") limits.infinite = true;
        }
        limits.depth = std::clamp(limits.depth, 1, MAX_PLY - 1);
        Position root = pos_;
        worker_ = std::thread([this, limits, root]() mutable {
            Move best = search_.think(root, limits, [](const SearchInfo& i) {
                std::ostringstream o;
                o << "info depth " << i.depth << " seldepth " << i.seldepth << " score ";
                if (std::abs(i.score) >= VALUE_MATE_IN_MAX)
                    o << "mate " << (i.score > 0 ? (VALUE_MATE - i.score + 1) / 2 : -(VALUE_MATE + i.score) / 2);
                else
                    o << "cp " << i.score;
                o << " nodes " << i.nodes << " nps " << (i.millis ? i.nodes * 1000 / i.millis : i.nodes) << " hashfull "
                  << i.hashfull << " time " << i.millis << " pv " << i.pv;
                say(o.str());
            });
            say("bestmove " + move_to_uci(best));
        });
    }

    void bench(int depth) {
        std::uint64_t nodes = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (const char* fen : BENCH_FENS) {
            Position p(fen);
            tt_.clear();
            search_.clear();
            Limits limits;
            limits.depth = depth;
            search_.think(p, limits);
            nodes += search_.nodes();
        }
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        say(std::to_string(nodes) + " nodes " + std::to_string(std::uint64_t(double(nodes) / std::max(s, 1e-3))) + " nps");
    }

    // For each "fen,score" line: phase (1 byte), side to move (1), feature count (2), score (4),
    // then (feature index (2), count (1)) pairs. The tuner reads this back.
    void write_features(const std::string& src, const std::string& dst, long long limit) {
        std::ifstream in(src);
        std::ofstream out(dst, std::ios::binary);
        if (!in || !out) {
            say("info string cannot open " + src + " or " + dst);
            return;
        }
        std::string line;
        eval::Trace t;
        long long n = 0, skipped = 0;
        Position p;
        while ((limit < 0 || n < limit) && std::getline(in, line)) {
            auto comma = line.rfind(',');
            if (comma == std::string::npos) continue;
            p.set_fen(line.substr(0, comma) + " 0 1");
            if (noisy(p)) {
                ++skipped;
                continue;
            }
            std::int32_t score = std::atoi(line.c_str() + comma + 1);
            eval::trace(p, t);
            std::uint8_t phase = std::uint8_t(t.phase), stm = std::uint8_t(t.stm);
            std::uint16_t count = std::uint16_t(t.features.size());
            out.write(reinterpret_cast<char*>(&phase), 1);
            out.write(reinterpret_cast<char*>(&stm), 1);
            out.write(reinterpret_cast<char*>(&count), 2);
            out.write(reinterpret_cast<char*>(&score), 4);
            for (auto [f, c] : t.features) {
                std::uint16_t idx = std::uint16_t(f);
                std::int8_t cnt = std::int8_t(std::clamp(c, -127, 127));
                out.write(reinterpret_cast<char*>(&idx), 2);
                out.write(reinterpret_cast<char*>(&cnt), 1);
            }
            ++n;
        }
        say("wrote features of " + std::to_string(n) + " positions; skipped " + std::to_string(skipped) + " noisy ones");
    }

    // 68-byte records: side to move (1), piece count (1), score from the side to move's view in
    // centipawns, clamped to +-10000 (2), then 32 slots of piece * 64 + square (2 each; 0xFFFF empty).
    void write_positions(const std::string& src, const std::string& dst, long long limit) {
        std::ifstream in(src);
        std::ofstream out(dst, std::ios::binary);
        if (!in || !out) {
            say("info string cannot open " + src + " or " + dst);
            return;
        }
        std::string line;
        long long n = 0, skipped = 0;
        Position p;
        while ((limit < 0 || n < limit) && std::getline(in, line)) {
            auto comma = line.rfind(',');
            if (comma == std::string::npos) continue;
            p.set_fen(line.substr(0, comma) + " 0 1");
            if (noisy(p)) {
                ++skipped;
                continue;
            }
            int score = std::clamp(std::atoi(line.c_str() + comma + 1), -10000, 10000);
            if (p.side_to_move() == BLACK) score = -score;
            std::uint8_t head[2] = {std::uint8_t(p.side_to_move()), std::uint8_t(popcount(p.occupied()))};
            std::int16_t sc = std::int16_t(score);
            std::uint16_t slots[32];
            std::fill(std::begin(slots), std::end(slots), std::uint16_t(0xFFFF));
            int k = 0;
            for (Bitboard b = p.occupied(); b && k < 32;) {
                Square s = pop_lsb(b);
                slots[k++] = std::uint16_t(p.piece_on(s) * 64 + s);
            }
            out.write(reinterpret_cast<char*>(head), 2);
            out.write(reinterpret_cast<char*>(&sc), 2);
            out.write(reinterpret_cast<char*>(slots), sizeof(slots));
            ++n;
        }
        say("wrote " + std::to_string(n) + " positions; skipped " + std::to_string(skipped) + " noisy ones");
    }

    // For each evaluation: the training loss (squared difference of win probabilities), the mean
    // absolute difference in centipawns (scores clamped to +-1000), and how often it agrees with
    // Stockfish on which side is better when Stockfish sees at least half a pawn.
    void eval_test(const std::string& src) {
        std::ifstream in(src);
        if (!in) {
            say("info string cannot open " + src);
            return;
        }
        auto net = nnue::load(eval_file_);
        if (!net) {
            say("info string no network at " + eval_file_);
            return;
        }
        nnue::Evaluator ev(net);
        auto sigmoid = [](double cp) { return 1.0 / (1.0 + std::exp(-cp / 400.0)); };
        double loss[2] = {}, abs_err[2] = {};
        long long agree[2] = {}, decided = 0, n = 0;
        std::string line;
        Position p;
        while (std::getline(in, line)) {
            auto comma = line.rfind(',');
            if (comma == std::string::npos) continue;
            p.set_fen(line.substr(0, comma) + " 0 1");
            if (noisy(p)) continue;
            int target = std::atoi(line.c_str() + comma + 1);
            if (p.side_to_move() == BLACK) target = -target;
            int got[2] = {eval::classical(p), ev.evaluate_fresh(p)};
            for (int k = 0; k < 2; ++k) {
                double d = sigmoid(got[k]) - sigmoid(target);
                loss[k] += d * d;
                abs_err[k] += std::abs(std::clamp(got[k], -1000, 1000) - std::clamp(target, -1000, 1000));
                if (std::abs(target) >= 50 && (got[k] > 0) == (target > 0)) ++agree[k];
            }
            decided += std::abs(target) >= 50;
            ++n;
        }
        const char* names[2] = {"classical", "nnue"};
        std::ostringstream o;
        o << "{\"positions\": " << n;
        for (int k = 0; k < 2; ++k)
            o << ", \"" << names[k] << "\": {\"loss\": " << loss[k] / n << ", \"mean_abs_cp\": " << abs_err[k] / n
              << ", \"right_side_better\": " << double(agree[k]) / std::max(1LL, decided) << "}";
        o << "}";
        say(o.str());
    }

    void random_fen(int plies, std::uint64_t seed) {
        Position p;
        std::uint64_t x = seed * 0x9E3779B97F4A7C15ULL + 1;
        for (int i = 0; i < plies; ++i) {
            MoveList list;
            generate(p, list);
            Move legal[256];
            int n = 0;
            for (Move m : list)
                if (p.legal(m)) legal[n++] = m;
            if (!n) break;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            p.make(legal[(x * 2685821657736338717ULL >> 33) % n]);
        }
        say(p.fen());
    }

    TranspositionTable tt_;
    Search search_;
    Position pos_;
    std::thread worker_;
    std::string eval_file_;
    bool use_nnue_ = true;
};

}  // namespace

int main(int argc, char** argv) {
    std::string dir = argc > 0 ? std::filesystem::absolute(argv[0]).parent_path().string() : ".";
    Engine engine(dir);
    // Commands may also come on the command line: "rookery bench 12".
    if (argc > 1) {
        std::string line;
        for (int i = 1; i < argc; ++i) line += (i > 1 ? " " : "") + std::string(argv[i]);
        std::istringstream in(line + "\nquit\n");
        engine.run(in);
        return 0;
    }
    engine.run(std::cin);
    return 0;
}
