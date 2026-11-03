// The neural evaluation (NNUE): 768 inputs (piece type and colour relative to a side, times
// square, mirrored for Black) feed a hidden layer of HIDDEN neurons, computed once for each side;
// the side to move's half and the other half go through SCReLU into one output.
//
// The hidden layer is an accumulator updated incrementally as moves are made: a move changes two
// or three inputs, so it costs a few hundred additions instead of a full recomputation.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "types.h"

namespace rookery {

class Position;

namespace nnue {

constexpr int INPUTS = 768;
constexpr int QA = 255;    // quantisation of the hidden layer
constexpr int QB = 64;     // quantisation of the output weights
constexpr int SCALE = 400; // network output to centipawns

struct Network {
    int hidden = 0;
    std::vector<std::int16_t> ft_weights;  // INPUTS x hidden
    std::vector<std::int16_t> ft_bias;     // hidden
    std::vector<std::int16_t> out_weights; // 2 x hidden: side to move, then the other side
    std::int32_t out_bias = 0;
};

// Loads a network written by trainer/rookery_train/export.py; returns nullptr on a bad file.
std::shared_ptr<const Network> load(const std::string& path, std::string* error = nullptr);

// The input index of a piece on a square, seen from `perspective`.
inline int feature(Color perspective, Piece p, Square s) {
    int own = color_of(p) == perspective ? 0 : 1;
    Square sq = perspective == WHITE ? s : flip(s);
    return own * 384 + type_of(p) * 64 + sq;
}

struct Accumulator {
    std::vector<std::int16_t> v[2];  // by perspective
    bool computed = false;
};

// One accumulator per position on the Position's state stack, updated lazily from its parent.
class Evaluator {
public:
    explicit Evaluator(std::shared_ptr<const Network> net) : net_(std::move(net)) {}
    const Network& network() const { return *net_; }

    // Forget all accumulators (call when the root position changes).
    void reset() { stack_.clear(); }
    Value evaluate(const Position& pos);
    // A full evaluation without the incremental machinery, for testing.
    Value evaluate_fresh(const Position& pos) const;

private:
    void refresh(const Position& pos, Accumulator& acc) const;
    Value output(const Accumulator& acc, Color stm) const;

    std::shared_ptr<const Network> net_;
    std::vector<Accumulator> stack_;
    std::vector<Key> keys_;
};

}  // namespace nnue
}  // namespace rookery
