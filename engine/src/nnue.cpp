#include "nnue.h"

#include <algorithm>
#include <cstring>
#include <fstream>

#include "position.h"

namespace rookery::nnue {

namespace {

template <typename T>
bool read(std::istream& in, T* data, std::size_t n) {
    in.read(reinterpret_cast<char*>(data), std::streamsize(n * sizeof(T)));
    return bool(in);
}

}  // namespace

// The file: "RKNN", version 1, hidden size, then the arrays in the order of Network, little-endian.
std::shared_ptr<const Network> load(const std::string& path, std::string* error) {
    auto fail = [&](const char* why) -> std::shared_ptr<const Network> {
        if (error) *error = path + ": " + why;
        return nullptr;
    };
    std::ifstream in(path, std::ios::binary);
    if (!in) return fail("cannot open");
    char magic[4];
    std::int32_t version = 0, hidden = 0;
    if (!read(in, magic, 4) || std::memcmp(magic, "RKNN", 4) != 0) return fail("not a Rookery network");
    if (!read(in, &version, 1) || version != 1) return fail("unknown version");
    if (!read(in, &hidden, 1) || hidden <= 0 || hidden > 4096) return fail("bad hidden size");
    auto net = std::make_shared<Network>();
    net->hidden = hidden;
    net->ft_weights.resize(std::size_t(INPUTS) * hidden);
    net->ft_bias.resize(hidden);
    net->out_weights.resize(2 * std::size_t(hidden));
    if (!read(in, net->ft_weights.data(), net->ft_weights.size()) || !read(in, net->ft_bias.data(), net->ft_bias.size()) ||
        !read(in, net->out_weights.data(), net->out_weights.size()) || !read(in, &net->out_bias, 1))
        return fail("file is truncated");
    if (in.peek() != std::char_traits<char>::eof()) return fail("unexpected data at the end");
    return net;
}

void Evaluator::refresh(const Position& pos, Accumulator& acc) const {
    const int h = net_->hidden;
    for (Color persp : {WHITE, BLACK}) {
        acc.v[persp].assign(net_->ft_bias.begin(), net_->ft_bias.end());
        std::int16_t* v = acc.v[persp].data();
        for (Bitboard b = pos.occupied(); b;) {
            Square s = pop_lsb(b);
            const std::int16_t* w = &net_->ft_weights[std::size_t(feature(persp, pos.piece_on(s), s)) * h];
            for (int i = 0; i < h; ++i) v[i] += w[i];
        }
    }
    acc.computed = true;
}

Value Evaluator::output(const Accumulator& acc, Color stm) const {
    const int h = net_->hidden;
    std::int64_t sum = 0;
    const std::int16_t* halves[2] = {acc.v[stm].data(), acc.v[~stm].data()};
    for (int half = 0; half < 2; ++half) {
        const std::int16_t* v = halves[half];
        const std::int16_t* w = &net_->out_weights[std::size_t(half) * h];
        for (int i = 0; i < h; ++i) {
            std::int64_t c = std::clamp<std::int32_t>(v[i], 0, QA);
            sum += c * c * w[i];  // SCReLU: clamp, then square
        }
    }
    sum = sum / QA + net_->out_bias;  // now in units of QA * QB
    Value cp = Value(sum * SCALE / (QA * QB));
    return std::clamp(cp, -VALUE_MATE_IN_MAX + 1, VALUE_MATE_IN_MAX - 1);
}

Value Evaluator::evaluate_fresh(const Position& pos) const {
    Accumulator acc;
    refresh(pos, acc);
    return output(acc, pos.side_to_move());
}

Value Evaluator::evaluate(const Position& pos) {
    const std::size_t n = pos.state_count();
    // Keep accumulators only for states that still match the position's stack.
    if (stack_.size() < n) {
        stack_.resize(n);
        keys_.resize(n, 0);
    }
    // The nearest state, walking back from the current one, whose accumulator is known; a matching
    // key means the same position, so its accumulator is right whatever line led there.
    std::size_t start = n;
    for (std::size_t i = n; i-- > 0;) {
        if (stack_[i].computed && keys_[i] == pos.state_at(i).key) {
            start = i;
            break;
        }
    }
    if (start == n) {
        // Nothing to start from: compute the current state from scratch.
        refresh(pos, stack_[n - 1]);
        keys_[n - 1] = pos.key();
        return output(stack_[n - 1], pos.side_to_move());
    }
    const std::size_t valid = start + 1;
    const int h = net_->hidden;
    for (std::size_t i = valid; i < n; ++i) {
        Accumulator& acc = stack_[i];
        const Accumulator& parent = stack_[i - 1];
        const DirtyPiece& d = pos.state_at(i).dirty;
        for (Color persp : {WHITE, BLACK}) {
            acc.v[persp] = parent.v[persp];
            std::int16_t* v = acc.v[persp].data();
            for (int k = 0; k < d.count; ++k) {
                if (d.from[k] != NO_SQUARE) {
                    const std::int16_t* w = &net_->ft_weights[std::size_t(feature(persp, d.piece[k], d.from[k])) * h];
                    for (int j = 0; j < h; ++j) v[j] -= w[j];
                }
                if (d.to[k] != NO_SQUARE) {
                    const std::int16_t* w = &net_->ft_weights[std::size_t(feature(persp, d.piece[k], d.to[k])) * h];
                    for (int j = 0; j < h; ++j) v[j] += w[j];
                }
            }
        }
        acc.computed = true;
        keys_[i] = pos.state_at(i).key;
    }
    return output(stack_[n - 1], pos.side_to_move());
}

}  // namespace rookery::nnue
