#pragma once
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace hillserve {
using Token = std::uint64_t;
struct Scope {
    std::uint64_t request, layer, step, head, epoch;
    bool operator==(const Scope & b) const;
};
struct Score { Token token; double value; };
struct ScoreBlock {
    Scope scope;
    std::string representation;
    std::string partition;
    std::vector<Score> scores;
};
// Completion barrier and identity validation; no implicit policy or head aggregation.
class ScoreMerge {
    Scope scope_;
    std::string representation_;
    std::map<Token, std::string> owners_;
    std::map<Token, double> received_;
public:
    ScoreMerge(Scope, std::string, const std::map<std::string, std::vector<Token>> &, const std::vector<Token>& universe);
    void accept(const ScoreBlock &);
    bool complete() const;
    std::vector<Score> finish() const;
};
// Explicitly diagnostic, not the paper's unresolved speculative selector.
std::vector<Token> diagnostic_top_k(std::vector<Score>, std::size_t k);
std::vector<double> diagnostic_full_dot_product(const std::vector<float>& q,
                                               const std::vector<float>& keys);
class ScoreRing {
    std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<ScoreBlock> blocks_;
    std::size_t slots_, bytes_limit_, bytes_ = 0;
    bool closed_ = false;
    std::string failure_;
public:
    ScoreRing(std::size_t slots, std::size_t bytes);
    void push(ScoreBlock block); // owns the block; only completed producers may push
    bool pop(ScoreBlock & block);
    void close();
    void fail(std::string message);
};
// Placement mechanics, not a paper HPM replacement policy.
class HotPool {
    std::uint64_t epoch_ = 0;
    std::size_t limit_, used_ = 0;
    std::map<Token, std::vector<std::uint8_t>> payloads_;
public:
    explicit HotPool(std::size_t limit) : limit_(limit) {}
    void apply(std::uint64_t expected_epoch, const std::set<Token> & durable_backing,
               const std::set<Token> & evict,
               std::map<Token, std::vector<std::uint8_t>> promote);
    std::size_t used() const { return used_; }
    std::uint64_t epoch() const { return epoch_; }
    std::set<Token> members() const;
};
struct Budget {
    std::uint64_t hot, pending, score, staging, metadata, limit;
    void validate() const;
};
struct Profile { double cpu_score, fpga_score, cpu_transfer, fpga_transfer; };
// HPCA Eq. 5 estimator only. No claimed empirical correction or cross-layer APP.
double paper_beta_estimate(Profile, double retention);
double hot_fraction(double beta);
void validate_prefetch(bool app, bool representation_ready, bool confirmed_rehearsal);
}
