#ifndef FINNY_FIRE_INSTR_H_INCLUDED
#define FINNY_FIRE_INSTR_H_INCLUDED

#include <atomic>
#include <cstdio>

namespace Stockfish::FinnyFireInstr {

inline std::atomic<unsigned long long> domove_total{0};
inline std::atomic<unsigned long long> domove_kingmove{0};
inline std::atomic<unsigned long long> domove_kingmove_white{0};
inline std::atomic<unsigned long long> domove_kingmove_black{0};

struct Dumper {
    ~Dumper() {
        const unsigned long long t  = domove_total.load();
        const unsigned long long k  = domove_kingmove.load();
        const unsigned long long w  = domove_kingmove_white.load();
        const unsigned long long b  = domove_kingmove_black.load();
        const double             pk = t > 0 ? 100.0 * double(k) / double(t) : 0.0;
        std::fprintf(stderr,
                     "FINNY_FIRE: domove_total=%llu kingmove=%llu rate=%.4f%% "
                     "white=%llu black=%llu\n",
                     t, k, pk, w, b);
    }
};

inline Dumper dumper;

}  // namespace Stockfish::FinnyFireInstr

#endif
