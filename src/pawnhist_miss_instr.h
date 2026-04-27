// Pawnhist do_move prefetch consumer-site cache-miss instrumentation
// (research-only, not shipped).
//
// Enabled by compiling with -DPAWNHIST_MISS_INSTRUMENT=1 (forced below
// since this branch is dedicated to the measurement). With the flag
// undefined the hooks expand to (void)0.
//
// Five named consumer sites are tracked:
//   0 PAWN_CORR     - search.cpp correction_value: pawn_correction read
//   1 MINOR_CORR    - search.cpp correction_value: minor_piece_correction read
//   2 NONPAWN_W     - search.cpp correction_value: nonpawn_W read
//   3 NONPAWN_B     - search.cpp correction_value: nonpawn_B read
//   4 PAWN_ENTRY    - movepick.cpp score<QUIETS>: pawn_entry read
//
// Same calibration and dump pattern as src/lowply_miss_instr.{h,cpp}
// (clflushopt-derived L1/L2/L3/DRAM thresholds, per-PID JSON dump on
// atexit).
//
// Output dir: $PAWNHIST_MISS_DIR (default scratchpad/pawnhist-miss-instr/).
// Tag: $PAWNHIST_MISS_TAG (default "unknown").

#pragma once

#ifndef PAWNHIST_MISS_INSTRUMENT
    #define PAWNHIST_MISS_INSTRUMENT 1
#endif

#if defined(PAWNHIST_MISS_INSTRUMENT) && PAWNHIST_MISS_INSTRUMENT

    #include <atomic>
    #include <cstdint>
    #include <string>

    #if defined(_MSC_VER)
        #include <intrin.h>
    #else
        #include <x86intrin.h>
    #endif

namespace Stockfish::PawnhistInstr {

constexpr int      PH_NSITES          = 5;
constexpr unsigned READ_SAMPLE_STRIDE = 64;

enum SiteId : int {
    SITE_PAWN_CORR  = 0,
    SITE_MINOR_CORR = 1,
    SITE_NONPAWN_W  = 2,
    SITE_NONPAWN_B  = 3,
    SITE_PAWN_ENTRY = 4
};

struct Bucket {
    std::atomic<std::uint64_t> l1{0}, l2{0}, l3{0}, dram{0};
};

struct PerThread {
    Bucket   read_hist[PH_NSITES];
    unsigned read_phase = 0;
    bool     registered = false;
    int      thread_idx = -1;
};

extern std::uint64_t threshold_l1;
extern std::uint64_t threshold_l2;
extern std::uint64_t threshold_l3;
extern std::string   output_dir;
extern std::string   config_tag;

PerThread& tls();
void       calibrate();

inline void record(Bucket* hist, int site, std::uint64_t cycles) {
    if (site < 0 || site >= PH_NSITES)
        return;
    Bucket& b = hist[site];
    if (cycles < threshold_l1)
        b.l1.fetch_add(1, std::memory_order_relaxed);
    else if (cycles < threshold_l2)
        b.l2.fetch_add(1, std::memory_order_relaxed);
    else if (cycles < threshold_l3)
        b.l3.fetch_add(1, std::memory_order_relaxed);
    else
        b.dram.fetch_add(1, std::memory_order_relaxed);
}

inline std::uint64_t rdtscp_now() {
    unsigned aux;
    return __rdtscp(&aux);
}

template<typename F>
inline auto timed_call(F&& f, int site) -> decltype(f()) {
    auto& s = tls();
    if (++s.read_phase % READ_SAMPLE_STRIDE != 0)
        return f();
    _mm_lfence();
    std::uint64_t t0 = rdtscp_now();
    _mm_lfence();
    auto v = f();
    _mm_lfence();
    std::uint64_t t1 = rdtscp_now();
    record(s.read_hist, site, t1 - t0);
    return v;
}

}  // namespace Stockfish::PawnhistInstr

    #define PAWNHIST_INSTR_READ(EXPR, SITE) \
        Stockfish::PawnhistInstr::timed_call([&]() -> int { return int(EXPR); }, (SITE))

#else  // !PAWNHIST_MISS_INSTRUMENT

    #define PAWNHIST_INSTR_READ(EXPR, SITE) (EXPR)

#endif
