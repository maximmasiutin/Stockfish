// Lowply cache-miss instrumentation (research-only, not shipped).
//
// Enabled by compiling with -DLOWPLY_MISS_INSTRUMENT=1. With the flag
// undefined, all hooks expand to (void)0 and the binary is master-equivalent.
//
// At process start:
//   * calibrate() records cycle-to-residency thresholds by issuing CLFLUSHOPT
//     against a known buffer and timing accesses (Option C in the plan).
// At each instrumented read/write of lowPlyHistory:
//   * a thread-local sampler decides (1/READ_SAMPLE_STRIDE, 1/WRITE_SAMPLE_STRIDE)
//     whether to time the access; if so, the cycle delta is binned into
//     L1 / L2 / L3 / DRAM per ply.
// At process exit:
//   * atexit dumper writes per-PID JSON to $LOWPLY_MISS_DIR (default
//     scratchpad/lowply-miss-instr/) with both calibration thresholds and
//     per-thread per-ply read/write residency histograms.
//
// All output paths are stamped with local + UTC ISO timestamps.

#pragma once

// This branch is dedicated to lowply cache-miss instrumentation; the
// Makefile's profile-build target overwrites EXTRACXXFLAGS, so the define
// is forced here.
#ifndef LOWPLY_MISS_INSTRUMENT
    #define LOWPLY_MISS_INSTRUMENT 1
#endif

#if defined(LOWPLY_MISS_INSTRUMENT) && LOWPLY_MISS_INSTRUMENT

    #include <atomic>
    #include <cstdint>
    #include <cstdio>
    #include <cstdlib>
    #include <cstring>
    #include <ctime>
    #include <mutex>
    #include <string>
    #include <vector>

    #if defined(_MSC_VER)
        #include <intrin.h>
    #else
        #include <x86intrin.h>
    #endif

namespace Stockfish::LowPlyInstr {

constexpr int      LP_NPLY             = 5;
constexpr unsigned READ_SAMPLE_STRIDE  = 64;
constexpr unsigned WRITE_SAMPLE_STRIDE = 8;

struct Bucket {
    std::atomic<std::uint64_t> l1{0}, l2{0}, l3{0}, dram{0};
};

struct PerThread {
    Bucket   read_hist[LP_NPLY];
    Bucket   write_hist[LP_NPLY];
    unsigned read_phase  = 0;
    unsigned write_phase = 0;
    bool     registered  = false;
    int      thread_idx  = -1;
};

// Cycle thresholds: <l1 -> L1; <l2 -> L2; <l3 -> L3; else DRAM.
extern std::uint64_t threshold_l1;
extern std::uint64_t threshold_l2;
extern std::uint64_t threshold_l3;

// Output directory ($LOWPLY_MISS_DIR or default).
extern std::string output_dir;

// Tag identifying which configuration (master / htf / ...).
extern std::string config_tag;

// Per-thread instrumentation slot. Storage is owned by the global
// registry (heap-allocated, never freed) so that pointers stored at
// thread-exit remain readable in the atexit dumper. The TLS only holds
// a (stable) pointer to that registry slot.
PerThread& tls();

// Run calibration once at engine init.
void calibrate();

// Bin a (cycles) measurement into the residency histogram for ply.
inline void record(Bucket* hist, int ply, std::uint64_t cycles) {
    if (ply < 0 || ply >= LP_NPLY)
        return;
    Bucket& b = hist[ply];
    if (cycles < threshold_l1)
        b.l1.fetch_add(1, std::memory_order_relaxed);
    else if (cycles < threshold_l2)
        b.l2.fetch_add(1, std::memory_order_relaxed);
    else if (cycles < threshold_l3)
        b.l3.fetch_add(1, std::memory_order_relaxed);
    else
        b.dram.fetch_add(1, std::memory_order_relaxed);
}

// rdtscp wrapper.
inline std::uint64_t rdtscp_now() {
    unsigned aux;
    return __rdtscp(&aux);
}

// Hot-path hooks: take the address of the lowPlyHistory entry, time the
// access, return the value. Templated so we can use them on both the dense
// master layout and any wrapped variant.

template<typename T>
inline T timed_read(const T* addr, int ply) {
    auto& s = tls();
    if (++s.read_phase % READ_SAMPLE_STRIDE != 0)
    {
        return *addr;
    }
    _mm_lfence();
    std::uint64_t t0 = rdtscp_now();
    _mm_lfence();
    T v = *addr;
    _mm_lfence();
    std::uint64_t t1 = rdtscp_now();
    record(s.read_hist, ply, t1 - t0);
    return v;
}

template<typename T>
inline void timed_observe_write(T* addr, int ply) {
    auto& s = tls();
    if (++s.write_phase % WRITE_SAMPLE_STRIDE != 0)
    {
        return;
    }
    _mm_lfence();
    std::uint64_t t0 = rdtscp_now();
    _mm_lfence();
    volatile T v = *addr;
    (void) v;
    _mm_lfence();
    std::uint64_t t1 = rdtscp_now();
    record(s.write_hist, ply, t1 - t0);
}

}  // namespace Stockfish::LowPlyInstr

    // Hook macros (used at the read/write sites in movepick.cpp / search.cpp).
    #define LOWPLY_INSTR_READ(EXPR, PLY) Stockfish::LowPlyInstr::timed_read(&(EXPR), (PLY))

    #define LOWPLY_INSTR_OBSERVE_WRITE(EXPR, PLY) \
        Stockfish::LowPlyInstr::timed_observe_write(&(EXPR), (PLY))

#else  // !LOWPLY_MISS_INSTRUMENT

    #define LOWPLY_INSTR_READ(EXPR, PLY) (EXPR)
    #define LOWPLY_INSTR_OBSERVE_WRITE(EXPR, PLY) ((void) 0)

#endif
