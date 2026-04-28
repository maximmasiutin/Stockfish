/*
  Sort-length instrumentation, throwaway. Records per-call N at the
  six hot sort sites: 3 partial_insertion_sort sites in movepick.cpp
  (S0=captures, S1=quiets-with-limit, S2=evasions) and 3
  std::stable_sort sites in search.cpp (S3=PV slice, S4=prefix,
  S5=final). On exit, if env SORT_INSTR_OUTDIR is set, dumps a per-pid
  text file in that directory.

  Counters are global atomics with cache-line alignment per site to
  avoid false sharing. SMP runs see contention but distribution
  values remain correct; expected NPS hit ~1-5% under SMP.
*/

#ifndef SORT_INSTR_H_INCLUDED
#define SORT_INSTR_H_INCLUDED

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <string>

#ifdef _WIN32
    #include <process.h>
    #define SORT_INSTR_GETPID() ((long) _getpid())
#else
    #include <unistd.h>
    #define SORT_INSTR_GETPID() ((long) getpid())
#endif

namespace Stockfish::SortInstr {

constexpr int NUM_SITES = 6;
constexpr int NUM_BINS  = 10;

struct alignas(64) SiteCounters {
    std::atomic<uint64_t> calls{0};
    std::atomic<uint64_t> sum_n{0};
    std::atomic<uint64_t> sum_n2{0};
    std::atomic<uint64_t> max_n{0};
    std::atomic<uint64_t> hist[NUM_BINS]{};
    // Only meaningful for site 1 (QUIET_INIT partial-sort above limit).
    std::atomic<uint64_t> calls_above{0};
    std::atomic<uint64_t> sum_n_above{0};
    std::atomic<uint64_t> hist_above[NUM_BINS]{};
};

inline SiteCounters g_sites[NUM_SITES];

inline int bin_for_n(int n) {
    if (n <= 0)
        return 0;
    if (n == 1)
        return 1;
    if (n <= 3)
        return 2;
    if (n <= 7)
        return 3;
    if (n <= 15)
        return 4;
    if (n <= 31)
        return 5;
    if (n <= 63)
        return 6;
    if (n <= 127)
        return 7;
    if (n <= 255)
        return 8;
    return 9;
}

inline void record(int site, int n) {
    SiteCounters& s = g_sites[site];
    s.calls.fetch_add(1, std::memory_order_relaxed);
    s.sum_n.fetch_add((uint64_t) n, std::memory_order_relaxed);
    s.sum_n2.fetch_add((uint64_t) n * (uint64_t) n, std::memory_order_relaxed);
    uint64_t prev = s.max_n.load(std::memory_order_relaxed);
    while (prev < (uint64_t) n
           && !s.max_n.compare_exchange_weak(prev, (uint64_t) n, std::memory_order_relaxed))
        ;
    s.hist[bin_for_n(n)].fetch_add(1, std::memory_order_relaxed);
}

inline void record_above(int site, int n_above) {
    SiteCounters& s = g_sites[site];
    s.calls_above.fetch_add(1, std::memory_order_relaxed);
    s.sum_n_above.fetch_add((uint64_t) n_above, std::memory_order_relaxed);
    s.hist_above[bin_for_n(n_above)].fetch_add(1, std::memory_order_relaxed);
}

inline void dump_to_file(const std::string& path) {
    std::ofstream f(path);
    if (!f)
        return;
    f << "SORT_INSTR begin pid=" << SORT_INSTR_GETPID() << "\n";
    for (int site = 0; site < NUM_SITES; ++site)
    {
        SiteCounters& s = g_sites[site];
        f << "SORT_SITE " << site << "\n";
        f << "calls " << s.calls.load(std::memory_order_relaxed) << "\n";
        f << "sum_n " << s.sum_n.load(std::memory_order_relaxed) << "\n";
        f << "sum_n2 " << s.sum_n2.load(std::memory_order_relaxed) << "\n";
        f << "max_n " << s.max_n.load(std::memory_order_relaxed) << "\n";
        for (int b = 0; b < NUM_BINS; ++b)
            f << "hist " << b << " " << s.hist[b].load(std::memory_order_relaxed) << "\n";
        if (site == 1)
        {
            f << "calls_above " << s.calls_above.load(std::memory_order_relaxed) << "\n";
            f << "sum_n_above " << s.sum_n_above.load(std::memory_order_relaxed) << "\n";
            for (int b = 0; b < NUM_BINS; ++b)
                f << "hist_above " << b << " " << s.hist_above[b].load(std::memory_order_relaxed)
                  << "\n";
        }
    }
    f << "SORT_INSTR end\n";
}

struct DumpAtExit {
    ~DumpAtExit() {
        const char* outdir = std::getenv("SORT_INSTR_OUTDIR");
        if (!outdir)
            return;
        std::string path = outdir;
        path += "/";
        path += std::to_string(SORT_INSTR_GETPID());
        path += ".txt";
        dump_to_file(path);
    }
};

inline DumpAtExit g_dump_at_exit;

}  // namespace Stockfish::SortInstr

#endif  // SORT_INSTR_H_INCLUDED
