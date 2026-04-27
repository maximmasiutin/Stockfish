// See lowply_miss_instr.h.

#include "lowply_miss_instr.h"

#if defined(LOWPLY_MISS_INSTRUMENT) && LOWPLY_MISS_INSTRUMENT


    #include <algorithm>
    #include <array>
    #include <chrono>
    #include <cstdio>
    #include <cstdlib>
    #include <cstring>
    #include <ctime>
    #include <mutex>
    #include <string>
    #include <thread>
    #include <vector>

    #if defined(_WIN32)
        #include <process.h>
        #define LP_GETPID() _getpid()
    #else
        #include <unistd.h>
        #define LP_GETPID() getpid()
    #endif

namespace Stockfish::LowPlyInstr {

std::uint64_t threshold_l1 = 0;
std::uint64_t threshold_l2 = 0;
std::uint64_t threshold_l3 = 0;
std::string   output_dir;
std::string   config_tag;

namespace {

constexpr std::size_t MAX_THREADS = 256;

struct Registry {
    std::mutex                          m;
    std::array<PerThread*, MAX_THREADS> threads{};
    std::size_t                         nthreads = 0;
    bool                                dumped   = false;
    std::array<std::uint64_t, 4>        calib{0, 0, 0, 0};  // raw medians
};

Registry& registry() {
    static Registry r;
    return r;
}

PerThread* allocate_slot() {
    auto&                       r = registry();
    std::lock_guard<std::mutex> g(r.m);
    if (r.nthreads >= MAX_THREADS)
        return nullptr;
    auto* s               = new PerThread();
    s->thread_idx         = static_cast<int>(r.nthreads);
    s->registered         = true;
    r.threads[r.nthreads] = s;
    ++r.nthreads;
    return s;
}

// ISO 8601 second-precision local + UTC timestamps.
void now_strings(std::string& local, std::string& utc) {
    using namespace std::chrono;
    auto        tnow = system_clock::now();
    std::time_t tt   = system_clock::to_time_t(tnow);
    char        buf[64];

    std::tm tm_local{};
    #if defined(_WIN32)
    localtime_s(&tm_local, &tt);
    #else
    localtime_r(&tt, &tm_local);
    #endif
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%S%z", &tm_local);
    local = buf;
    // Pretty-print offset as +HH:MM.
    if (local.size() >= 5)
    {
        auto n = local.size();
        local.insert(n - 2, ":");
    }

    std::tm tm_utc{};
    #if defined(_WIN32)
    gmtime_s(&tm_utc, &tt);
    #else
    gmtime_r(&tt, &tm_utc);
    #endif
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
    utc = buf;
    // Convert trailing 'Z' to '+00:00' for symmetry.
    if (!utc.empty() && utc.back() == 'Z')
    {
        utc.pop_back();
        utc += "+00:00";
    }
}

void announce(const char* phase) {
    std::string local, utc;
    now_strings(local, utc);
    std::fprintf(stderr, "[lowply-instr] %s tag=%s pid=%d local=%s utc=%s\n", phase,
                 config_tag.c_str(), static_cast<int>(LP_GETPID()), local.c_str(), utc.c_str());
    std::fflush(stderr);
}

// Calibrate cycle-to-residency thresholds.
//
// Procedure: allocate 64 MiB buffer. Issue clflushopt + sfence on a target
// line, then read it (cold -> DRAM); time the read. Read it again (now in L1)
// -> L1; time. Use the L1 median as base, then synthesize coarse L2 and L3
// thresholds at 4x and 12x the L1 median (Sapphire Rapids / Zen4 typical).
// This is approximate but stable enough to bin master vs htf relative miss
// rates.
void do_calibrate() {
    constexpr std::size_t BUF  = 64 * 1024 * 1024;
    constexpr int         REPS = 4096;
    auto                  raw  = std::make_unique<unsigned char[]>(BUF + 4096);
    unsigned char*        buf  = raw.get();

    // Touch all pages so the allocator does not page-fault inside the loop.
    for (std::size_t i = 0; i < BUF; i += 4096)
        buf[i] = static_cast<unsigned char>(i);

    std::vector<std::uint64_t> hot;
    std::vector<std::uint64_t> cold;
    hot.reserve(REPS);
    cold.reserve(REPS);

    for (int r = 0; r < REPS; ++r)
    {
        std::size_t off = (static_cast<std::size_t>(r) * 4099u) & (BUF - 1);
        // Cold (after flush)
        _mm_clflush(buf + off);
        _mm_mfence();
        _mm_lfence();
        std::uint64_t t0 = rdtscp_now();
        _mm_lfence();
        volatile unsigned char vread = buf[off];
        _mm_lfence();
        std::uint64_t t1 = rdtscp_now();
        cold.push_back(t1 - t0);
        // Hot (immediately after)
        _mm_lfence();
        std::uint64_t h0 = rdtscp_now();
        _mm_lfence();
        volatile unsigned char vhot = buf[off];
        _mm_lfence();
        std::uint64_t h1 = rdtscp_now();
        hot.push_back(h1 - h0);
        (void) vread;
        (void) vhot;
    }

    auto median = [](std::vector<std::uint64_t>& v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    std::uint64_t l1_med   = median(hot);
    std::uint64_t dram_med = median(cold);

    // Heuristic split: <2.5x L1 -> L1; <(L1+DRAM)/4 -> L2; <(L1+DRAM)/2 -> L3.
    threshold_l1 = std::max<std::uint64_t>(l1_med + 2, l1_med * 5 / 2);
    threshold_l2 = std::max<std::uint64_t>(threshold_l1 + 2, (l1_med * 7 + dram_med) / 8);
    threshold_l3 = std::max<std::uint64_t>(threshold_l2 + 2, (l1_med + dram_med) / 2);

    auto& r = registry();
    {
        std::lock_guard<std::mutex> g(r.m);
        r.calib[0] = l1_med;
        r.calib[1] = threshold_l1;
        r.calib[2] = threshold_l2;
        r.calib[3] = threshold_l3;
    }

    std::fprintf(stderr,
                 "[lowply-instr] calibration tag=%s l1_med=%llu dram_med=%llu "
                 "thresholds={l1<%llu, l2<%llu, l3<%llu, else dram}\n",
                 config_tag.c_str(), (unsigned long long) l1_med, (unsigned long long) dram_med,
                 (unsigned long long) threshold_l1, (unsigned long long) threshold_l2,
                 (unsigned long long) threshold_l3);
    std::fflush(stderr);
}

void emit_json() {
    auto&                       r = registry();
    std::lock_guard<std::mutex> g(r.m);
    if (r.dumped)
        return;
    r.dumped = true;

    std::string dir = output_dir.empty() ? std::string("scratchpad/lowply-miss-instr") : output_dir;
    // mkdir best-effort.
    #if defined(_WIN32)
    std::system(("if not exist \"" + dir + "\" mkdir \"" + dir + "\"").c_str());
    #else
    std::system(("mkdir -p '" + dir + "'").c_str());
    #endif

    std::string local, utc;
    now_strings(local, utc);

    char fname[512];
    std::snprintf(fname, sizeof fname, "%s/%s-%d.json", dir.c_str(),
                  config_tag.empty() ? "run" : config_tag.c_str(), static_cast<int>(LP_GETPID()));

    std::FILE* f = std::fopen(fname, "wb");
    if (!f)
    {
        std::fprintf(stderr, "[lowply-instr] cannot open %s\n", fname);
        return;
    }

    std::fprintf(f, "{\n");
    std::fprintf(f, "  \"config_tag\": \"%s\",\n", config_tag.c_str());
    std::fprintf(f, "  \"pid\": %d,\n", static_cast<int>(LP_GETPID()));
    std::fprintf(f, "  \"dump_local\": \"%s\",\n", local.c_str());
    std::fprintf(f, "  \"dump_utc\": \"%s\",\n", utc.c_str());
    std::fprintf(f, "  \"calibration\": {\n");
    std::fprintf(f, "    \"l1_median_cycles\": %llu,\n", (unsigned long long) r.calib[0]);
    std::fprintf(f, "    \"threshold_l1\": %llu,\n", (unsigned long long) r.calib[1]);
    std::fprintf(f, "    \"threshold_l2\": %llu,\n", (unsigned long long) r.calib[2]);
    std::fprintf(f, "    \"threshold_l3\": %llu\n", (unsigned long long) r.calib[3]);
    std::fprintf(f, "  },\n");
    std::fprintf(f, "  \"sampling\": {\"read\": %u, \"write\": %u},\n", READ_SAMPLE_STRIDE,
                 WRITE_SAMPLE_STRIDE);
    std::fprintf(f, "  \"per_thread\": [\n");
    for (std::size_t i = 0; i < r.nthreads; ++i)
    {
        PerThread* s = r.threads[i];
        if (!s)
            continue;
        std::fprintf(f, "    %s{\n      \"thread_idx\": %d,\n      \"reads\": [\n",
                     i == 0 ? "" : ",", s->thread_idx);
        for (int p = 0; p < LP_NPLY; ++p)
        {
            std::fprintf(f,
                         "        %s{\"ply\": %d, \"l1\": %llu, \"l2\": %llu, "
                         "\"l3\": %llu, \"dram\": %llu}\n",
                         p == 0 ? "" : ",", p, (unsigned long long) s->read_hist[p].l1.load(),
                         (unsigned long long) s->read_hist[p].l2.load(),
                         (unsigned long long) s->read_hist[p].l3.load(),
                         (unsigned long long) s->read_hist[p].dram.load());
        }
        std::fprintf(f, "      ],\n      \"writes\": [\n");
        for (int p = 0; p < LP_NPLY; ++p)
        {
            std::fprintf(f,
                         "        %s{\"ply\": %d, \"l1\": %llu, \"l2\": %llu, "
                         "\"l3\": %llu, \"dram\": %llu}\n",
                         p == 0 ? "" : ",", p, (unsigned long long) s->write_hist[p].l1.load(),
                         (unsigned long long) s->write_hist[p].l2.load(),
                         (unsigned long long) s->write_hist[p].l3.load(),
                         (unsigned long long) s->write_hist[p].dram.load());
        }
        std::fprintf(f, "      ]\n    }\n");
    }
    std::fprintf(f, "  ]\n}\n");
    std::fclose(f);

    std::fprintf(stderr, "[lowply-instr] wrote %s local=%s utc=%s\n", fname, local.c_str(),
                 utc.c_str());
    std::fflush(stderr);
}

struct Init {
    Init() {
        const char* env_dir = std::getenv("LOWPLY_MISS_DIR");
        if (env_dir && *env_dir)
            output_dir = env_dir;
        const char* env_tag = std::getenv("LOWPLY_MISS_TAG");
        if (env_tag && *env_tag)
            config_tag = env_tag;
        else
            config_tag = "unknown";
        announce("init");
        do_calibrate();
        std::atexit(&emit_json);
    }
};

Init g_init;

}  // namespace

PerThread& tls() {
    static thread_local PerThread* slot = nullptr;
    if (!slot)
    {
        slot = allocate_slot();
        if (!slot)
        {
            // MAX_THREADS exceeded: return a sink slot whose counters are
            // discarded (still safe to write to).
            static PerThread sink;
            slot = &sink;
        }
    }
    return *slot;
}

void calibrate() { do_calibrate(); }

}  // namespace Stockfish::LowPlyInstr

#endif  // LOWPLY_MISS_INSTRUMENT
