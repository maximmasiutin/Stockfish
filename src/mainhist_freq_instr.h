/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef MAINHIST_FREQ_INSTR_H_INCLUDED
#define MAINHIST_FREQ_INSTR_H_INCLUDED

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifdef _WIN32
    #include <process.h>
    #define MAINHIST_FREQ_INSTR_PID _getpid
#else
    #include <unistd.h>
    #define MAINHIST_FREQ_INSTR_PID getpid
#endif

namespace Stockfish::MainHistFreqInstr {

inline std::atomic<std::uint64_t> total_calls{0};
inline std::atomic<std::uint64_t> cold_calls{0};

inline void record_call() { total_calls.fetch_add(1, std::memory_order_relaxed); }
inline void record_cold() { cold_calls.fetch_add(1, std::memory_order_relaxed); }

inline void dump_at_exit() {
    char fname[256];
    std::snprintf(fname, sizeof(fname), "mainhist-freq-counters-%d.txt",
                  static_cast<int>(MAINHIST_FREQ_INSTR_PID()));
    std::FILE* f = std::fopen(fname, "w");
    if (!f)
        return;

    const std::uint64_t total    = total_calls.load(std::memory_order_relaxed);
    const std::uint64_t cold     = cold_calls.load(std::memory_order_relaxed);
    const std::uint64_t hot      = total - cold;
    const double        pct_cold = total ? 100.0 * double(cold) / double(total) : 0.0;
    const double        pct_hot  = total ? 100.0 * double(hot) / double(total) : 0.0;

    std::fprintf(f, "total_calls=%llu\n", static_cast<unsigned long long>(total));
    std::fprintf(f, "cold_calls=%llu\n", static_cast<unsigned long long>(cold));
    std::fprintf(f, "hot_calls=%llu\n", static_cast<unsigned long long>(hot));
    std::fprintf(f, "pct_cold=%.4f\n", pct_cold);
    std::fprintf(f, "pct_hot=%.4f\n", pct_hot);
    std::fclose(f);
}

struct Registrar {
    Registrar() { std::atexit(&dump_at_exit); }
};

inline Registrar registrar{};

}  // namespace Stockfish::MainHistFreqInstr

#endif  // MAINHIST_FREQ_INSTR_H_INCLUDED
