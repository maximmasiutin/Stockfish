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

inline std::atomic<std::uint64_t> cold_count{0};   // r >= 0x4000 (non-NORMAL)
inline std::atomic<std::uint64_t> pawn_count{0};   // pawn-push tier (tier 0/1)
inline std::atomic<std::uint64_t> king_count{0};   // king-class tier 2 (cheb=1)
inline std::atomic<std::uint64_t> tier3_count{0};  // tier 3 master fallback

inline void record_cold() { cold_count.fetch_add(1, std::memory_order_relaxed); }
inline void record_pawn() { pawn_count.fetch_add(1, std::memory_order_relaxed); }
inline void record_king() { king_count.fetch_add(1, std::memory_order_relaxed); }
inline void record_tier3() { tier3_count.fetch_add(1, std::memory_order_relaxed); }

inline void dump_at_exit() {
    char fname[256];
    std::snprintf(fname, sizeof(fname), "mainhist-freq-cluster-%d.txt",
                  static_cast<int>(MAINHIST_FREQ_INSTR_PID()));
    std::FILE* f = std::fopen(fname, "w");
    if (!f)
        return;

    const std::uint64_t cold  = cold_count.load(std::memory_order_relaxed);
    const std::uint64_t pawn  = pawn_count.load(std::memory_order_relaxed);
    const std::uint64_t king  = king_count.load(std::memory_order_relaxed);
    const std::uint64_t tier3 = tier3_count.load(std::memory_order_relaxed);
    const std::uint64_t total = cold + pawn + king + tier3;

    auto pct = [&](std::uint64_t n) { return total ? 100.0 * double(n) / double(total) : 0.0; };

    std::fprintf(f, "cold=%llu\n", static_cast<unsigned long long>(cold));
    std::fprintf(f, "pawn=%llu\n", static_cast<unsigned long long>(pawn));
    std::fprintf(f, "king=%llu\n", static_cast<unsigned long long>(king));
    std::fprintf(f, "tier3=%llu\n", static_cast<unsigned long long>(tier3));
    std::fprintf(f, "total=%llu\n", static_cast<unsigned long long>(total));
    std::fprintf(f, "pct_cold=%.4f\n", pct(cold));
    std::fprintf(f, "pct_pawn=%.4f\n", pct(pawn));
    std::fprintf(f, "pct_king=%.4f\n", pct(king));
    std::fprintf(f, "pct_tier3=%.4f\n", pct(tier3));
    std::fclose(f);
}

struct Registrar {
    Registrar() { std::atexit(&dump_at_exit); }
};

inline Registrar registrar{};

}  // namespace Stockfish::MainHistFreqInstr

#endif  // MAINHIST_FREQ_INSTR_H_INCLUDED
