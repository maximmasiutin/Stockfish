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

#ifndef LOWPLY_INSTR_H_INCLUDED
#define LOWPLY_INSTR_H_INCLUDED

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifdef _WIN32
    #include <process.h>
    #define LOWPLY_INSTR_PID _getpid
#else
    #include <unistd.h>
    #define LOWPLY_INSTR_PID getpid
#endif

#include "history.h"

namespace Stockfish::LowPlyInstr {

constexpr std::size_t RAW_COUNT  = 65536;
constexpr std::size_t RW_COUNT   = 2;  // [0] = reads, [1] = writes
constexpr std::size_t TOTAL_U64S = LOW_PLY_HISTORY_SIZE * RAW_COUNT * RW_COUNT;

inline std::array<std::array<std::array<std::atomic<std::uint64_t>, RW_COUNT>, RAW_COUNT>,
                  LOW_PLY_HISTORY_SIZE>
  counters{};

inline void record_read(int ply, std::uint16_t raw) {
    counters[ply][raw][0].fetch_add(1, std::memory_order_relaxed);
}

inline void record_write(int ply, std::uint16_t raw) {
    counters[ply][raw][1].fetch_add(1, std::memory_order_relaxed);
}

inline void dump_at_exit() {
    char fname[256];
    std::snprintf(fname, sizeof(fname), "lowply-counters-%d.bin",
                  static_cast<int>(LOWPLY_INSTR_PID()));
    std::FILE* f = std::fopen(fname, "wb");
    if (!f)
        return;

    static thread_local std::uint64_t buf[TOTAL_U64S];
    std::size_t                       i = 0;
    for (int p = 0; p < LOW_PLY_HISTORY_SIZE; ++p)
        for (std::size_t r = 0; r < RAW_COUNT; ++r)
            for (std::size_t rw = 0; rw < RW_COUNT; ++rw)
                buf[i++] = counters[p][r][rw].load(std::memory_order_relaxed);

    std::fwrite(buf, sizeof(std::uint64_t), TOTAL_U64S, f);
    std::fclose(f);
}

struct Registrar {
    Registrar() { std::atexit(&dump_at_exit); }
};

inline Registrar registrar{};

}  // namespace Stockfish::LowPlyInstr

#endif  // #ifndef LOWPLY_INSTR_H_INCLUDED
