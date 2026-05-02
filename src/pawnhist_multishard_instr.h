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

#ifndef PAWNHIST_MULTISHARD_INSTR_H_INCLUDED
#define PAWNHIST_MULTISHARD_INSTR_H_INCLUDED

#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "bitboard.h"
#include "position.h"
#include "types.h"

#ifdef _WIN32
    #include <process.h>
    #define PAWNHIST_MULTISHARD_PID _getpid
#else
    #include <unistd.h>
    #define PAWNHIST_MULTISHARD_PID getpid
#endif

// Multi-design replay instrumentation for pawnHistory shard candidates.
//
// At every pawnHistory access (5 hook sites), record the bucket index
// FIVE candidate addressing functions would have produced, into FIVE
// separate counter arrays. The actual table access in master code uses
// master's hash unchanged, so search behavior is preserved (bench-byte-
// equal at d13 = 2723949 must be verified after build).
//
// Designs:
//   master  : pos.pawn_key() & mask                          (baseline)
//   huff    : Huffman variable-length pawn-count prefix     (Design A)
//   4bit    : 4-bit fixed pawn-count prefix                  (Design B, canonical v2)
//   xor     : pawn_key XOR phase_perm[p], no shard           (Design C, H3 only)
//   tier    : 2-bit chess-tier prefix {0..1, 2..7, 8..12, 13..16}  (Design D)
//
// Counter array layout: [SLOTS][RW] of atomic uint64. Phase dimension
// dropped to fit memory. Total BSS = 5 * 4 MiB = 20 MiB.
//
// SLOTS = 262144 covers the maximum threadCount we want to instrument
// (32 worker threads x 8192 base-size, after next_power_of_two rounding).

namespace Stockfish::PawnHistMultishardInstr {

constexpr std::size_t SLOTS    = 262144;
constexpr std::size_t RW_COUNT = 2;  // [0]=read, [1]=write

static_assert((SLOTS & (SLOTS - 1)) == 0, "SLOTS must be a power of two");

constexpr std::size_t TOTAL_U64S_PER_DESIGN = SLOTS * RW_COUNT;

// Five separate counter arrays, one per design. Inline storage so the
// atexit dump can find them without a separate .cpp file.

inline std::array<std::array<std::atomic<std::uint64_t>, RW_COUNT>, SLOTS> counters_master{};
inline std::array<std::array<std::atomic<std::uint64_t>, RW_COUNT>, SLOTS> counters_huff{};
inline std::array<std::array<std::atomic<std::uint64_t>, RW_COUNT>, SLOTS> counters_4bit{};
inline std::array<std::array<std::atomic<std::uint64_t>, RW_COUNT>, SLOTS> counters_xor{};
inline std::array<std::array<std::atomic<std::uint64_t>, RW_COUNT>, SLOTS> counters_tier{};

// -----------------------------------------------------------------
// Design A: Huffman variable-length codes
// -----------------------------------------------------------------
// Codes derived offline from smp-ltc-8t-c2 cell frequency table.
// MSB-aligned in the high bits of the bucket address so phase regions
// remain spatially contiguous (preserves H1 mechanism even with
// variable-length prefixes).

struct HuffmanCode {
    std::uint8_t code;
    std::uint8_t code_bits;
};

inline constexpr HuffmanCode HUFF[17] = {
    /*  0 */ {0b1111110, 7},
    /*  1 */ {0b111110, 6},
    /*  2 */ {0b110010, 6},
    /*  3 */ {0b110011, 6},
    /*  4 */ {0b11000, 5},
    /*  5 */ {0b11010, 5},
    /*  6 */ {0b1110, 4},
    /*  7 */ {0b0001, 4},
    /*  8 */ {0b010, 3},
    /*  9 */ {0b011, 3},
    /* 10 */ {0b0010, 4},
    /* 11 */ {0b1011, 4},
    /* 12 */ {0b1010, 4},
    /* 13 */ {0b11011, 5},
    /* 14 */ {0b100, 3},
    /* 15 */ {0b1111111, 7},
    /* 16 */ {0b000, 3},
};

inline std::size_t slot_huff(int p, Key pk, int total_bits) {
    const HuffmanCode hc         = HUFF[p];
    const int         inner_bits = total_bits - int(hc.code_bits);
    const std::size_t inner_mask = (std::size_t(1) << inner_bits) - 1;
    return (std::size_t(hc.code) << inner_bits) | (std::size_t(pk) & inner_mask);
}

// -----------------------------------------------------------------
// Design B: 4-bit fixed pawn-count shard (canonical v2)
// -----------------------------------------------------------------

inline std::size_t slot_4bit(int p, Key pk, int total_bits) {
    const std::size_t shard      = std::size_t(p) - (std::size_t(p) >> 4);
    const int         inner_bits = total_bits - 4;
    const std::size_t inner_mask = (std::size_t(1) << inner_bits) - 1;
    return (shard << inner_bits) | (std::size_t(pk) & inner_mask);
}

// -----------------------------------------------------------------
// Design C: Phase-augmented XOR hash (uniform, no shard)
// -----------------------------------------------------------------

inline constexpr std::uint64_t PHASE_PERM[17] = {
    0xa1b2c3d4e5f60718ULL, 0x923c0879ab12fed4ULL, 0xc7d3f1a290b6e845ULL, 0x5e8b4a26d0f17392ULL,
    0xf2a91c5d8e034b67ULL, 0x3d6e2b91f7c8a504ULL, 0x0b8d4f2a6e519c73ULL, 0xe4f60d83a572c19bULL,
    0x71c5938b2ad60ef4ULL, 0x9628fa410d3eb75cULL, 0xb35d8e0c612af987ULL, 0x4a92ec1f78b305d6ULL,
    0xd80f37e95b46a213ULL, 0x86f1d4a2c507b39eULL, 0x29b7e0c5fa813d6bULL, 0x5cd09a3f1e87b264ULL,
    0xa30b67e94d5fc218ULL,
};

inline std::size_t slot_xor(int p, Key pk, std::size_t mask) {
    return std::size_t(pk ^ PHASE_PERM[p]) & mask;
}

// -----------------------------------------------------------------
// Design D: 2-bit chess-tier shard
// -----------------------------------------------------------------

inline std::size_t slot_tier(int p, Key pk, int total_bits) {
    const std::size_t tier =
      std::size_t(p > 1) + std::size_t(p > 7) + std::size_t(p > 12);
    const int         inner_bits = total_bits - 2;
    const std::size_t inner_mask = (std::size_t(1) << inner_bits) - 1;
    return (tier << inner_bits) | (std::size_t(pk) & inner_mask);
}

// -----------------------------------------------------------------
// Hook: record one access on all five designs
// -----------------------------------------------------------------
//
// total_bits = log2(actual pawnHistory size) at runtime. Caller computes
// from the SharedHistories::pawnHistory size and passes in. mask =
// (1 << total_bits) - 1 is the master mask.

inline void record(Key pk, int p, std::size_t master_slot, int total_bits, std::size_t op) {
    assert(p >= 0 && p <= 16);
    assert(master_slot < SLOTS);
    assert(op < RW_COUNT);

    const std::size_t mask     = (std::size_t(1) << total_bits) - 1;
    const std::size_t s_huff   = slot_huff(p, pk, total_bits);
    const std::size_t s_4bit   = slot_4bit(p, pk, total_bits);
    const std::size_t s_xor    = slot_xor(p, pk, mask);
    const std::size_t s_tier   = slot_tier(p, pk, total_bits);

    counters_master[master_slot][op].fetch_add(1, std::memory_order_relaxed);
    counters_huff  [s_huff     ][op].fetch_add(1, std::memory_order_relaxed);
    counters_4bit  [s_4bit     ][op].fetch_add(1, std::memory_order_relaxed);
    counters_xor   [s_xor      ][op].fetch_add(1, std::memory_order_relaxed);
    counters_tier  [s_tier     ][op].fetch_add(1, std::memory_order_relaxed);
}

inline void record_read(const Position& pos, std::size_t master_slot, int total_bits) {
    record(pos.pawn_key(), popcount(pos.pieces(PAWN)), master_slot, total_bits, 0);
}

inline void record_write(const Position& pos, std::size_t master_slot, int total_bits) {
    record(pos.pawn_key(), popcount(pos.pieces(PAWN)), master_slot, total_bits, 1);
}

// -----------------------------------------------------------------
// Dump on process exit
// -----------------------------------------------------------------
//
// Writes 5 binary files (one per design) named:
//   pawnhist-multidesign-{master,huff,4bit,xor,tier}-{pid}.bin
//
// Each file contains SLOTS * RW_COUNT uint64 values, layout
// [slot=0,rw=0] [slot=0,rw=1] [slot=1,rw=0] ... [slot=SLOTS-1,rw=1].

inline void dump_one(const char* tag,
                     int         pid,
                     const std::array<std::array<std::atomic<std::uint64_t>, RW_COUNT>, SLOTS>& arr) {
    char fname[256];
    std::snprintf(fname, sizeof(fname),
                  "pawnhist-multidesign-%s-%d.bin", tag, pid);
    std::FILE* f = std::fopen(fname, "wb");
    if (!f)
        return;

    static std::uint64_t buf[TOTAL_U64S_PER_DESIGN];
    std::size_t          i = 0;
    for (std::size_t s = 0; s < SLOTS; ++s)
        for (std::size_t rw = 0; rw < RW_COUNT; ++rw)
            buf[i++] = arr[s][rw].load(std::memory_order_relaxed);

    std::fwrite(buf, sizeof(std::uint64_t), TOTAL_U64S_PER_DESIGN, f);
    std::fclose(f);
}

inline void dump_at_exit() {
    const int pid = static_cast<int>(PAWNHIST_MULTISHARD_PID());
    dump_one("master", pid, counters_master);
    dump_one("huff",   pid, counters_huff);
    dump_one("4bit",   pid, counters_4bit);
    dump_one("xor",    pid, counters_xor);
    dump_one("tier",   pid, counters_tier);
}

struct Registrar {
    Registrar() { std::atexit(&dump_at_exit); }
};

inline Registrar registrar{};

}  // namespace Stockfish::PawnHistMultishardInstr

#endif  // PAWNHIST_MULTISHARD_INSTR_H_INCLUDED
