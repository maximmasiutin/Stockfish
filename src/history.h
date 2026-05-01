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

#ifndef HISTORY_H_INCLUDED
#define HISTORY_H_INCLUDED

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <type_traits>  // IWYU pragma: keep

#include "memory.h"
#include "misc.h"
#include "position.h"

namespace Stockfish {

constexpr int PAWN_HISTORY_BASE_SIZE   = 8192;  // has to be a power of 2
constexpr int UINT_16_HISTORY_SIZE     = std::numeric_limits<uint16_t>::max() + 1;
constexpr int CORRHIST_BASE_SIZE       = UINT_16_HISTORY_SIZE;
constexpr int CORRECTION_HISTORY_LIMIT = 1024;
constexpr int LOW_PLY_HISTORY_SIZE     = 5;

static_assert((PAWN_HISTORY_BASE_SIZE & (PAWN_HISTORY_BASE_SIZE - 1)) == 0,
              "PAWN_HISTORY_BASE_SIZE has to be a power of 2");

static_assert((CORRHIST_BASE_SIZE & (CORRHIST_BASE_SIZE - 1)) == 0,
              "CORRHIST_BASE_SIZE has to be a power of 2");

// StatsEntry is the container of various numerical statistics. We use a class
// instead of a naked value to directly call history update operator<<() on
// the entry. The first template parameter T is the base type of the array,
// and the second template parameter D limits the range of updates in [-D, D]
// when we update values with the << operator
template<typename T, int D, bool Atomic = false>
struct StatsEntry {
    static_assert(std::is_arithmetic_v<T>, "Not an arithmetic type");

   private:
    std::conditional_t<Atomic, std::atomic<T>, T> entry;

   public:
    void operator=(const T& v) {
        if constexpr (Atomic)
            entry.store(v, std::memory_order_relaxed);
        else
            entry = v;
    }

    operator T() const {
        if constexpr (Atomic)
            return entry.load(std::memory_order_relaxed);
        else
            return entry;
    }

    void operator<<(int bonus) {
        // Make sure that bonus is in range [-D, D]
        int clampedBonus = std::clamp(bonus, -D, D);
        T   val          = *this;
        *this            = val + clampedBonus - val * std::abs(clampedBonus) / D;

        assert(std::abs(T(*this)) <= D);
    }
};

enum StatsType {
    NoCaptures,
    Captures
};

template<typename T, int D, std::size_t... Sizes>
using Stats = MultiArray<StatsEntry<T, D>, Sizes...>;

template<typename T, int D, std::size_t... Sizes>
using AtomicStats = MultiArray<StatsEntry<T, D, true>, Sizes...>;

// DynStats is a dynamically sized array of Stats, used for thread-shared histories
// which should scale with the total number of threads. The SizeMultiplier gives
// the per-thread allocation count of T.
template<typename T, int SizeMultiplier>
struct DynStats {
    explicit DynStats(size_t s) {
        size = s * SizeMultiplier;
        data = make_unique_large_page<T[]>(size);
    }
    // Sets all values in the range to 0
    void clear_range(int value, size_t threadIdx, size_t numaTotal) {
        size_t start = uint64_t(threadIdx) * size / numaTotal;
        assert(start < size);
        size_t end = threadIdx + 1 == numaTotal ? size : uint64_t(threadIdx + 1) * size / numaTotal;

        while (start < end)
            data[start++].fill(value);
    }
    size_t get_size() const { return size; }
    T&     operator[](size_t index) {
        assert(index < size);
        return data.get()[index];
    }
    const T& operator[](size_t index) const {
        assert(index < size);
        return data.get()[index];
    }

   private:
    size_t            size;
    LargePagePtr<T[]> data;
};

// ButterflyHistory: frequency-aware indexed mainHistory with split-mask pawn
// slot computation. Init pawn pushes (first-row source: fr in {1,6}) at
// [0,32) and continuation pawn pushes (fr in {2..5}) at [32,96) are computed
// by separate slot expressions selected via mask blend rather than a unified
// variable-shift chain. King-class chebyshev=1 (any rank) at [96,608); NORMAL
// rest at [608,4704); PROMOTION/CASTLING/EN_PASSANT in cold tail. Hot path is
// fully branchless inside NORMAL: a single conditional branch (the cold gate)
// routes non-NORMAL types to a small cold tail.
//
// Slot layout (per color slice; total 4930 slots):
//   [0, 32)        Tier 0  NORMAL initial-rank pawn pushes
//   [32, 96)       Tier 1  NORMAL continuation single-pushes
//   [96, 608)      Tier 2  NORMAL chebyshev=1 from any rank (king-class)
//   [608, 4704)    Tier 3  NORMAL fallback, (from << 6) | to ordering
//   [4704, 4768)   Tier 4  PROMOTION
//   [4768, 4896)   Tier 5  CASTLING (DFRC-safe)
//   [4896, 4928)   Tier 6  EN_PASSANT
//   [4928, 4930)   unused (former sentinels; never reached at call sites)
constexpr int MAINHIST_FREQ_SIZE = 4930;

sf_noinline std::size_t main_hist_freq_index_special(std::uint32_t r);

sf_always_inline inline std::size_t main_hist_freq_index(Move m) {
    const std::uint32_t r = std::uint32_t(m.raw());
    if (r >= 0x4000u)
        return main_hist_freq_index_special(r);

    const std::uint32_t from = (r >> 6) & 0x3Fu;
    const std::uint32_t to   = r & 0x3Fu;
    const std::uint32_t fr   = from >> 3;
    const std::uint32_t ff   = from & 7u;
    const std::uint32_t tr   = to >> 3;
    const std::uint32_t tf   = to & 7u;

    const std::uint32_t tier3 = 608u + (r & 0xFFFu);

    constexpr std::uint64_t PAWN_MASK = 0x00305028140a0c00ULL;
    const std::uint32_t     same_file = std::uint32_t(((from ^ to) & 7u) == 0u);
    const std::uint32_t pawn_hit  = same_file & std::uint32_t((PAWN_MASK >> (fr * 8u + tr)) & 1u);
    const std::uint32_t pawn_mask = std::uint32_t(0u) - pawn_hit;

    // pawn_color: 0=white (tr>=fr, moves up), 1=black (tr<fr, moves down).
    // Move-geometry inferred (matches WHITE/BLACK semantics for legal pawn pushes).
    const std::uint32_t pawn_color = std::uint32_t(tr < fr);
    // is_init: 1 iff fr in {1, 6} (pawn home rank). 0x42 = bits 1 and 6 set.
    const std::uint32_t is_init   = std::uint32_t((0x42u >> fr) & 1u);
    const std::uint32_t init_mask = std::uint32_t(0u) - is_init;

    // is_double: 1 iff |tr - fr| == 2 (pawn double push). Equivalent to
    // ((tr ^ fr) & 3) == 2 on the valid (fr, tr) tuples in PAWN_MASK.
    const std::uint32_t is_double = std::uint32_t(((tr ^ fr) & 3u) == 2u);
    const std::uint32_t init_slot = (pawn_color << 4u) + (ff << 1u) + is_double;
    const std::uint32_t cont_slot = 32u + (pawn_color << 5u) + (ff << 2u) + (fr - 2u);
    const std::uint32_t pawn_slot = (init_slot & init_mask) | (cont_slot & ~init_mask);

    // King-class chebyshev=1 (any rank).
    const int           fdiff     = int(tf) - int(ff);
    const int           rdiff     = int(tr) - int(fr);
    const std::uint32_t fdiff_ok  = std::uint32_t(std::uint32_t(fdiff + 1) <= 2u);
    const std::uint32_t rdiff_ok  = std::uint32_t(std::uint32_t(rdiff + 1) <= 2u);
    const std::uint32_t not_null  = std::uint32_t((fdiff != 0) | (rdiff != 0));
    const std::uint32_t king_hit  = fdiff_ok & rdiff_ok & not_null;
    const std::uint32_t king_mask = std::uint32_t(0u) - king_hit;
    const std::uint32_t pos       = std::uint32_t(fdiff + 1) * 3u + std::uint32_t(rdiff + 1);
    const std::uint32_t dest      = pos - std::uint32_t(pos > 3u);
    const std::uint32_t king_slot = 96u + fr * 64u + ff * 8u + dest;

    std::uint32_t s = (king_slot & king_mask) | (tier3 & ~king_mask);
    s               = (pawn_slot & pawn_mask) | (s & ~pawn_mask);
    return std::size_t(s);
}

// Permute-only variant: freq-aware permutation function with the full 64K
// table size. Slots [0, MAINHIST_FREQ_SIZE) are touched as in the freq-aware
// branch; slots [MAINHIST_FREQ_SIZE, UINT_16_HISTORY_SIZE) are allocated but
// never accessed. Tests permutation WITHOUT compaction.
using ButterflyHistory = Stats<std::int16_t, 7183, COLOR_NB, UINT_16_HISTORY_SIZE>;

// LowPlyHistory is addressed by ply and move's from and to squares, used
// to improve move ordering near the root
using LowPlyHistory = Stats<std::int16_t, 7183, LOW_PLY_HISTORY_SIZE, UINT_16_HISTORY_SIZE>;

// CapturePieceToHistory is addressed by a move's [piece][to][captured piece type]
using CapturePieceToHistory = Stats<std::int16_t, 10692, PIECE_NB, SQUARE_NB, PIECE_TYPE_NB>;

// PieceToHistory is like ButterflyHistory but is addressed by a move's [piece][to]
using PieceToHistory = Stats<std::int16_t, 30000, PIECE_NB, SQUARE_NB>;

// ContinuationHistory is the combined history of a given pair of moves, usually
// the current one given a previous one. The nested history table is based on
// PieceToHistory instead of ButterflyBoards.
using ContinuationHistory = MultiArray<PieceToHistory, PIECE_NB, SQUARE_NB>;

// PawnHistory is addressed by the pawn structure and a move's [piece][to]
using PawnHistory =
  DynStats<AtomicStats<std::int16_t, 8192, PIECE_NB, SQUARE_NB>, PAWN_HISTORY_BASE_SIZE>;

// Correction histories record differences between the static evaluation of
// positions and their search score. It is used to improve the static evaluation
// used by some search heuristics.
// see https://www.chessprogramming.org/Static_Evaluation_Correction_History
enum CorrHistType {
    Pawn,          // By color and pawn structure
    Minor,         // By color and positions of minor pieces (Knight, Bishop)
    NonPawn,       // By non-pawn material positions and color
    PieceTo,       // By [piece][to] move
    Continuation,  // Combined history of move pairs
};

template<typename T, int D>
struct CorrectionBundle {
    StatsEntry<T, D, true> pawn;
    StatsEntry<T, D, true> minor;
    StatsEntry<T, D, true> nonPawnWhite;
    StatsEntry<T, D, true> nonPawnBlack;

    void operator=(T val) {
        pawn         = val;
        minor        = val;
        nonPawnWhite = val;
        nonPawnBlack = val;
    }
};

namespace Detail {

template<CorrHistType>
struct CorrHistTypedef {
    using type =
      DynStats<Stats<std::int16_t, CORRECTION_HISTORY_LIMIT, COLOR_NB>, CORRHIST_BASE_SIZE>;
};

template<>
struct CorrHistTypedef<PieceTo> {
    using type = Stats<std::int16_t, CORRECTION_HISTORY_LIMIT, PIECE_NB, SQUARE_NB>;
};

template<>
struct CorrHistTypedef<Continuation> {
    using type = MultiArray<CorrHistTypedef<PieceTo>::type, PIECE_NB, SQUARE_NB>;
};

template<>
struct CorrHistTypedef<NonPawn> {
    using type = DynStats<Stats<std::int16_t, CORRECTION_HISTORY_LIMIT, COLOR_NB, COLOR_NB>,
                          CORRHIST_BASE_SIZE>;
};

}

using UnifiedCorrectionHistory =
  DynStats<MultiArray<CorrectionBundle<std::int16_t, CORRECTION_HISTORY_LIMIT>, COLOR_NB>,
           CORRHIST_BASE_SIZE>;

template<CorrHistType T>
using CorrectionHistory = typename Detail::CorrHistTypedef<T>::type;

using TTMoveHistory = StatsEntry<std::int16_t, 8192>;

// Set of histories shared between groups of threads. To avoid excessive
// cross-node data transfer, histories are shared only between threads
// on a given NUMA node. The passed size must be a power of two to make
// the indexing more efficient.
struct SharedHistories {
    SharedHistories(size_t threadCount) :
        correctionHistory(threadCount),
        pawnHistory(threadCount) {
        assert((threadCount & (threadCount - 1)) == 0 && threadCount != 0);
        sizeMinus1         = correctionHistory.get_size() - 1;
        pawnHistSizeMinus1 = pawnHistory.get_size() - 1;
    }

    size_t get_size() const { return sizeMinus1 + 1; }

    auto& pawn_entry(const Position& pos) {
        return pawnHistory[pos.pawn_key() & pawnHistSizeMinus1];
    }
    const auto& pawn_entry(const Position& pos) const {
        return pawnHistory[pos.pawn_key() & pawnHistSizeMinus1];
    }

    auto& pawn_correction_entry(const Position& pos) {
        return correctionHistory[pos.pawn_key() & sizeMinus1];
    }
    const auto& pawn_correction_entry(const Position& pos) const {
        return correctionHistory[pos.pawn_key() & sizeMinus1];
    }

    auto& minor_piece_correction_entry(const Position& pos) {
        return correctionHistory[pos.minor_piece_key() & sizeMinus1];
    }
    const auto& minor_piece_correction_entry(const Position& pos) const {
        return correctionHistory[pos.minor_piece_key() & sizeMinus1];
    }

    template<Color c>
    auto& nonpawn_correction_entry(const Position& pos) {
        return correctionHistory[pos.non_pawn_key(c) & sizeMinus1];
    }
    template<Color c>
    const auto& nonpawn_correction_entry(const Position& pos) const {
        return correctionHistory[pos.non_pawn_key(c) & sizeMinus1];
    }

    UnifiedCorrectionHistory correctionHistory;
    PawnHistory              pawnHistory;


   private:
    size_t sizeMinus1, pawnHistSizeMinus1;
};

}  // namespace Stockfish

#endif  // #ifndef HISTORY_H_INCLUDED
