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

// ButterflyHistory records how often quiet moves have been successful or unsuccessful
// during the current search, and is used for reduction and move ordering decisions.
// It uses 2 tables (one for each color) indexed by the move's from and to squares,
// see https://www.chessprogramming.org/Butterfly_Boards
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

// Non-atomic pawn history entry for L0 cache
using PawnHistoryEntry = Stats<std::int16_t, 8192, PIECE_NB, SQUARE_NB>;

// L0 cache entry with optional per-piece lazy loading
struct L0PawnCacheEntry {
    uint64_t         pawnKey = 0;
    PawnHistoryEntry data;
    uint16_t         loadedMask = 0;  // Which pieces have been loaded (for partial mode)
    bool             dirty      = false;
};

// Per-thread L0 cache for pawn history
// - 1-2 threads: full size (8192), no collisions, matches master behavior
// - 3+ threads: small size (256) with per-piece lazy loading, reduces contention
class L0PawnCache {
   public:
    void init(size_t) {
        // v1 design: 256-entry L0 for all thread counts
        cacheSize   = 256;
        partialMode = false;
        cacheMask   = cacheSize - 1;
        cache       = std::make_unique<L0PawnCacheEntry[]>(cacheSize);
        clear();
    }

    void clear() {
        for (size_t i = 0; i < cacheSize; ++i)
        {
            cache[i].pawnKey = ~uint64_t(0);  // Use max value as "empty" sentinel
            cache[i].data.fill(-1238);
            cache[i].loadedMask = 0xFFFF;  // All pieces "loaded" (initialized)
            cache[i].dirty      = false;
        }
    }

    size_t size() const { return cacheSize; }
    bool   is_partial_mode() const { return partialMode; }

    bool enabled() const { return cache != nullptr; }

    bool contains(uint64_t pawnKey) const {
        return cache && cache[pawnKey & cacheMask].pawnKey == pawnKey;
    }

    // Get entry for reading (used in movepick)
    const PawnHistoryEntry& get(uint64_t pawnKey) const { return cache[pawnKey & cacheMask].data; }

    // Get entry for writing
    PawnHistoryEntry& get(uint64_t pawnKey) { return cache[pawnKey & cacheMask].data; }

    L0PawnCacheEntry& entry_at(size_t idx) { return cache[idx]; }

    // Insert entry into L0 cache (full 2KB copy from L1)
    template<typename T>
    void insert(uint64_t pawnKey, const T& l1Entry, Piece) {
        size_t idx = pawnKey & cacheMask;
        if (cache[idx].pawnKey != pawnKey)
        {
            cache[idx].pawnKey = pawnKey;
            cache[idx].dirty   = false;
            // Full copy
            for (int p = 0; p < PIECE_NB; ++p)
                for (int sq = 0; sq < SQUARE_NB; ++sq)
                    cache[idx].data[p][sq] = l1Entry[p][sq];
            cache[idx].loadedMask = 0xFFFF;
        }
    }

    // Get value from L0, loading from L1 if necessary (single operation, fastest)
    template<typename T>
    int16_t get_or_load(uint64_t pawnKey, const T& l1Entry, Piece pc, Square to) {
        size_t idx = pawnKey & cacheMask;

        if (cache[idx].pawnKey != pawnKey)
        {
            // Different pawn key - evict and load
            cache[idx].pawnKey = pawnKey;
            cache[idx].dirty   = false;

            if (partialMode)
            {
                cache[idx].loadedMask = 0;
                load_piece(idx, l1Entry, pc);
            }
            else
            {
                // Full mode: copy entire entry
                for (int p = 0; p < PIECE_NB; ++p)
                    for (int sq = 0; sq < SQUARE_NB; ++sq)
                        cache[idx].data[p][sq] = l1Entry[p][sq];
                cache[idx].loadedMask = 0xFFFF;
            }
        }
        else if (partialMode && !(cache[idx].loadedMask & (1 << pc)))
        {
            // Same pawn key but piece not loaded yet
            load_piece(idx, l1Entry, pc);
        }

        return cache[idx].data[pc][to];
    }

    void mark_dirty(uint64_t pawnKey, Piece pc) {
        size_t idx = pawnKey & cacheMask;
        if (cache[idx].pawnKey == pawnKey)
        {
            cache[idx].dirty = true;
            if (partialMode)
                cache[idx].loadedMask |= (1 << pc);
        }
    }

    uint16_t get_loaded_mask(uint64_t pawnKey) const {
        return cache[pawnKey & cacheMask].loadedMask;
    }

   private:
    template<typename T>
    void load_piece(size_t idx, const T& l1Entry, Piece pc) {
        for (int sq = 0; sq < SQUARE_NB; ++sq)
            cache[idx].data[pc][sq] = l1Entry[pc][sq];
        cache[idx].loadedMask |= (1 << pc);
    }

    std::unique_ptr<L0PawnCacheEntry[]> cache;
    size_t                              cacheSize   = 0;
    size_t                              cacheMask   = 0;
    bool                                partialMode = false;
};

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

    auto&       pawn_entry(uint64_t pawnKey) { return pawnHistory[pawnKey & pawnHistSizeMinus1]; }
    const auto& pawn_entry(uint64_t pawnKey) const {
        return pawnHistory[pawnKey & pawnHistSizeMinus1];
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
