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
#include <memory>
#include <thread>
#include <type_traits>  // IWYU pragma: keep
#include <vector>

#include "memory.h"
#include "misc.h"
#include "position.h"

namespace Stockfish {

constexpr int PAWN_HISTORY_BASE_SIZE = 8192;  // has to be a power of 2
constexpr int UINT_16_HISTORY_SIZE   = std::numeric_limits<uint16_t>::max() + 1;

constexpr size_t THRESHOLD_AGGREGATED = 8;
constexpr size_t THRESHOLD_COUNTMIN   = 32;

constexpr int CM_DEPTH = 4;
constexpr int CM_WIDTH = 8192;

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

// PawnHistory types for atomic implementation
using PawnHistoryAtomic =
  DynStats<AtomicStats<std::int16_t, 8192, PIECE_NB, SQUARE_NB>, PAWN_HISTORY_BASE_SIZE>;

using PawnHistoryLocal = Stats<std::int16_t, 8192, PIECE_NB, SQUARE_NB>;

enum class PawnHistMode {
    ATOMIC,
    AGGREGATED,
    COUNTMIN
};

template<int Width, int Depth, int MaxValue>
class CountMinSketch {
   public:
    CountMinSketch() { clear(); }

    void clear() {
        for (int d = 0; d < Depth; d++)
            for (int w = 0; w < Width; w++)
                table[d][w].store(0, std::memory_order_relaxed);
    }

    void update(uint64_t key, int bonus) {
        int clampedBonus = std::clamp(bonus, -MaxValue, MaxValue);
        for (int d = 0; d < Depth; d++)
        {
            size_t idx    = hash(key, d) % Width;
            int    oldVal = table[d][idx].load(std::memory_order_relaxed);
            int    newVal =
              std::clamp(oldVal + clampedBonus - oldVal * std::abs(clampedBonus) / MaxValue,
                         -MaxValue, MaxValue);
            table[d][idx].store(static_cast<int16_t>(newVal), std::memory_order_relaxed);
        }
    }

    int query(uint64_t key) const {
        int minVal = MaxValue;
        for (int d = 0; d < Depth; d++)
        {
            size_t idx = hash(key, d) % Width;
            minVal     = std::min(minVal, int(table[d][idx].load(std::memory_order_relaxed)));
        }
        return minVal;
    }

   private:
    static size_t hash(uint64_t key, int depth) {
        constexpr uint64_t GOLDEN = 0x9E3779B97F4A7C15ULL;
        key ^= depth * GOLDEN;
        key ^= key >> 33;
        key *= 0xFF51AFD7ED558CCDULL;
        key ^= key >> 33;
        return static_cast<size_t>(key);
    }

    std::array<std::array<std::atomic<int16_t>, Width>, Depth> table;
};

template<int SizeMultiplier>
class CombinedPawnHistory;

template<int SizeMultiplier>
class PawnHistoryEntryProxy {
   public:
    PawnHistoryEntryProxy(CombinedPawnHistory<SizeMultiplier>* parent,
                          uint64_t                             pawnKey,
                          Piece                                pc,
                          Square                               sq,
                          size_t                               threadIdx = 0) :
        parent_(parent),
        pawnKey_(pawnKey),
        pc_(pc),
        sq_(sq),
        threadIdx_(threadIdx) { }

         operator int() const;
    void operator<<(int bonus);
    void operator=(int value);

   private:
    CombinedPawnHistory<SizeMultiplier>* parent_;
    uint64_t                             pawnKey_;
    Piece                                pc_;
    Square                               sq_;
    size_t                               threadIdx_;
};

template<int SizeMultiplier>
class PawnHistoryPieceProxy {
   public:
    PawnHistoryPieceProxy(CombinedPawnHistory<SizeMultiplier>* parent,
                          uint64_t                             pawnKey,
                          Piece                                pc,
                          size_t                               threadIdx = 0) :
        parent_(parent),
        pawnKey_(pawnKey),
        pc_(pc),
        threadIdx_(threadIdx) { }

    PawnHistoryEntryProxy<SizeMultiplier> operator[](Square sq) {
        return PawnHistoryEntryProxy<SizeMultiplier>(parent_, pawnKey_, pc_, sq, threadIdx_);
    }

   private:
    CombinedPawnHistory<SizeMultiplier>* parent_;
    uint64_t                             pawnKey_;
    Piece                                pc_;
    size_t                               threadIdx_;
};

template<int SizeMultiplier>
class PawnHistoryKeyProxy {
   public:
    PawnHistoryKeyProxy(CombinedPawnHistory<SizeMultiplier>* parent,
                        uint64_t                             pawnKey,
                        size_t                               threadIdx = 0) :
        parent_(parent),
        pawnKey_(pawnKey),
        threadIdx_(threadIdx) { }

    PawnHistoryPieceProxy<SizeMultiplier> operator[](Piece pc) {
        return PawnHistoryPieceProxy<SizeMultiplier>(parent_, pawnKey_, pc, threadIdx_);
    }

    void fill(int value) { parent_->fill_entry(pawnKey_, value); }

   private:
    CombinedPawnHistory<SizeMultiplier>* parent_;
    uint64_t                             pawnKey_;
    size_t                               threadIdx_;
};

template<int SizeMultiplier>
class PawnHistoryEntryProxyConst {
   public:
    PawnHistoryEntryProxyConst(const CombinedPawnHistory<SizeMultiplier>* parent,
                               uint64_t                                   pawnKey,
                               Piece                                      pc,
                               Square                                     sq) :
        parent_(parent),
        pawnKey_(pawnKey),
        pc_(pc),
        sq_(sq) { }

    operator int() const;

   private:
    const CombinedPawnHistory<SizeMultiplier>* parent_;
    uint64_t                                   pawnKey_;
    Piece                                      pc_;
    Square                                     sq_;
};

template<int SizeMultiplier>
class PawnHistoryPieceProxyConst {
   public:
    PawnHistoryPieceProxyConst(const CombinedPawnHistory<SizeMultiplier>* parent,
                               uint64_t                                   pawnKey,
                               Piece                                      pc) :
        parent_(parent),
        pawnKey_(pawnKey),
        pc_(pc) { }

    PawnHistoryEntryProxyConst<SizeMultiplier> operator[](Square sq) const {
        return PawnHistoryEntryProxyConst<SizeMultiplier>(parent_, pawnKey_, pc_, sq);
    }

   private:
    const CombinedPawnHistory<SizeMultiplier>* parent_;
    uint64_t                                   pawnKey_;
    Piece                                      pc_;
};

template<int SizeMultiplier>
class PawnHistoryKeyProxyConst {
   public:
    PawnHistoryKeyProxyConst(const CombinedPawnHistory<SizeMultiplier>* parent, uint64_t pawnKey) :
        parent_(parent),
        pawnKey_(pawnKey) { }

    PawnHistoryPieceProxyConst<SizeMultiplier> operator[](Piece pc) const {
        return PawnHistoryPieceProxyConst<SizeMultiplier>(parent_, pawnKey_, pc);
    }

   private:
    const CombinedPawnHistory<SizeMultiplier>* parent_;
    uint64_t                                   pawnKey_;
};

// Combined Pawn History: adaptive algorithm based on thread count
// 1-7 threads: Atomic (identical to current)
// 8-31 threads: Aggregated (per-thread local tables)
// 32+ threads: Count-Min Sketch (O(4) always)
template<int SizeMultiplier>
class CombinedPawnHistory {
   public:
    explicit CombinedPawnHistory(size_t threadCount) :
        numThreads_(threadCount),
        tableSize_(threadCount * SizeMultiplier) {

        if (threadCount < THRESHOLD_AGGREGATED)
        {
            mode_ = PawnHistMode::ATOMIC;
            // Only allocate atomicTable_ in ATOMIC mode
            atomicTable_ = std::make_unique<PawnHistoryAtomic>(threadCount);
            sizeMinus1_ = tableSize_ - 1;
        }
        else if (threadCount < THRESHOLD_COUNTMIN)
        {
            mode_ = PawnHistMode::AGGREGATED;
            localTables_.reserve(threadCount);
            // FIX: Each local table should be SizeMultiplier entries, not threadCount * SizeMultiplier
            // Total across all threads = threadCount * SizeMultiplier (same as ATOMIC)
            for (size_t i = 0; i < threadCount; i++)
                localTables_.emplace_back(
                  make_unique_large_page<PawnHistoryLocal[]>(SizeMultiplier));
            // FIX: Mask should be SizeMultiplier - 1 for AGGREGATED (each table is SizeMultiplier)
            sizeMinus1_ = SizeMultiplier - 1;
        }
        else
        {
            mode_ = PawnHistMode::COUNTMIN;
            sizeMinus1_ = tableSize_ - 1;
        }
    }

    int read_value(uint64_t pawnKey, Piece pc, Square sq) const {
        size_t idx = pawnKey & sizeMinus1_;

        switch (mode_)
        {
        case PawnHistMode::ATOMIC :
            return (*atomicTable_)[idx][pc][sq];

        case PawnHistMode::AGGREGATED : {
            int sum = 0;
            for (size_t t = 0; t < numThreads_; t++)
                sum += localTables_[t].get()[idx][pc][sq];
            return std::clamp(sum, -8192, 8192);
        }

        case PawnHistMode::COUNTMIN :
            return sketch_.query(make_cm_key(pawnKey, pc, sq));
        }
        return 0;
    }

    void write_value(uint64_t pawnKey, Piece pc, Square sq, int bonus, size_t threadIdx) {
        size_t idx = pawnKey & sizeMinus1_;

        switch (mode_)
        {
        case PawnHistMode::ATOMIC :
            (*atomicTable_)[idx][pc][sq] << bonus;
            break;

        case PawnHistMode::AGGREGATED : {
            // Use the passed thread index to write to correct per-thread table
            size_t tidx = threadIdx % numThreads_;
            localTables_[tidx].get()[idx][pc][sq] << bonus;
            break;
        }

        case PawnHistMode::COUNTMIN :
            sketch_.update(make_cm_key(pawnKey, pc, sq), bonus);
            break;
        }
    }

    void set_value(uint64_t pawnKey, Piece pc, Square sq, int value, size_t threadIdx) {
        size_t idx = pawnKey & sizeMinus1_;

        switch (mode_)
        {
        case PawnHistMode::ATOMIC :
            (*atomicTable_)[idx][pc][sq] = static_cast<int16_t>(value);
            break;

        case PawnHistMode::AGGREGATED : {
            // Set value in specified thread's table, zero others
            size_t tidx = threadIdx % numThreads_;
            for (size_t t = 0; t < numThreads_; t++)
                localTables_[t].get()[idx][pc][sq] = static_cast<int16_t>(t == tidx ? value : 0);
            break;
        }

        case PawnHistMode::COUNTMIN :
            break;
        }
    }

    void fill_entry(uint64_t pawnKey, int value) {
        size_t idx = pawnKey & sizeMinus1_;

        switch (mode_)
        {
        case PawnHistMode::ATOMIC :
            (*atomicTable_)[idx].fill(value);
            break;

        case PawnHistMode::AGGREGATED : {
            // Fill current thread's table, zero others
            size_t tidx = std::hash<std::thread::id>{}(std::this_thread::get_id()) % numThreads_;
            for (size_t t = 0; t < numThreads_; t++)
                localTables_[t].get()[idx].fill(t == tidx ? value : 0);
            break;
        }

        case PawnHistMode::COUNTMIN :
            break;
        }
    }

    void clear_range(int value, size_t threadIdx, size_t numaTotal) {
        switch (mode_)
        {
        case PawnHistMode::ATOMIC :
            atomicTable_->clear_range(value, threadIdx, numaTotal);
            break;

        case PawnHistMode::AGGREGATED :
            if (threadIdx < numThreads_)
            {
                // Each local table has SizeMultiplier entries (not numThreads_ * SizeMultiplier)
                for (size_t i = 0; i < SizeMultiplier; i++)
                    localTables_[threadIdx].get()[i].fill(value);
            }
            break;

        case PawnHistMode::COUNTMIN :
            if (threadIdx == 0)
                sketch_.clear();
            break;
        }
    }

    // Returns the logical table size (entries per table, not total entries across all threads)
    // ATOMIC: threadCount * SizeMultiplier (single shared table)
    // AGGREGATED: SizeMultiplier (each local table)
    // COUNTMIN: threadCount * SizeMultiplier
    size_t       get_size() const { return sizeMinus1_ + 1; }
    PawnHistMode get_mode() const { return mode_; }

    const void* get_prefetch_address(uint64_t pawnKey, Piece pc, Square sq) const {
        if (mode_ == PawnHistMode::ATOMIC && atomicTable_)
        {
            size_t idx = pawnKey & sizeMinus1_;
            return &(*atomicTable_)[idx][pc][sq];
        }
        return nullptr;
    }

    PawnHistoryKeyProxy<SizeMultiplier> get_entry(uint64_t pawnKey, size_t threadIdx = 0) {
        return PawnHistoryKeyProxy<SizeMultiplier>(this, pawnKey, threadIdx);
    }

    PawnHistoryKeyProxyConst<SizeMultiplier> get_entry(uint64_t pawnKey) const {
        return PawnHistoryKeyProxyConst<SizeMultiplier>(this, pawnKey);
    }

   private:
    static uint64_t make_cm_key(uint64_t pawnKey, Piece pc, Square sq) {
        return pawnKey ^ (uint64_t(pc) << 48) ^ (uint64_t(sq) << 40);
    }

    size_t       numThreads_;
    size_t       tableSize_;
    size_t       sizeMinus1_;
    PawnHistMode mode_;

    std::unique_ptr<PawnHistoryAtomic>               atomicTable_;  // Only allocated in ATOMIC mode
    std::vector<LargePagePtr<PawnHistoryLocal[]>>    localTables_;
    mutable CountMinSketch<CM_WIDTH, CM_DEPTH, 8192> sketch_;
};

template<int SizeMultiplier>
PawnHistoryEntryProxy<SizeMultiplier>::operator int() const {
    return parent_->read_value(pawnKey_, pc_, sq_);
}

template<int SizeMultiplier>
void PawnHistoryEntryProxy<SizeMultiplier>::operator<<(int bonus) {
    parent_->write_value(pawnKey_, pc_, sq_, bonus, threadIdx_);
}

template<int SizeMultiplier>
void PawnHistoryEntryProxy<SizeMultiplier>::operator=(int value) {
    parent_->set_value(pawnKey_, pc_, sq_, value, threadIdx_);
}

template<int SizeMultiplier>
PawnHistoryEntryProxyConst<SizeMultiplier>::operator int() const {
    return parent_->read_value(pawnKey_, pc_, sq_);
}

using PawnHistory = CombinedPawnHistory<PAWN_HISTORY_BASE_SIZE>;

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

    auto pawn_entry(const Position& pos, size_t threadIdx = 0) {
        return pawnHistory.get_entry(pos.pawn_key() & pawnHistSizeMinus1, threadIdx);
    }
    auto pawn_entry(const Position& pos) const {
        return pawnHistory.get_entry(pos.pawn_key() & pawnHistSizeMinus1);
    }

    const void* pawn_entry_prefetch_addr(const Position& pos, Piece pc, Square sq) const {
        return pawnHistory.get_prefetch_address(pos.pawn_key() & pawnHistSizeMinus1, pc, sq);
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
