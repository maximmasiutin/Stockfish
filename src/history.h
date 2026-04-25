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
constexpr int CORRHIST_BASE_SIZE       = 2 * UINT_16_HISTORY_SIZE;
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

constexpr int           CORRHIST_VALUE_MIN     = -1024;
constexpr int           CORRHIST_VALUE_MAX     = -CORRHIST_VALUE_MIN - 1;
constexpr int           CORRHIST_INIT_VALUE    = 0;
constexpr std::size_t   CORRHIST_SLOT_BITS     = sizeof(std::uint16_t) * 8;
constexpr std::size_t   CORRHIST_VALUE_BITS    = ilog2(std::size_t(-CORRHIST_VALUE_MIN)) + 1;
constexpr std::size_t   CORRHIST_TAG_BITS      = CORRHIST_SLOT_BITS - CORRHIST_VALUE_BITS;
constexpr std::size_t   CORRHIST_KEY_BITS      = sizeof(std::uint64_t) * 8;
constexpr std::size_t   CORRHIST_TAG_KEY_SHIFT = CORRHIST_KEY_BITS - CORRHIST_TAG_BITS;
constexpr std::uint16_t CORRHIST_TAG_RAW_MASK  = std::uint16_t((1u << CORRHIST_TAG_BITS) - 1u);

constexpr std::size_t CORRHIST_INDEX_BASE_BITS     = ilog2(std::size_t(CORRHIST_BASE_SIZE));
constexpr std::size_t CORRHIST_DOMAIN_LOG2_PAWN    = 26;
constexpr std::size_t CORRHIST_DOMAIN_LOG2_MINOR   = 23;
constexpr std::size_t CORRHIST_DOMAIN_LOG2_NONPAWN = 25;
constexpr std::size_t CORRHIST_DOMAIN_LOG2_MAX     = CORRHIST_DOMAIN_LOG2_PAWN;

constexpr std::uint64_t CORRHIST_MAX_THREADS_STRUCT =
  std::uint64_t(1) << (CORRHIST_TAG_KEY_SHIFT - CORRHIST_INDEX_BASE_BITS);
constexpr std::uint64_t CORRHIST_MAX_THREADS_DATA =
  std::uint64_t(1) << (CORRHIST_DOMAIN_LOG2_MAX - CORRHIST_INDEX_BASE_BITS);
constexpr std::size_t CORRHIST_MAX_THREADS =
  std::size_t(std::min(CORRHIST_MAX_THREADS_STRUCT, CORRHIST_MAX_THREADS_DATA));

using CorrhistTag = std::uint8_t;

static_assert(CORRHIST_VALUE_BITS + CORRHIST_TAG_BITS == CORRHIST_SLOT_BITS);
static_assert(CORRHIST_TAG_BITS >= 1);
static_assert(CORRHIST_TAG_BITS <= sizeof(CorrhistTag) * 8);
static_assert(-CORRHIST_VALUE_MIN == CORRECTION_HISTORY_LIMIT);
static_assert(CORRHIST_INIT_VALUE == 0);
static_assert(CORRHIST_INDEX_BASE_BITS <= CORRHIST_TAG_KEY_SHIFT);
static_assert(CORRHIST_INDEX_BASE_BITS <= CORRHIST_DOMAIN_LOG2_MAX);

inline CorrhistTag corrhist_tag_from(std::uint64_t key) {
    const CorrhistTag t = CorrhistTag((key >> CORRHIST_TAG_KEY_SHIFT) & CORRHIST_TAG_RAW_MASK);
    return t == 0 ? CorrhistTag(1) : t;
}

struct alignas(2) CorrhistTaggedSlot {
    std::atomic<std::int16_t> word{0};

    void operator=(int v) {
        assert(v == 0);
        (void) v;
        word.store(0, std::memory_order_relaxed);
    }

    int read(CorrhistTag tag) const {
        const int u = word.load(std::memory_order_relaxed);
        return (u & CORRHIST_TAG_RAW_MASK) == int(tag) ? u >> CORRHIST_TAG_BITS : 0;
    }

    inline sf_always_inline void update(CorrhistTag tag, int bonus) {
        assert((tag & ~CORRHIST_TAG_RAW_MASK) == 0);
        constexpr int D = -CORRHIST_VALUE_MIN;
        assert(std::abs(bonus) <= D);
        int v = read(tag);
        v     = v + bonus - v * std::abs(bonus) / D;
        v     = std::min(v, CORRHIST_VALUE_MAX);
        assert(v >= CORRHIST_VALUE_MIN);
        word.store(std::int16_t((std::uint32_t(v) << CORRHIST_TAG_BITS) | std::uint32_t(tag)),
                   std::memory_order_relaxed);
    }
};
static_assert(sizeof(CorrhistTaggedSlot) == 2);

struct CorrhistReadAccess {
    const CorrhistTaggedSlot* slot;
    CorrhistTag               tag;

    inline sf_always_inline int read() const { return slot->read(tag); }
    inline sf_always_inline     operator int() const { return read(); }

    static CorrhistReadAccess pawn(const struct SharedHistories& sh, const Position& pos);
    static CorrhistReadAccess minor(const struct SharedHistories& sh, const Position& pos);
    template<Color Us>
    static CorrhistReadAccess nonpawn(const struct SharedHistories& sh, const Position& pos);
};

struct CorrhistAccess {
    CorrhistTaggedSlot* slot;
    CorrhistTag         tag;

    inline sf_always_inline void operator<<(int bonus) const { slot->update(tag, bonus); }

    static CorrhistAccess pawn(struct SharedHistories& sh, const Position& pos);
    static CorrhistAccess minor(struct SharedHistories& sh, const Position& pos);
    template<Color Us>
    static CorrhistAccess nonpawn(struct SharedHistories& sh, const Position& pos);
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
    static_assert(std::is_same_v<T, std::int16_t> && D == CORRECTION_HISTORY_LIMIT,
                  "Tagged CorrectionBundle supports only int16_t and CORRECTION_HISTORY_LIMIT");

    CorrhistTaggedSlot pawn;
    CorrhistTaggedSlot minor;
    CorrhistTaggedSlot nonPawnWhite;
    CorrhistTaggedSlot nonPawnBlack;

    void operator=(T val) {
        assert(val == 0);
        (void) val;
        pawn         = 0;
        minor        = 0;
        nonPawnWhite = 0;
        nonPawnBlack = 0;
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
        correctionHistory(std::min(threadCount, CORRHIST_MAX_THREADS)),
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

    UnifiedCorrectionHistory correctionHistory;
    PawnHistory              pawnHistory;

    CorrectionBundle<std::int16_t, CORRECTION_HISTORY_LIMIT>& get_bundle(std::uint64_t key,
                                                                         Color         us) {
        return correctionHistory[key & sizeMinus1][us];
    }
    const CorrectionBundle<std::int16_t, CORRECTION_HISTORY_LIMIT>& get_bundle(std::uint64_t key,
                                                                               Color us) const {
        return correctionHistory[key & sizeMinus1][us];
    }

   private:
    size_t sizeMinus1, pawnHistSizeMinus1;
};

inline CorrhistReadAccess CorrhistReadAccess::pawn(const SharedHistories& sh, const Position& pos) {
    const auto k = pos.pawn_key();
    return {&sh.get_bundle(k, pos.side_to_move()).pawn, corrhist_tag_from(k)};
}
inline CorrhistReadAccess CorrhistReadAccess::minor(const SharedHistories& sh,
                                                    const Position&        pos) {
    const auto k = pos.minor_piece_key();
    return {&sh.get_bundle(k, pos.side_to_move()).minor, corrhist_tag_from(k)};
}
template<Color Us>
inline CorrhistReadAccess CorrhistReadAccess::nonpawn(const SharedHistories& sh,
                                                      const Position&        pos) {
    const auto  k      = pos.non_pawn_key(Us);
    const auto& bundle = sh.get_bundle(k, pos.side_to_move());
    if constexpr (Us == WHITE)
        return {&bundle.nonPawnWhite, corrhist_tag_from(k)};
    else
        return {&bundle.nonPawnBlack, corrhist_tag_from(k)};
}

inline CorrhistAccess CorrhistAccess::pawn(SharedHistories& sh, const Position& pos) {
    const auto k = pos.pawn_key();
    return {&sh.get_bundle(k, pos.side_to_move()).pawn, corrhist_tag_from(k)};
}
inline CorrhistAccess CorrhistAccess::minor(SharedHistories& sh, const Position& pos) {
    const auto k = pos.minor_piece_key();
    return {&sh.get_bundle(k, pos.side_to_move()).minor, corrhist_tag_from(k)};
}
template<Color Us>
inline CorrhistAccess CorrhistAccess::nonpawn(SharedHistories& sh, const Position& pos) {
    const auto k      = pos.non_pawn_key(Us);
    auto&      bundle = sh.get_bundle(k, pos.side_to_move());
    if constexpr (Us == WHITE)
        return {&bundle.nonPawnWhite, corrhist_tag_from(k)};
    else
        return {&bundle.nonPawnBlack, corrhist_tag_from(k)};
}

}  // namespace Stockfish

#endif  // #ifndef HISTORY_H_INCLUDED
