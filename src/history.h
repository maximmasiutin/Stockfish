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

constexpr int      HASHED_CONTCORR_INDEX_BITS = 17;
constexpr size_t   HASHED_CONTCORR_BASE_SIZE  = size_t(1) << HASHED_CONTCORR_INDEX_BITS;
constexpr uint32_t HASHED_CONTCORR_INDEX_MASK = uint32_t(HASHED_CONTCORR_BASE_SIZE - 1);

static_assert((PAWN_HISTORY_BASE_SIZE & (PAWN_HISTORY_BASE_SIZE - 1)) == 0,
              "PAWN_HISTORY_BASE_SIZE has to be a power of 2");

static_assert((CORRHIST_BASE_SIZE & (CORRHIST_BASE_SIZE - 1)) == 0,
              "CORRHIST_BASE_SIZE has to be a power of 2");

static_assert((HASHED_CONTCORR_BASE_SIZE & (HASHED_CONTCORR_BASE_SIZE - 1)) == 0,
              "HASHED_CONTCORR_BASE_SIZE has to be a power of 2");

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

struct alignas(2) HashedContCorrSlot {
    using Tag = std::uint8_t;

    static constexpr int     TAG_BITS = 5;
    static constexpr Tag     TAG_MASK = Tag((1u << TAG_BITS) - 1);
    static constexpr int16_t INIT     = 6;
    static constexpr int16_t NOK      = 8;

    static constexpr int WORD_BITS  = 16;
    static constexpr int VALUE_BITS = WORD_BITS - TAG_BITS;
    static constexpr int VALUE_MAX  = (1 << (VALUE_BITS - 1)) - 1;
    static constexpr int VALUE_MIN  = -(1 << (VALUE_BITS - 1));

    static constexpr std::size_t GRAVITY_SHIFT_BITS = ilog2(std::size_t(CORRECTION_HISTORY_LIMIT));

    static_assert(VALUE_BITS == 11, "HashedContCorrSlot value field must be 11 bits");
    static_assert(CORRECTION_HISTORY_LIMIT <= 1024,
                  "value is an 11-bit saturated signed field; widen the field or "
                  "lower CORRECTION_HISTORY_LIMIT if this grows");
    static_assert((std::size_t(1) << GRAVITY_SHIFT_BITS) == std::size_t(CORRECTION_HISTORY_LIMIT),
                  "CORRECTION_HISTORY_LIMIT must be a power of 2 for shift-based gravity");

    std::uint16_t word;

    static inline sf_always_inline constexpr std::uint16_t pack(int v, Tag t) {
        return std::uint16_t((std::uint32_t(v) << TAG_BITS) | std::uint32_t(t));
    }
    static inline sf_always_inline constexpr std::uint16_t empty_data() {
        return pack(INIT, Tag(0));
    }

    static inline sf_always_inline constexpr int extract(std::uint16_t w, Tag t) {
        const std::uint32_t u         = std::uint32_t(std::uint16_t(w));
        const std::uint32_t storedTag = u & std::uint32_t(TAG_MASK);
        const int           v         = int(std::int16_t(w)) >> TAG_BITS;
        const std::uint32_t mask      = std::uint32_t(-std::int32_t(storedTag != std::uint32_t(t)));
        return int(int32_t((std::uint32_t(v) & ~mask) | (std::uint32_t(INIT) & mask)));
    }

    static inline sf_always_inline int read(std::uint16_t w, Tag t) { return extract(w, t); }

    static inline sf_always_inline void update(HashedContCorrSlot& s, Tag t, int b) {
        assert((t & ~TAG_MASK) == 0);
        assert(std::abs(b) <= CORRECTION_HISTORY_LIMIT);
        const int  u            = int(std::int16_t(s.word));
        const int  storedTag    = u & int(TAG_MASK);
        const int  v_old        = u >> TAG_BITS;
        const Tag  effectiveTag = (b != 0) ? t : Tag(storedTag);
        const bool tagMatch     = (storedTag == int(effectiveTag));
        int        v            = tagMatch ? v_old : int(INIT);
        v                       = v + b - ((v * std::abs(b)) >> GRAVITY_SHIFT_BITS);
        v                       = std::min(v, int(VALUE_MAX));
        assert(VALUE_MIN <= v && v <= VALUE_MAX);
        if (!tagMatch && std::abs(v) <= std::abs(v_old))
            return;
        s.word = pack(v, effectiveTag);
    }
};
static_assert(sizeof(HashedContCorrSlot) == 2, "HashedContCorrSlot must be 2 bytes");
static_assert(HashedContCorrSlot::extract(HashedContCorrSlot::empty_data(),
                                          HashedContCorrSlot::Tag(0))
                == HashedContCorrSlot::INIT,
              "empty_data must return INIT for tag 0");
static_assert(HashedContCorrSlot::extract(HashedContCorrSlot::empty_data(),
                                          HashedContCorrSlot::Tag(1))
                == HashedContCorrSlot::INIT,
              "empty_data must return INIT for non-zero tag");
static_assert(HashedContCorrSlot::extract(HashedContCorrSlot::pack(HashedContCorrSlot::VALUE_MAX,
                                                                   HashedContCorrSlot::Tag(1)),
                                          HashedContCorrSlot::Tag(1))
                == HashedContCorrSlot::VALUE_MAX,
              "pack/extract round-trip positive VALUE_MAX");
static_assert(HashedContCorrSlot::extract(HashedContCorrSlot::pack(HashedContCorrSlot::VALUE_MIN,
                                                                   HashedContCorrSlot::Tag(1)),
                                          HashedContCorrSlot::Tag(1))
                == HashedContCorrSlot::VALUE_MIN,
              "pack/extract round-trip negative VALUE_MIN");

struct HashedContCorrReadAccess {
    const HashedContCorrSlot* slot;
    HashedContCorrSlot::Tag   tag;

    inline sf_always_inline int read() const { return HashedContCorrSlot::read(slot->word, tag); }
    inline sf_always_inline     operator int() const { return read(); }
};

struct HashedContCorrAccess {
    HashedContCorrSlot*     slot;
    HashedContCorrSlot::Tag tag;

    inline sf_always_inline void operator<<(int bonus) {
        HashedContCorrSlot::update(*slot, tag, bonus);
    }
};

using HashedContCorrHistory = std::array<HashedContCorrSlot, HASHED_CONTCORR_BASE_SIZE>;

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
