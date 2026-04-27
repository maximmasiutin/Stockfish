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
#include <cstring>
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

constexpr int      HASHED_LOW_PLY_INDEX_BITS = 16;
constexpr size_t   HASHED_LOW_PLY_BASE_SIZE  = size_t(1) << HASHED_LOW_PLY_INDEX_BITS;
constexpr uint32_t HASHED_LOW_PLY_INDEX_MASK = uint32_t(HASHED_LOW_PLY_BASE_SIZE - 1);
constexpr int16_t  HASHED_LOW_PLY_INIT       = 98;
constexpr uint8_t  HASHED_LOW_PLY_EMPTY_TAG  = 0;

static_assert((PAWN_HISTORY_BASE_SIZE & (PAWN_HISTORY_BASE_SIZE - 1)) == 0,
              "PAWN_HISTORY_BASE_SIZE has to be a power of 2");

static_assert((CORRHIST_BASE_SIZE & (CORRHIST_BASE_SIZE - 1)) == 0,
              "CORRHIST_BASE_SIZE has to be a power of 2");

static_assert((HASHED_LOW_PLY_BASE_SIZE & (HASHED_LOW_PLY_BASE_SIZE - 1)) == 0,
              "HASHED_LOW_PLY_BASE_SIZE has to be a power of 2");

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

// LowPlyHistory: tagged hashed table keyed by (ply, move.raw()).
// Slot is 3 bytes: int16_t value followed by uint8_t tag. The fast read
// path issues a 4-byte load and unpacks; the trailing byte is the next
// slot's first byte, or tail padding for the last valid slot.
struct __attribute__((packed)) LowPlySlot {
    std::int16_t value;
    std::uint8_t tag;

    constexpr LowPlySlot() noexcept :
        value(HASHED_LOW_PLY_INIT),
        tag(HASHED_LOW_PLY_EMPTY_TAG) {}
    constexpr LowPlySlot(std::int16_t v, std::uint8_t t) noexcept :
        value(v),
        tag(t) {}
};
static_assert(sizeof(LowPlySlot) == 3, "LowPlySlot must be 3 bytes");
static_assert(alignof(LowPlySlot) == 1, "LowPlySlot must be byte-aligned for 3-byte stride");

// One trailing slot beyond HASHED_LOW_PLY_BASE_SIZE provides padding so
// that a 4-byte load at the last valid index does not overflow the
// storage. With sizeof(LowPlySlot)==3, the 4-byte load at index
// BASE_SIZE-1 reads bytes [3*(BASE_SIZE-1), 3*(BASE_SIZE-1)+3]; the
// final byte sits inside the trailing slot.
using LowPlyHistory = std::array<LowPlySlot, HASHED_LOW_PLY_BASE_SIZE + 1>;

// Static assertion: storage must include at least one byte of pad
// beyond the last valid slot so a 4-byte tail load is in-bounds.
static_assert(sizeof(LowPlyHistory) >= sizeof(LowPlySlot) * HASHED_LOW_PLY_BASE_SIZE + 1,
              "LowPlyHistory storage must extend at least 1 byte beyond the last "
              "valid slot so a 4-byte load at index BASE_SIZE-1 stays in-bounds");

struct LowPly {
    static constexpr int     VALUE_LIMIT = 7183;
    static constexpr int16_t INIT        = HASHED_LOW_PLY_INIT;

    static_assert(-VALUE_LIMIT >= std::numeric_limits<std::int16_t>::min()
                    && VALUE_LIMIT <= std::numeric_limits<std::int16_t>::max(),
                  "LowPly::VALUE_LIMIT must fit in int16_t for packed-slot storage");
    static_assert(INIT >= std::numeric_limits<std::int16_t>::min()
                    && INIT <= std::numeric_limits<std::int16_t>::max(),
                  "LowPly::INIT must fit in int16_t");

    static inline sf_always_inline std::uint32_t input(int ply, std::uint16_t move_raw) {
        assert(0 <= ply && ply < LOW_PLY_HISTORY_SIZE);
        return (std::uint32_t(unsigned(ply)) << 16) | move_raw;
    }
    static inline sf_always_inline std::uint64_t mix(std::uint32_t in) {
        std::uint64_t k = in;
        k *= 0x9E3779B97F4A7C15ULL;
        k ^= k >> 32;
        k *= 0xBF58476D1CE4E5B9ULL;
        k ^= k >> 27;
        return k;
    }
    static inline sf_always_inline std::uint32_t idx_from(std::uint64_t h) {
        return std::uint32_t(h) & HASHED_LOW_PLY_INDEX_MASK;
    }
    // 8-bit tag: take 8 bits of the hash above the index bits. Tag 0 is
    // reserved as the empty sentinel; remap 0 -> 1 so any occupied slot
    // tag is non-zero.
    static inline sf_always_inline std::uint8_t tag_from(std::uint64_t h) {
        const std::uint8_t t = std::uint8_t(h >> HASHED_LOW_PLY_INDEX_BITS);
        return t == 0 ? std::uint8_t(1) : t;
    }
    // Pack a (value, tag) pair into the low 24 bits of a uint32_t. The
    // high 8 bits are unused (next slot's first byte in storage).
    static inline sf_always_inline constexpr std::uint32_t pack(int v, std::uint8_t tag) {
        return std::uint32_t(std::uint16_t(v)) | (std::uint32_t(tag) << 16);
    }
    // Branchless extract: returns slot value when stored tag matches the
    // queried tag, else INIT. Operates on the raw 32-bit word loaded
    // from a slot address (low 24 bits = slot data, high 8 bits unused).
    static inline sf_always_inline constexpr int extract(std::uint32_t raw, std::uint8_t tag) {
        const std::uint32_t v       = std::uint32_t(std::uint16_t(raw & 0xFFFFu));
        const std::uint8_t  slotTag = std::uint8_t((raw >> 16) & 0xFFu);
        const std::uint32_t miss    = std::uint32_t(-std::int32_t(slotTag != tag));
        const std::uint32_t initu   = std::uint32_t(std::uint16_t(INIT));
        return int(std::int16_t((v & ~miss) | (initu & miss)));
    }
};

struct LowPlyReadAccess {
    const LowPlySlot* slot;
    std::uint8_t      tag;

    inline sf_always_inline int read() const {
        // 4-byte load at slot address: low 16 bits = value, next 8 bits
        // = stored tag, top 8 bits = next-slot byte (or tail pad).
        std::uint32_t raw;
        std::memcpy(&raw, slot, sizeof(raw));
        return LowPly::extract(raw, tag);
    }
    inline sf_always_inline operator int() const { return read(); }

    static inline sf_always_inline LowPlyReadAccess access(const LowPlyHistory& t,
                                                           int                  ply,
                                                           std::uint16_t        move_raw) {
        const std::uint64_t h = LowPly::mix(LowPly::input(ply, move_raw));
        return {&t[LowPly::idx_from(h)], LowPly::tag_from(h)};
    }
};

struct LowPlyAccess {
    LowPlySlot*  slot;
    std::uint8_t tag;

    inline sf_always_inline void operator<<(int bonus) {
        std::uint32_t raw;
        std::memcpy(&raw, slot, sizeof(raw));
        const int clampedBonus = std::clamp(bonus, -LowPly::VALUE_LIMIT, LowPly::VALUE_LIMIT);
        int       v            = LowPly::extract(raw, tag);
        v += clampedBonus - v * std::abs(clampedBonus) / LowPly::VALUE_LIMIT;
        assert(std::abs(v) <= LowPly::VALUE_LIMIT);
        // Write only the 3 bytes of this slot; preserve the trailing
        // byte (next slot's first byte) untouched.
        slot->value = std::int16_t(v);
        slot->tag   = tag;
    }

    static inline sf_always_inline LowPlyAccess access(LowPlyHistory& t,
                                                       int            ply,
                                                       std::uint16_t  move_raw) {
        const std::uint64_t h = LowPly::mix(LowPly::input(ply, move_raw));
        return {&t[LowPly::idx_from(h)], LowPly::tag_from(h)};
    }
};

// Compile-time round-trip checks of pack/extract.
static_assert(LowPly::extract(LowPly::pack(0, 0), 0) == 0,
              "extract(pack(0, 0), 0) must round-trip 0");
static_assert(LowPly::extract(LowPly::pack(0, 0), 1) == LowPly::INIT,
              "extract on tag mismatch must return INIT");
static_assert(LowPly::extract(LowPly::pack(LowPly::VALUE_LIMIT, 1), 1) == LowPly::VALUE_LIMIT,
              "extract must round-trip positive VALUE_LIMIT");
static_assert(LowPly::extract(LowPly::pack(-LowPly::VALUE_LIMIT, 1), 1) == -LowPly::VALUE_LIMIT,
              "extract must round-trip negative VALUE_LIMIT");
static_assert(LowPly::extract(LowPly::pack(LowPly::INIT, HASHED_LOW_PLY_EMPTY_TAG), 1)
                == LowPly::INIT,
              "extract on cleared slot (empty tag) must return INIT for any non-empty query tag");

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
