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

// Frequency-aware continuationHistory inner permutation. Reorders the inner
// Piece axis of PieceToHistory so the highest-volume piece codes (W_ROOK,
// B_ROOK, W_KING, B_KING, ...) cluster at the lowest addresses inside each
// 1.5 KiB PieceToHistory instance. Designed from the empirical 9-cell,
// 134-blob, 670-billion-access conthist capture documented in
// research/conthist-cell-counters.md. Bijective on the full 16-entry Piece
// space, so every distinct chess piece code maps to a unique permuted slot
// and bench-byte-equality is preserved when applied uniformly at every
// inner conthist read/write site.
//
// Order (canonical aggregate ranking, hot to cold):
//   W_ROOK, B_ROOK, W_KING, B_KING, W_QUEEN, B_QUEEN,
//   W_PAWN, W_BISHOP, B_BISHOP, B_PAWN, W_KNIGHT, B_KNIGHT,
//   NO_PIECE, [unused 7], [unused 8], [unused 15]
alignas(64) inline constexpr std::array<std::uint8_t, PIECE_NB> CONT_HIST_PIECE_PERM = {
  13,  //  0 NO_PIECE -> rank 14 (cold)
  6,   //  1 W_PAWN   -> rank 7
  10,  //  2 W_KNIGHT -> rank 11
  7,   //  3 W_BISHOP -> rank 8
  0,   //  4 W_ROOK   -> rank 1 (hottest)
  4,   //  5 W_QUEEN  -> rank 5
  2,   //  6 W_KING   -> rank 3
  14,  //  7 unused   -> dead slot
  15,  //  8 unused   -> dead slot
  9,   //  9 B_PAWN   -> rank 10
  11,  // 10 B_KNIGHT -> rank 12 (coldest realized)
  8,   // 11 B_BISHOP -> rank 9
  1,   // 12 B_ROOK   -> rank 2
  5,   // 13 B_QUEEN  -> rank 6
  3,   // 14 B_KING   -> rank 4
  12   // 15 unused   -> dead slot
};

sf_always_inline inline Piece cont_hist_piece(Piece pc) {
    return Piece(CONT_HIST_PIECE_PERM[std::uint32_t(pc) & 0xFu]);
}

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
