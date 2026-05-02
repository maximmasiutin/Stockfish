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

#include "movepick.h"

#include <cassert>
#include <limits>
#include <utility>

#if defined(USE_AVX512ICL)
    #include <immintrin.h>
#endif

#include "bitboard.h"
#include "misc.h"
#include "position.h"

namespace Stockfish {

namespace {

enum Stages {
    // generate main search moves
    MAIN_TT,
    CAPTURE_INIT,
    GOOD_CAPTURE,
    QUIET_INIT,
    GOOD_QUIET,
    BAD_CAPTURE,
    BAD_QUIET,

    // generate evasion moves
    EVASION_TT,
    EVASION_INIT,
    EVASION,

    // generate probcut moves
    PROBCUT_TT,
    PROBCUT_INIT,
    PROBCUT,

    // generate qsearch moves
    QSEARCH_TT,
    QCAPTURE_INIT,
    QCAPTURE
};


// Full insertion sort in descending order; used at CAPTURE/PROBCUT/QCAPTURE_INIT.
inline sf_always_inline void insertion_sort(ExtMove* begin, ExtMove* end) {
    for (ExtMove *sortedEnd = begin, *p = begin + 1; p < end; ++p)
    {
        ExtMove tmp = *p, *q;
        *p          = *++sortedEnd;
        for (q = sortedEnd; q != begin && *(q - 1) < tmp; --q)
            *q = *(q - 1);
        *q = tmp;
    }
}

#if defined(USE_AVX512ICL)

// Splat the move bits and value of an ExtMove into all lanes of two registers.
void splat_extmove(const ExtMove& m, __m512i& move, __m512i& value) {
    move  = _mm512_set1_epi32(m.raw());
    value = _mm512_set1_epi32(m.value);
}

// 16-element stable descending insertion sorter; per-insert is cmplt + kadd + 2x expand.
struct SimdSorter {
    static constexpr int MAX_ELEMENTS = 16;
    __m512i              sortedValues, sortedMoves;

    explicit SimdSorter(const ExtMove& first) {
        splat_extmove(first, sortedMoves, sortedValues);
        sortedValues = _mm512_mask_set1_epi32(sortedValues, ~1, std::numeric_limits<int>::min());
    }

    void insert(const ExtMove& m) {
        __m512i move, value;
        splat_extmove(m, move, value);

        assert(m.value != std::numeric_limits<int>::min());
        const __mmask16 expand =
          _kadd_mask16(_mm512_cmplt_epi32_mask(sortedValues, value), __mmask16(-1));

        sortedValues = _mm512_mask_expand_epi32(value, expand, sortedValues);
        sortedMoves  = _mm512_mask_expand_epi32(move, expand, sortedMoves);
    }

    // Drain with predicted branch; skips second store at count <= 8 (EVASIONS-typical).
    void write_sorted_branchful(ExtMove* dst, int count) const {
        static_assert(sizeof(ExtMove) == 8);
        assert(count >= 1 && count <= MAX_ELEMENTS);

        auto write = [&](int offset, const __m512i indices) {
            const __m512i extMoves = _mm512_permutex2var_epi32(sortedMoves, indices, sortedValues);
            const std::ptrdiff_t storeCount = count - offset;
            if (storeCount > 0)
                _mm512_mask_storeu_epi64(dst + offset, (1U << storeCount) - 1, extMoves);
        };
        write(0, _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23));
        write(8, _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31));
    }

    // Branchless drain; avoids the mispredict tax at QUIETS where count straddles 8.
    void write_sorted_branchless(ExtMove* dst, int count) const {
        static_assert(sizeof(ExtMove) == 8);
        assert(count >= 1 && count <= MAX_ELEMENTS);

        const __m512i interleave_lo =
          _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
        const __m512i interleave_hi =
          _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);

        const __m512i v0 = _mm512_permutex2var_epi32(sortedMoves, interleave_lo, sortedValues);
        const __m512i v1 = _mm512_permutex2var_epi32(sortedMoves, interleave_hi, sortedValues);

        const __mmask8 m0 = __mmask8((1U << std::min(count, 8)) - 1);
        const __mmask8 m1 = __mmask8((1U << std::max(std::min(count - 8, 8), 0)) - 1);

        _mm512_mask_storeu_epi64(dst, m0, v0);
        _mm512_mask_storeu_epi64(dst + 8, m1, v1);
    }
};

#endif

// Full SIMD insertion sort for EVASION_INIT; inlined to keep per-call fixed cost low at N ~ 6.
inline sf_always_inline void insertion_sort_medium(ExtMove* begin, ExtMove* end) {
    // EVASIONS may be empty in checkmate edges; SIMD tolerates begin == end (harmless masked store).
    assert(begin <= end);

    ExtMove* sortedEnd = begin;
    ExtMove* p         = begin + 1;

#if defined(USE_AVX512ICL)
    SimdSorter sorter(*begin);
    for (; p < end; ++p)
    {
        if (sortedEnd - begin + 1 >= SimdSorter::MAX_ELEMENTS)
            break;
        sorter.insert(*p);
        *p = *++sortedEnd;
    }
    sorter.write_sorted_branchful(begin, int(sortedEnd - begin + 1));
#endif

    for (; p < end; ++p)
    {
        ExtMove tmp = *p, *q;
        *p          = *++sortedEnd;
        for (q = sortedEnd; q != begin && *(q - 1) < tmp; --q)
            *q = *(q - 1);
        *q = tmp;
    }
}

// Descending sort for QUIETS; SIMD prefix byte-equivalent to scalar, scalar tail past MAX_ELEMENTS.
// Suffix "long" reflects that QUIETS mean N exceeds 16 here.
sf_noinline void partial_insertion_sort_long(ExtMove* begin, ExtMove* end, int limit) {

    // QUIETS may be empty in rare DFRC; SIMD tolerates begin == end (harmless masked store).
    assert(begin <= end);

    ExtMove* sortedEnd = begin;
    ExtMove* p         = begin + 1;

#if defined(USE_AVX512ICL)
    SimdSorter sorter(*begin);
    for (; p < end; ++p)
        if (p->value >= limit)
        {
            if (sortedEnd - begin + 1 >= SimdSorter::MAX_ELEMENTS)
                break;
            sorter.insert(*p);
            *p = *++sortedEnd;
        }
    sorter.write_sorted_branchless(begin, int(sortedEnd - begin + 1));
#endif

    for (; p < end; ++p)
        if (p->value >= limit)
        {
            ExtMove tmp = *p, *q;
            *p          = *++sortedEnd;
            for (q = sortedEnd; q != begin && *(q - 1) < tmp; --q)
                *q = *(q - 1);
            *q = tmp;
        }
}

}  // namespace


// Constructors of the MovePicker class. As arguments, we pass information
// to decide which class of moves to emit, to help sorting the (presumably)
// good moves first, and how important move ordering is at the current node.

// MovePicker constructor for the main search and for the quiescence search
MovePicker::MovePicker(const Position&              p,
                       Move                         ttm,
                       Depth                        d,
                       const ButterflyHistory*      mh,
                       const LowPlyHistory*         lph,
                       const CapturePieceToHistory* cph,
                       const PieceToHistory**       ch,
                       const SharedHistories*       sh,
                       int                          pl) :
    pos(p),
    mainHistory(mh),
    lowPlyHistory(lph),
    captureHistory(cph),
    continuationHistory(ch),
    sharedHistory(sh),
    ttMove(ttm),
    depth(d),
    ply(pl) {

    if (pos.checkers())
        stage = EVASION_TT + !(ttm && pos.pseudo_legal(ttm));

    else
        stage = (depth > 0 ? MAIN_TT : QSEARCH_TT) + !(ttm && pos.pseudo_legal(ttm));
}

// MovePicker constructor for ProbCut: we generate captures with Static Exchange
// Evaluation (SEE) greater than or equal to the given threshold.
MovePicker::MovePicker(const Position& p, Move ttm, int th, const CapturePieceToHistory* cph) :
    pos(p),
    captureHistory(cph),
    ttMove(ttm),
    threshold(th) {
    assert(!pos.checkers());

    stage = PROBCUT_TT + !(ttm && pos.capture_stage(ttm) && pos.pseudo_legal(ttm));
}

// Assigns a numerical value to each move in a list, used for sorting.
// Captures are ordered by Most Valuable Victim (MVV), preferring captures
// with a good history. Quiets moves are ordered using the history tables.
template<GenType Type>
ExtMove* MovePicker::score(const MoveList<Type>& ml) {

    static_assert(Type == CAPTURES || Type == QUIETS || Type == EVASIONS, "Wrong type");

    Color us = pos.side_to_move();

    [[maybe_unused]] Bitboard threatByLesser[KING + 1];
    if constexpr (Type == QUIETS)
    {
        threatByLesser[PAWN]   = 0;
        threatByLesser[KNIGHT] = threatByLesser[BISHOP] = pos.attacks_by<PAWN>(~us);
        threatByLesser[ROOK] =
          pos.attacks_by<KNIGHT>(~us) | pos.attacks_by<BISHOP>(~us) | threatByLesser[KNIGHT];
        threatByLesser[QUEEN] = pos.attacks_by<ROOK>(~us) | threatByLesser[ROOK];
        threatByLesser[KING]  = 0;
    }

    ExtMove* it = cur;
    for (auto move : ml)
    {
        ExtMove& m = *it++;
        m          = move;

        const Square    from          = m.from_sq();
        const Square    to            = m.to_sq();
        const Piece     pc            = pos.moved_piece(m);
        const PieceType pt            = type_of(pc);
        const Piece     capturedPiece = pos.piece_on(to);

        if constexpr (Type == CAPTURES)
            m.value = (*captureHistory)[pc][to][type_of(capturedPiece)]
                    + 7 * int(PieceValue[capturedPiece]);

        else if constexpr (Type == QUIETS)
        {
            // histories
            m.value = 2 * (*mainHistory)[us][m.raw()];
            m.value += 2 * sharedHistory->pawn_entry(pos)[pc][to];
            m.value += (*continuationHistory[0])[pc][to];
            m.value += (*continuationHistory[1])[pc][to];
            m.value += (*continuationHistory[2])[pc][to];
            m.value += (*continuationHistory[3])[pc][to];
            m.value += (*continuationHistory[5])[pc][to];

            // bonus for checks
            m.value += ((pos.check_squares(pt) & to) && pos.see_ge(m, -75)) * 16384;

            // penalty for moving to a square threatened by a lesser piece
            // or bonus for escaping an attack by a lesser piece.
            int v = 20 * (bool(threatByLesser[pt] & from) - bool(threatByLesser[pt] & to));
            m.value += PieceValue[pt] * v;


            if (ply < LOW_PLY_HISTORY_SIZE)
                m.value += 8 * (*lowPlyHistory)[ply][m.raw()] / (1 + ply);
        }

        else  // Type == EVASIONS
        {
            if (pos.capture_stage(m))
                m.value = PieceValue[capturedPiece] + (1 << 28);
            else
                m.value = (*mainHistory)[us][m.raw()] + (*continuationHistory[0])[pc][to];
        }
    }
    return it;
}

// Returns the next move satisfying a predicate function.
// This never returns the TT move, as it was emitted before.
template<typename Pred>
Move MovePicker::select(Pred filter) {

    for (; cur < endCur; ++cur)
        if (*cur != ttMove && filter())
            return *cur++;

    return Move::none();
}

// This is the most important method of the MovePicker class. We emit one
// new pseudo-legal move on every call until there are no more moves left,
// picking the move with the highest score from a list of generated moves.
Move MovePicker::next_move() {

    constexpr int goodQuietThreshold = -14000;
top:
    switch (stage)
    {

    case MAIN_TT :
    case EVASION_TT :
    case QSEARCH_TT :
    case PROBCUT_TT :
        ++stage;
        return ttMove;

    case CAPTURE_INIT :
    case PROBCUT_INIT :
    case QCAPTURE_INIT : {
        MoveList<CAPTURES> ml(pos);

        cur = endBadCaptures = moves;
        endCur = endCaptures = score<CAPTURES>(ml);

        insertion_sort(cur, endCur);
        ++stage;
        goto top;
    }

    case GOOD_CAPTURE :
        if (select([&]() {
                if (pos.see_ge(*cur, -cur->value / 18))
                    return true;
                std::swap(*endBadCaptures++, *cur);
                return false;
            }))
            return *(cur - 1);

        ++stage;
        [[fallthrough]];

    case QUIET_INIT :
        if (!skipQuiets)
        {
            MoveList<QUIETS> ml(pos);

            endCur = endGenerated = score<QUIETS>(ml);

            partial_insertion_sort_long(cur, endCur, -3560 * depth);
        }

        ++stage;
        [[fallthrough]];

    case GOOD_QUIET :
        if (!skipQuiets && select([&]() { return cur->value > goodQuietThreshold; }))
            return *(cur - 1);

        // Prepare the pointers to loop over the bad captures
        cur    = moves;
        endCur = endBadCaptures;

        ++stage;
        [[fallthrough]];

    case BAD_CAPTURE :
        if (select([]() { return true; }))
            return *(cur - 1);

        // Prepare the pointers to loop over quiets again
        cur    = endCaptures;
        endCur = endGenerated;

        ++stage;
        [[fallthrough]];

    case BAD_QUIET :
        if (!skipQuiets)
            return select([&]() { return cur->value <= goodQuietThreshold; });

        return Move::none();

    case EVASION_INIT : {
        MoveList<EVASIONS> ml(pos);

        cur    = moves;
        endCur = endGenerated = score<EVASIONS>(ml);

        insertion_sort_medium(cur, endCur);
        ++stage;
        [[fallthrough]];
    }

    case EVASION :
    case QCAPTURE :
        return select([]() { return true; });

    case PROBCUT :
        return select([&]() { return pos.see_ge(*cur, threshold); });
    }

    assert(false);
    return Move::none();  // Silence warning
}

void MovePicker::skip_quiet_moves() { skipQuiets = true; }

}  // namespace Stockfish
