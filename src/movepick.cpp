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


// Sort moves in descending order up to and including a given limit.
// The order of moves smaller than the limit is left unspecified.
void partial_insertion_sort(ExtMove* begin, ExtMove* end, int limit) {

    for (ExtMove *sortedEnd = begin, *p = begin + 1; p < end; ++p)
        if (p->value >= limit)
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

// Stable descending insertion sorter for up to 32 ExtMoves; lo holds the
// 16 largest, hi holds the next 16, with cross-register propagation lo[15] -> hi[0].
struct SimdSorter {
    static constexpr int MAX_ELEMENTS = 32;
    __m512i              sorted_values_lo, sorted_moves_lo;
    __m512i              sorted_values_hi, sorted_moves_hi;

    explicit SimdSorter(const ExtMove& first) {
        splat_extmove(first, sorted_moves_lo, sorted_values_lo);
        sorted_values_lo = _mm512_mask_set1_epi32(sorted_values_lo, __mmask16(0xFFFE),
                                                  std::numeric_limits<int>::min());
        sorted_values_hi = _mm512_set1_epi32(std::numeric_limits<int>::min());
        sorted_moves_hi  = _mm512_setzero_si512();
    }

    void insert(const ExtMove& m) {
        __m512i move, value;
        splat_extmove(m, move, value);

        const __mmask16 to_right_lo = _mm512_cmplt_epi32_mask(sorted_values_lo, value);
        const __mmask16 to_right_hi = _mm512_cmplt_epi32_mask(sorted_values_hi, value);

        // Extract OLD lo[15] before the lo expand overwrites it.
        const __m512i idx15  = _mm512_set1_epi32(15);
        const __m512i lo15_v = _mm512_permutexvar_epi32(idx15, sorted_values_lo);
        const __m512i lo15_m = _mm512_permutexvar_epi32(idx15, sorted_moves_lo);

        // Bit 15 of to_right_lo: 1 = v displaces lo[15] (v_in_lo), 0 = v in hi;
        // blend mask flips it so lane 0 of src_v_hi picks lo15 vs v accordingly.
        const __mmask16 v_in_lo_bit = _kshiftri_mask16(to_right_lo, 15);
        const __mmask16 blend_mask  = _kxor_mask16(__mmask16(0xFFFF), v_in_lo_bit);
        const __m512i   src_v_hi    = _mm512_mask_blend_epi32(blend_mask, lo15_v, value);
        const __m512i   src_m_hi    = _mm512_mask_blend_epi32(blend_mask, lo15_m, move);

        const __mmask16 expand_lo = _kadd_mask16(to_right_lo, __mmask16(-1));
        const __mmask16 expand_hi = _kadd_mask16(to_right_hi, __mmask16(-1));

        sorted_values_hi = _mm512_mask_expand_epi32(src_v_hi, expand_hi, sorted_values_hi);
        sorted_moves_hi  = _mm512_mask_expand_epi32(src_m_hi, expand_hi, sorted_moves_hi);
        sorted_values_lo = _mm512_mask_expand_epi32(value, expand_lo, sorted_values_lo);
        sorted_moves_lo  = _mm512_mask_expand_epi32(move, expand_lo, sorted_moves_lo);
    }

    void write_sorted(ExtMove* dst, int count) const {
        static_assert(sizeof(ExtMove) == 8);
        const __m512i interleave_lo =
          _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
        const __m512i interleave_hi =
          _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
        if (count > 0)
        {
            const __m512i v =
              _mm512_permutex2var_epi32(sorted_moves_lo, interleave_lo, sorted_values_lo);
            _mm512_mask_storeu_epi64(dst, __mmask8((1U << std::min(count, 8)) - 1), v);
        }
        if (count > 8)
        {
            const __m512i v =
              _mm512_permutex2var_epi32(sorted_moves_lo, interleave_hi, sorted_values_lo);
            _mm512_mask_storeu_epi64(dst + 8, __mmask8((1U << std::min(count - 8, 8)) - 1), v);
        }
        if (count > 16)
        {
            const __m512i v =
              _mm512_permutex2var_epi32(sorted_moves_hi, interleave_lo, sorted_values_hi);
            _mm512_mask_storeu_epi64(dst + 16, __mmask8((1U << std::min(count - 16, 8)) - 1), v);
        }
        if (count > 24)
        {
            const __m512i v =
              _mm512_permutex2var_epi32(sorted_moves_hi, interleave_hi, sorted_values_hi);
            _mm512_mask_storeu_epi64(dst + 24, __mmask8((1U << std::min(count - 24, 8)) - 1), v);
        }
    }
};

#endif

// Descending sort for the QUIETS site; SIMD prefix is byte-equivalent to the
// scalar loop and falls through to the scalar tail past MAX_ELEMENTS.
#if defined(USE_AVX512ICL)
    #define SF_SORT_LONG_NOINLINE sf_noinline
#else
    #define SF_SORT_LONG_NOINLINE
#endif

SF_SORT_LONG_NOINLINE void partial_insertion_sort_long(ExtMove* begin, ExtMove* end, int limit) {

    ExtMove* sortedEnd = begin;
    ExtMove* p         = begin + 1;

#if defined(USE_AVX512ICL)
    if (p < end)
    {
        SimdSorter sorter(*begin);
        for (; p < end; ++p)
            if (p->value >= limit)
            {
                if (sortedEnd - begin + 1 >= SimdSorter::MAX_ELEMENTS)
                    break;
                sorter.insert(*p);
                *p = *++sortedEnd;
            }
        sorter.write_sorted(begin, int(sortedEnd - begin + 1));
    }
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

#undef SF_SORT_LONG_NOINLINE

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

        partial_insertion_sort(cur, endCur, std::numeric_limits<int>::min());
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

        partial_insertion_sort(cur, endCur, std::numeric_limits<int>::min());
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
