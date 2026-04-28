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
#include <cstdint>
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


#if defined(USE_AVX512ICL)

void swap_with(__m512i& data, __m512i shuffled, __mmask8 Mask) {
    __m512i max = _mm512_max_epi64(shuffled, data);
    __m512i min = _mm512_min_epi64(shuffled, data);
    data        = _mm512_mask_mov_epi64(max, Mask, min);
}

template<int I0, int I1, int I2, int I3>
void sort_256(__m256i& data) {
    __m256i shuffled =
      _mm256_permutex_epi64(data, I0 << 2 * I1 | I1 << 2 * I0 | I2 << 2 * I3 | I3 << 2 * I2);
    constexpr __mmask8 Mask = 1 << I1 | 1 << I3;
    __m256i            max  = _mm256_max_epi64(shuffled, data);
    __m256i            min  = _mm256_min_epi64(shuffled, data);
    data                    = _mm256_mask_mov_epi64(max, Mask, min);
}

template<int I0, int I1, int I2, int I3>
void sort_512_256(__m512i& data) {
    __m512i shuffled =
      _mm512_permutex_epi64(data, I0 << 2 * I1 | I1 << 2 * I0 | I2 << 2 * I3 | I3 << 2 * I2);
    swap_with(data, shuffled, 0x11 << I1 | 0x11 << I3);
}

bool simd_sort(ExtMove* begin, ExtMove* end) {
    if (end - begin <= 1)
        return true;
    if (end - begin <= 4)
    {
        __mmask8 mask = (1 << (end - begin)) - 1;
        __m256i  data = _mm256_mask_loadu_epi64(_mm256_set1_epi64x(INT64_MIN), mask, begin);
        sort_256<0, 2, 1, 3>(data);
        sort_256<0, 1, 2, 3>(data);
        sort_256<1, 2, 0, 3>(data);
        _mm256_mask_storeu_epi64(begin, mask, data);
        return true;
    }
    if (end - begin <= 31)
    {
        __mmask8 mask = (__mmask8) ((1U << (end - begin)) - 1);
        __m512i  data = _mm512_mask_loadu_epi64(_mm512_set1_epi64(INT64_MIN), mask, begin);
        sort_512_256<0, 2, 1, 3>(data);
        swap_with(data, _mm512_shuffle_i64x2(data, data, 0b01001110), 0b11110000);
        sort_512_256<0, 1, 2, 3>(data);
        swap_with(data, _mm512_shuffle_i64x2(data, data, 0b11011000), 0b11110000);
        swap_with(data, _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 4, 2, 6, 1, 5, 3, 7), data),
                  0b11110000);
        swap_with(data, _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 2, 1, 4, 3, 6, 5, 7), data),
                  0b11010100);
        _mm512_mask_storeu_epi64(begin, mask, data);
        return end - begin <= 8;
    }
    return false;
}

#endif

sf_noinline void insertion_sort(ExtMove* begin, ExtMove* end) {
#if defined(USE_AVX512ICL)
    if (simd_sort(begin, end))
        return;
#endif
    for (ExtMove* p = begin + 1; p < end; ++p)
    {
        ExtMove tmp = *p, *q;
        for (q = p; q != begin && *(q - 1) < tmp; --q)
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
// For QUIETS, performs a partition during scoring: moves with value >= partition_quiets
// are written to the front (encounter order), the rest to the back (reverse encounter
// order). Returns the partition boundary.
template<GenType Type>
ExtMove* MovePicker::score(const MoveList<Type>& ml, [[maybe_unused]] int partition_quiets) {

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

    ExtMove*                  it  = cur;
    [[maybe_unused]] ExtMove* end = cur + ml.size() - 1;
    for (auto move : ml)
    {
        ExtMove m;
        m = move;

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

            int      should_sort = m.value >= partition_quiets;
            ExtMove* write       = should_sort ? it : end;
            it += should_sort;
            end -= !should_sort;
            *write = m;
            continue;
        }

        else  // Type == EVASIONS
        {
            if (pos.capture_stage(m))
                m.value = PieceValue[capturedPiece] + (1 << 28);
            else
                m.value = (*mainHistory)[us][m.raw()] + (*continuationHistory[0])[pc][to];
        }

        *it++ = m;
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

            endCur = endGenerated  = cur + ml.size();
            ExtMove* partition_end = score<QUIETS>(ml, -3560 * depth);
            insertion_sort(cur, partition_end);
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

        insertion_sort(cur, endCur);
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
