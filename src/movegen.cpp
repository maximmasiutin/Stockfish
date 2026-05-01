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

#include "movegen.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>

#include "bitboard.h"
#include "history.h"
#include "misc.h"
#include "position.h"

#if defined(USE_AVX512ICL)
    #include <array>
    #include <algorithm>
    #include <immintrin.h>
#endif

namespace Stockfish {

namespace {

#if defined(USE_AVX512ICL)

template<Direction offset>
inline Move* splat_pawn_moves(Move* moveList, Bitboard to_bb) {
    assert(popcount(to_bb) <= 8);  // <= 8 pawns per side

    const __m128i toSquares =
      _mm_cvtepi8_epi16(_mm512_castsi512_si128(_mm512_maskz_compress_epi8(to_bb, AllSquares)));
    const __m128i fromSquares = _mm_subs_epi16(toSquares, _mm_set1_epi16(offset));
    const __m128i moves       = _mm_or_si128(_mm_slli_epi16(fromSquares, Move::FromSqShift),
                                             _mm_slli_epi16(toSquares, Move::ToSqShift));

    _mm_storeu_si128(reinterpret_cast<__m128i*>(moveList), moves);
    return moveList + popcount(to_bb);
}

inline Move* splat_moves(Move* moveList, Square from, Bitboard to_bb) {
    assert(popcount(to_bb) <= 32);  // Q can attack up to 27 squares

    const __m512i fromVec = _mm512_set1_epi16(Move(from, SQUARE_ZERO).raw());
    const __m512i toSquares =
      _mm512_cvtepi8_epi16(_mm512_castsi512_si256(_mm512_maskz_compress_epi8(to_bb, AllSquares)));
    const __m512i moves = _mm512_or_si512(fromVec, _mm512_slli_epi16(toSquares, Move::ToSqShift));

    _mm512_storeu_si512(moveList, moves);
    return moveList + popcount(to_bb);
}

#else

template<Direction offset>
inline Move* splat_pawn_moves(Move* moveList, Bitboard to_bb) {
    while (to_bb)
    {
        Square to   = pop_lsb(to_bb);
        *moveList++ = Move(to - offset, to);
    }
    return moveList;
}

inline Move* splat_moves(Move* moveList, Square from, Bitboard to_bb) {
    while (to_bb)
        *moveList++ = Move(from, pop_lsb(to_bb));
    return moveList;
}

#endif

template<GenType Type, Direction D, bool Enemy>
Move* make_promotions(Move* moveList, [[maybe_unused]] Square to) {

    constexpr bool all = Type == EVASIONS || Type == NON_EVASIONS;

    if constexpr (Type == CAPTURES || all)
        *moveList++ = Move::make<PROMOTION>(to - D, to, QUEEN);

    if constexpr ((Type == CAPTURES && Enemy) || (Type == QUIETS && !Enemy) || all)
    {
        *moveList++ = Move::make<PROMOTION>(to - D, to, ROOK);
        *moveList++ = Move::make<PROMOTION>(to - D, to, BISHOP);
        *moveList++ = Move::make<PROMOTION>(to - D, to, KNIGHT);
    }

    return moveList;
}


template<Color Us, GenType Type>
Move* generate_pawn_moves(const Position& pos, Move* moveList, Bitboard target) {

    constexpr Color     Them     = ~Us;
    constexpr Bitboard  TRank7BB = (Us == WHITE ? Rank7BB : Rank2BB);
    constexpr Bitboard  TRank3BB = (Us == WHITE ? Rank3BB : Rank6BB);
    constexpr Direction Up       = pawn_push(Us);
    constexpr Direction UpRight  = (Us == WHITE ? NORTH_EAST : SOUTH_WEST);
    constexpr Direction UpLeft   = (Us == WHITE ? NORTH_WEST : SOUTH_EAST);

    const Bitboard emptySquares = ~pos.pieces();
    const Bitboard enemies      = Type == EVASIONS ? pos.checkers() : pos.pieces(Them);

    Bitboard pawnsOn7    = pos.pieces(Us, PAWN) & TRank7BB;
    Bitboard pawnsNotOn7 = pos.pieces(Us, PAWN) & ~TRank7BB;

    // Single and double pawn pushes, no promotions
    if constexpr (Type != CAPTURES)
    {
        Bitboard b1 = shift<Up>(pawnsNotOn7) & emptySquares;
        Bitboard b2 = shift<Up>(b1 & TRank3BB) & emptySquares;

        if constexpr (Type == EVASIONS)  // Consider only blocking squares
        {
            b1 &= target;
            b2 &= target;
        }

        moveList = splat_pawn_moves<Up>(moveList, b1);
        moveList = splat_pawn_moves<Up + Up>(moveList, b2);
    }

    // Promotions and underpromotions
    if (pawnsOn7)
    {
        Bitboard b1 = shift<UpRight>(pawnsOn7) & enemies;
        Bitboard b2 = shift<UpLeft>(pawnsOn7) & enemies;
        Bitboard b3 = shift<Up>(pawnsOn7) & emptySquares;

        if constexpr (Type == EVASIONS)
            b3 &= target;

        while (b1)
            moveList = make_promotions<Type, UpRight, true>(moveList, pop_lsb(b1));

        while (b2)
            moveList = make_promotions<Type, UpLeft, true>(moveList, pop_lsb(b2));

        while (b3)
            moveList = make_promotions<Type, Up, false>(moveList, pop_lsb(b3));
    }

    // Standard and en passant captures
    if constexpr (Type == CAPTURES || Type == EVASIONS || Type == NON_EVASIONS)
    {
        Bitboard b1 = shift<UpRight>(pawnsNotOn7) & enemies;
        Bitboard b2 = shift<UpLeft>(pawnsNotOn7) & enemies;

        moveList = splat_pawn_moves<UpRight>(moveList, b1);
        moveList = splat_pawn_moves<UpLeft>(moveList, b2);

        if (pos.ep_square() != SQ_NONE)
        {
            assert(rank_of(pos.ep_square()) == relative_rank(Us, RANK_6));

            // An en passant capture cannot resolve a discovered check
            if (Type == EVASIONS && (target & (pos.ep_square() + Up)))
                return moveList;

            b1 = pawnsNotOn7 & attacks_bb<PAWN>(pos.ep_square(), Them);

            assert(b1);

            while (b1)
                *moveList++ = Move::make<EN_PASSANT>(pop_lsb(b1), pos.ep_square());
        }
    }

    return moveList;
}


template<Color Us, PieceType Pt>
Move* generate_moves(const Position& pos, Move* moveList, Bitboard target) {

    static_assert(Pt != KING && Pt != PAWN, "Unsupported piece type in generate_moves()");

    Bitboard bb = pos.pieces(Us, Pt);

    while (bb)
    {
        Square   from = pop_lsb(bb);
        Bitboard b    = attacks_bb<Pt>(from, pos.pieces()) & target;

        moveList = splat_moves(moveList, from, b);
    }

    return moveList;
}


template<Color Us, GenType Type>
Move* generate_all(const Position& pos, Move* moveList) {

    static_assert(Type != LEGAL, "Unsupported type in generate_all()");

    const Square ksq = pos.square<KING>(Us);
    Bitboard     target;

    // Skip generating non-king moves when in double check
    if (Type != EVASIONS || !more_than_one(pos.checkers()))
    {
        target = Type == EVASIONS     ? between_bb(ksq, lsb(pos.checkers()))
               : Type == NON_EVASIONS ? ~pos.pieces(Us)
               : Type == CAPTURES     ? pos.pieces(~Us)
                                      : ~pos.pieces();  // QUIETS

        moveList = generate_pawn_moves<Us, Type>(pos, moveList, target);
        moveList = generate_moves<Us, KNIGHT>(pos, moveList, target);
        moveList = generate_moves<Us, BISHOP>(pos, moveList, target);
        moveList = generate_moves<Us, ROOK>(pos, moveList, target);
        moveList = generate_moves<Us, QUEEN>(pos, moveList, target);
    }

    Bitboard b = attacks_bb<KING>(ksq) & (Type == EVASIONS ? ~pos.pieces(Us) : target);

    moveList = splat_moves(moveList, ksq, b);

    if ((Type == QUIETS || Type == NON_EVASIONS) && pos.can_castle(Us & ANY_CASTLING))
        for (CastlingRights cr : {Us & KING_SIDE, Us & QUEEN_SIDE})
            if (!pos.castling_impeded(cr) && pos.can_castle(cr))
                *moveList++ = Move::make<CASTLING>(ksq, pos.castling_rook_square(cr));

    return moveList;
}

}  // namespace


// <CAPTURES>     Generates all pseudo-legal captures plus queen promotions
// <QUIETS>       Generates all pseudo-legal non-captures and underpromotions
// <EVASIONS>     Generates all pseudo-legal check evasions
// <NON_EVASIONS> Generates all pseudo-legal captures and non-captures
//
// Returns a pointer to the end of the move list.
template<GenType Type>
Move* generate(const Position& pos, Move* moveList) {

    static_assert(Type != LEGAL, "Unsupported type in generate()");
    assert((Type == EVASIONS) == bool(pos.checkers()));

    Color us = pos.side_to_move();

    return us == WHITE ? generate_all<WHITE, Type>(pos, moveList)
                       : generate_all<BLACK, Type>(pos, moveList);
}

// Explicit template instantiations
template Move* generate<CAPTURES>(const Position&, Move*);
template Move* generate<QUIETS>(const Position&, Move*);
template Move* generate<EVASIONS>(const Position&, Move*);
template Move* generate<NON_EVASIONS>(const Position&, Move*);

// generate<LEGAL> generates all the legal moves in the given position

template<>
Move* generate<LEGAL>(const Position& pos, Move* moveList) {

    Color    us     = pos.side_to_move();
    Bitboard pinned = pos.blockers_for_king(us) & pos.pieces(us);
    Square   ksq    = pos.square<KING>(us);
    Move*    cur    = moveList;

    moveList =
      pos.checkers() ? generate<EVASIONS>(pos, moveList) : generate<NON_EVASIONS>(pos, moveList);
    while (cur != moveList)
        if (((pinned & cur->from_sq()) || cur->from_sq() == ksq || cur->type_of() == EN_PASSANT)
            && !pos.legal(*cur))
            *cur = *(--moveList);
        else
            ++cur;

    return moveList;
}

// Cold tail of main_hist_freq_index. Only entered for non-NORMAL move types
// (PROMOTION, EN_PASSANT, CASTLING) since the hot path's `r >= 0x4000u` gate
// catches those. Sentinels (raw=0, raw=65) are NORMAL-typed and route through
// the inline path; the call sites in movepick/search are guarded upstream by
// is_ok() so sentinels never reach mainHistory access.
sf_noinline std::size_t main_hist_freq_index_special(std::uint32_t r) {
    const std::uint32_t type = (r >> 14) & 3u;
    const std::uint32_t from = (r >> 6) & 0x3Fu;
    const std::uint32_t to   = r & 0x3Fu;
    const std::uint32_t ff   = from & 7u;
    const std::uint32_t tf   = to & 7u;
    const std::uint32_t fr   = from >> 3;
    // half = from >> 5 splits ranks 0-3 (0) vs 4-7 (1). Matches WHITE/BLACK for
    // CASTLING; reversed for PROMOTION (white promotes from rank 6 -> half=1).
    // Used as a bijection bit only.
    const std::uint32_t half = from >> 5;

    if (type == 1u)
    {
        const std::uint32_t promo = (r >> 12) & 3u;
        return 4704u + (half << 5) + ff * 4u + promo;
    }
    if (type == 3u)
        return 4768u + half * 64u + ff * 8u + tf;

    // EN_PASSANT (type == 2): from rank 4 (white) or rank 3 (black).
    const std::uint32_t ep_half = std::uint32_t(fr >= 4u);
    const std::uint32_t dir     = std::uint32_t(tf > ff);
    return 4896u + ep_half * 16u + tf * 2u + dir;
}

// Frequency-aware mainHistory index. Both inner and outer marked sf_noinline
// to force fully out-of-line codegen at every call site, isolating
// "function-call-cost only" from "inline-bloat".
// See research/mainhist-fa-attr-and-cache-experiments.md.
sf_noinline std::size_t main_hist_freq_index(Move m) {
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

    // PAWN_MASK encodes the 12 legal NORMAL pawn pushes as a 64-bit bitmap
    // indexed by (fr * 8 + tr): bit set iff a pawn can move from rank fr to
    // rank tr in a single NORMAL move (init single + double pushes from
    // ranks 1 and 6, continuation single pushes from ranks 2..5 in either
    // direction). Set bits at positions {10,11,17,19,26,28,35,37,44,46,52,53}
    // = 0x305028140a0c00. Captures and promotions are routed elsewhere; this
    // mask only matters for same-file moves which can only be pawn pushes.
    constexpr std::uint64_t PAWN_MASK = 0x00305028140a0c00ULL;
    const std::uint32_t     same_file = std::uint32_t(((from ^ to) & 7u) == 0u);
    const std::uint32_t pawn_hit  = same_file & std::uint32_t((PAWN_MASK >> (fr * 8u + tr)) & 1u);
    const std::uint32_t pawn_mask = std::uint32_t(0u) - pawn_hit;

    const std::uint32_t pawn_color = std::uint32_t(tr < fr);
    const std::uint32_t is_init    = std::uint32_t((0x42u >> fr) & 1u);
    const std::uint32_t init_mask  = std::uint32_t(0u) - is_init;

    const std::uint32_t is_double = std::uint32_t(((tr ^ fr) & 3u) == 2u);
    const std::uint32_t init_slot = (pawn_color << 4u) + (ff << 1u) + is_double;
    const std::uint32_t cont_slot = 32u + (pawn_color << 5u) + (ff << 2u) + (fr - 2u);
    const std::uint32_t pawn_slot = (init_slot & init_mask) | (cont_slot & ~init_mask);

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

}  // namespace Stockfish
