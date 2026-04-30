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

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>

#include "bitboard.h"
#include "history.h"
#include "misc.h"
#include "position.h"
#include "types.h"

#if defined(USE_AVX512ICL)
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

// Cold tier-dispatch path for the freq-aware LowPlyHistory address.
// Reached only on a hot-set miss (the inline outer in history.h handles hits).
// Branches reordered so the most common case (NORMAL same-file pawn push)
// returns first; rare cases fall through.
sf_noinline std::size_t low_ply_freq_index_slow(Move m) {
    const std::uint32_t r = m.raw();

    // NORMAL move (top 2 bits of raw == 0): hot path.
    if (r < 0x4000u)
    {
        const std::uint32_t from = (r >> 6) & 0x3Fu;
        const std::uint32_t to   = r & 0x3Fu;
        const std::uint32_t ff   = from & 7u;
        const std::uint32_t tf   = to & 7u;

        // Same-file => pawn-push candidate (tier 0 / tier 1).
        if (ff == tf)
        {
            const std::uint32_t fr = from >> 3;
            const std::uint32_t tr = to >> 3;

            // White single push (fr in [1,5], tr=fr+1): tier 0 if fr=1, else tier 1.
            if (fr >= 1u && fr <= 5u && tr == fr + 1u)
            {
                if (fr == 1u)
                    return ff * 2u;
                return 32u + ff * 4u + (fr - 2u);
            }
            // Black single push (fr in [2,6], tr=fr-1): tier 0 if fr=6, else tier 1.
            if (fr >= 2u && fr <= 6u && tr + 1u == fr)
            {
                if (fr == 6u)
                    return 16u + ff * 2u;
                return 64u + ff * 4u + (fr - 2u);
            }
            // White init double push (fr=1, tr=3).
            if (fr == 1u && tr == 3u)
                return ff * 2u + 1u;
            // Black init double push (fr=6, tr=4).
            if (fr == 6u && tr == 4u)
                return 16u + ff * 2u + 1u;
        }

        // Tier 2: back-rank chebyshev-1 single-step (king and adjacent piece moves).
        const std::uint32_t fr = from >> 3;
        if (fr == 0u || fr == 7u)
        {
            const std::uint32_t tr    = to >> 3;
            const int           fdiff = static_cast<int>(tf) - static_cast<int>(ff);
            const int           rdiff = static_cast<int>(tr) - static_cast<int>(fr);
            if (fdiff >= -1 && fdiff <= 1 && rdiff >= -1 && rdiff <= 1
                && (fdiff != 0 || rdiff != 0))
            {
                // half = fr >> 2: 0 for back rank 1, 1 for back rank 8 (not the WHITE/BLACK enum).
                // dest = (fdiff+1)*3 + (rdiff+1) is sparse in 0..8 (center 4 excluded).
                // Stride 8 is collision-free per half: fr=0 cannot yield rdiff=-1 and
                // fr=7 cannot yield rdiff=+1, so each half realizes at most 8 dest values.
                const std::uint32_t half = fr >> 2;
                const std::uint32_t dest = static_cast<std::uint32_t>(fdiff + 1) * 3u
                                         + static_cast<std::uint32_t>(rdiff + 1);
                return 96u + half * 64u + ff * 8u + dest;
            }
        }

        // Tier 3: rest of NORMAL with (from << 6) | to ordering.
        return 224u + ((from << 6) | to);
    }

    const std::uint32_t type = (r >> 14) & 3u;
    // PROMOTION: QUIETS-only invariant -> promo in {0,1,2} (queen goes via
    // CAPTURES); file*3+promo packs into [0, 24). half = from>>5
    // (board-half index 0 or 1, not the WHITE/BLACK enum).
    if (type == 1u)
    {
        const std::uint32_t from  = (r >> 6) & 0x3Fu;
        const std::uint32_t promo = (r >> 12) & 3u;
        const std::uint32_t half  = from >> 5;
        const std::uint32_t file  = from & 7u;
        assert(promo < 3u);
        return 4320u + (half << 5) + file * 3u + promo;
    }

    // CASTLING (type=3): Chess960-safe side bit via file comparison. EN_PASSANT
    // (type=2) is generated only by CAPTURES/EVASIONS/NON_EVASIONS movegen and
    // never reaches this QUIETS-only path.
    assert(type == 3u);
    const std::uint32_t from = (r >> 6) & 0x3Fu;
    const std::uint32_t to   = r & 0x3Fu;
    const std::uint32_t half = from >> 5;
    const std::uint32_t side = static_cast<std::uint32_t>((to & 7u) > (from & 7u));
    return 4384u + half * 2u + side;
}

}  // namespace Stockfish
