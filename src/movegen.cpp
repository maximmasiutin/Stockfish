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
#include <initializer_list>

#include "bitboard.h"
#include "position.h"

#if defined(USE_AVX512ICL) || defined(USE_AVX2)
    #include <array>
    #include <algorithm>
    #include <immintrin.h>
#endif

namespace Stockfish {

namespace {

#if defined(USE_AVX512ICL)

inline Move* write_moves(Move* moveList, uint32_t mask, __m512i vector) {
    // Avoid _mm512_mask_compressstoreu_epi16() as it's 256 uOps on Zen4
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(moveList),
                        _mm512_maskz_compress_epi16(mask, vector));
    return moveList + popcount(mask);
}

template<Direction offset>
inline Move* splat_pawn_moves(Move* moveList, Bitboard to_bb) {
    alignas(64) static constexpr auto SPLAT_TABLE = [] {
        std::array<Move, 64> table{};
        for (int i = 0; i < 64; i++)
        {
            Square from{uint8_t(std::clamp(i - offset, 0, 63))};
            table[i] = {Move(from, Square{uint8_t(i)})};
        }
        return table;
    }();

    auto table = reinterpret_cast<const __m512i*>(SPLAT_TABLE.data());

    moveList =
      write_moves(moveList, static_cast<uint32_t>(to_bb >> 0), _mm512_load_si512(table + 0));
    moveList =
      write_moves(moveList, static_cast<uint32_t>(to_bb >> 32), _mm512_load_si512(table + 1));

    return moveList;
}

inline Move* splat_moves(Move* moveList, Square from, Bitboard to_bb) {
    alignas(64) static constexpr auto SPLAT_TABLE = [] {
        std::array<Move, 64> table{};
        for (uint8_t i = 0; i < 64; i++)
            table[i] = {Move(SQUARE_ZERO, Square{i})};
        return table;
    }();

    __m512i fromVec = _mm512_set1_epi16(Move(from, SQUARE_ZERO).raw());

    auto table = reinterpret_cast<const __m512i*>(SPLAT_TABLE.data());

    moveList = write_moves(moveList, static_cast<uint32_t>(to_bb >> 0),
                           _mm512_or_si512(_mm512_load_si512(table + 0), fromVec));
    moveList = write_moves(moveList, static_cast<uint32_t>(to_bb >> 32),
                           _mm512_or_si512(_mm512_load_si512(table + 1), fromVec));

    return moveList;
}

#elif defined(USE_AVX2)

// Lookup table for compressing 8 x 16-bit elements based on 8-bit mask
// Each entry contains shuffle indices for _mm_shuffle_epi8
alignas(64) static constexpr auto COMPRESS_LUT = [] {
    std::array<std::array<uint8_t, 16>, 256> table{};
    for (int mask = 0; mask < 256; ++mask)
    {
        int k = 0;
        for (int i = 0; i < 8; ++i)
        {
            if (mask & (1 << i))
            {
                table[mask][k * 2]     = i * 2;
                table[mask][k * 2 + 1] = i * 2 + 1;
                ++k;
            }
        }
        // Fill remaining with zeros (doesn't matter, won't be used)
        for (; k < 8; ++k)
        {
            table[mask][k * 2]     = 0x80;  // Zero out unused positions
            table[mask][k * 2 + 1] = 0x80;
        }
    }
    return table;
}();

// Compress 16 x 16-bit elements from 256-bit vector based on 16-bit mask
inline Move* write_moves_avx2(Move* moveList, uint16_t mask, __m256i vector) {
    // Split into low and high 128-bit halves
    __m128i lo = _mm256_castsi256_si128(vector);
    __m128i hi = _mm256_extracti128_si256(vector, 1);

    uint8_t mask_lo = mask & 0xFF;
    uint8_t mask_hi = (mask >> 8) & 0xFF;

    int count_lo = popcount(mask_lo);
    int count_hi = popcount(mask_hi);

    // Compress low half
    __m128i shuf_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(COMPRESS_LUT[mask_lo].data()));
    __m128i compressed_lo = _mm_shuffle_epi8(lo, shuf_lo);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(moveList), compressed_lo);

    // Compress high half
    __m128i shuf_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(COMPRESS_LUT[mask_hi].data()));
    __m128i compressed_hi = _mm_shuffle_epi8(hi, shuf_hi);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(moveList + count_lo), compressed_hi);

    return moveList + count_lo + count_hi;
}

template<Direction offset>
inline Move* splat_pawn_moves(Move* moveList, Bitboard to_bb) {
    alignas(32) static constexpr auto SPLAT_TABLE = [] {
        std::array<Move, 64> table{};
        for (int i = 0; i < 64; i++)
        {
            Square from{uint8_t(std::clamp(i - offset, 0, 63))};
            table[i] = {Move(from, Square{uint8_t(i)})};
        }
        return table;
    }();

    auto table = reinterpret_cast<const __m256i*>(SPLAT_TABLE.data());

    // Process 16 squares at a time (256 bits / 16 bits per move)
    moveList = write_moves_avx2(moveList, static_cast<uint16_t>(to_bb >> 0),  _mm256_load_si256(table + 0));
    moveList = write_moves_avx2(moveList, static_cast<uint16_t>(to_bb >> 16), _mm256_load_si256(table + 1));
    moveList = write_moves_avx2(moveList, static_cast<uint16_t>(to_bb >> 32), _mm256_load_si256(table + 2));
    moveList = write_moves_avx2(moveList, static_cast<uint16_t>(to_bb >> 48), _mm256_load_si256(table + 3));

    return moveList;
}

inline Move* splat_moves(Move* moveList, Square from, Bitboard to_bb) {
    alignas(32) static constexpr auto SPLAT_TABLE = [] {
        std::array<Move, 64> table{};
        for (uint8_t i = 0; i < 64; i++)
            table[i] = {Move(SQUARE_ZERO, Square{i})};
        return table;
    }();

    __m256i fromVec = _mm256_set1_epi16(Move(from, SQUARE_ZERO).raw());

    auto table = reinterpret_cast<const __m256i*>(SPLAT_TABLE.data());

    moveList = write_moves_avx2(moveList, static_cast<uint16_t>(to_bb >> 0),
                                _mm256_or_si256(_mm256_load_si256(table + 0), fromVec));
    moveList = write_moves_avx2(moveList, static_cast<uint16_t>(to_bb >> 16),
                                _mm256_or_si256(_mm256_load_si256(table + 1), fromVec));
    moveList = write_moves_avx2(moveList, static_cast<uint16_t>(to_bb >> 32),
                                _mm256_or_si256(_mm256_load_si256(table + 2), fromVec));
    moveList = write_moves_avx2(moveList, static_cast<uint16_t>(to_bb >> 48),
                                _mm256_or_si256(_mm256_load_si256(table + 3), fromVec));

    return moveList;
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

}  // namespace Stockfish
