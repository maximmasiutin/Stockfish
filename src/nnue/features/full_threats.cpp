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

//Definition of input features FullThreats of NNUE evaluation function

#include "full_threats.h"

#include <array>
#include <cstdint>
#include <initializer_list>
#include <utility>

#ifdef USE_AVX512
    #include <immintrin.h>
#endif

#include "../../bitboard.h"
#include "../../misc.h"
#include "../../position.h"
#include "../../types.h"
#include "../nnue_common.h"

namespace Stockfish::Eval::NNUE::Features {

struct HelperOffsets {
    int cumulativePieceOffset, cumulativeOffset;
};

constexpr std::array<Piece, 12> AllPieces = {
  W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
  B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING,
};

template<PieceType PT>
constexpr auto make_piece_indices_type() {
    static_assert(PT != PieceType::PAWN);

    std::array<std::array<uint8_t, SQUARE_NB>, SQUARE_NB> out{};

    for (Square from = SQ_A1; from <= SQ_H8; ++from)
    {
        Bitboard attacks = PseudoAttacks[PT][from];

        for (Square to = SQ_A1; to <= SQ_H8; ++to)
        {
            out[from][to] = constexpr_popcount(((1ULL << to) - 1) & attacks);
        }
    }

    return out;
}

template<Piece P>
constexpr auto make_piece_indices_piece() {
    static_assert(type_of(P) == PieceType::PAWN);

    std::array<std::array<uint8_t, SQUARE_NB>, SQUARE_NB> out{};

    constexpr Color C = color_of(P);

    for (Square from = SQ_A1; from <= SQ_H8; ++from)
    {
        Bitboard attacks = PseudoAttacks[C][from];

        for (Square to = SQ_A1; to <= SQ_H8; ++to)
        {
            out[from][to] = constexpr_popcount(((1ULL << to) - 1) & attacks);
        }
    }

    return out;
}

constexpr auto index_lut2_array() {
    constexpr auto KNIGHT_ATTACKS = make_piece_indices_type<PieceType::KNIGHT>();
    constexpr auto BISHOP_ATTACKS = make_piece_indices_type<PieceType::BISHOP>();
    constexpr auto ROOK_ATTACKS   = make_piece_indices_type<PieceType::ROOK>();
    constexpr auto QUEEN_ATTACKS  = make_piece_indices_type<PieceType::QUEEN>();
    constexpr auto KING_ATTACKS   = make_piece_indices_type<PieceType::KING>();

    std::array<std::array<std::array<uint8_t, SQUARE_NB>, SQUARE_NB>, PIECE_NB> indices{};

    indices[W_PAWN] = make_piece_indices_piece<W_PAWN>();
    indices[B_PAWN] = make_piece_indices_piece<B_PAWN>();

    indices[W_KNIGHT] = KNIGHT_ATTACKS;
    indices[B_KNIGHT] = KNIGHT_ATTACKS;

    indices[W_BISHOP] = BISHOP_ATTACKS;
    indices[B_BISHOP] = BISHOP_ATTACKS;

    indices[W_ROOK] = ROOK_ATTACKS;
    indices[B_ROOK] = ROOK_ATTACKS;

    indices[W_QUEEN] = QUEEN_ATTACKS;
    indices[B_QUEEN] = QUEEN_ATTACKS;

    indices[W_KING] = KING_ATTACKS;
    indices[B_KING] = KING_ATTACKS;

    return indices;
}

constexpr auto init_threat_offsets() {
    std::array<HelperOffsets, PIECE_NB>                    indices{};
    std::array<std::array<IndexType, SQUARE_NB>, PIECE_NB> offsets{};

    int cumulativeOffset = 0;
    for (Piece piece : AllPieces)
    {
        int pieceIdx              = piece;
        int cumulativePieceOffset = 0;

        for (Square from = SQ_A1; from <= SQ_H8; ++from)
        {
            offsets[pieceIdx][from] = cumulativePieceOffset;

            if (type_of(piece) != PAWN)
            {
                Bitboard attacks = PseudoAttacks[type_of(piece)][from];
                cumulativePieceOffset += constexpr_popcount(attacks);
            }

            else if (from >= SQ_A2 && from <= SQ_H7)
            {
                Bitboard attacks = (pieceIdx < 8) ? pawn_attacks_bb<WHITE>(square_bb(from))
                                                  : pawn_attacks_bb<BLACK>(square_bb(from));
                cumulativePieceOffset += constexpr_popcount(attacks);
            }
        }

        indices[pieceIdx] = {cumulativePieceOffset, cumulativeOffset};

        cumulativeOffset += numValidTargets[pieceIdx] * cumulativePieceOffset;
    }

    return std::pair{indices, offsets};
}

constexpr auto helper_offsets = init_threat_offsets().first;
// Lookup array for indexing threats
constexpr auto offsets = init_threat_offsets().second;

constexpr auto init_index_luts() {
    std::array<std::array<std::array<uint32_t, 2>, PIECE_NB>, PIECE_NB> indices{};

    for (Piece attacker : AllPieces)
    {
        for (Piece attacked : AllPieces)
        {
            bool      enemy        = (attacker ^ attacked) == 8;
            PieceType attackerType = type_of(attacker);
            PieceType attackedType = type_of(attacked);

            int  map           = FullThreats::map[attackerType - 1][attackedType - 1];
            bool semi_excluded = attackerType == attackedType && (enemy || attackerType != PAWN);
            IndexType feature  = helper_offsets[attacker].cumulativeOffset
                              + (color_of(attacked) * (numValidTargets[attacker] / 2) + map)
                                  * helper_offsets[attacker].cumulativePieceOffset;

            bool excluded                  = map < 0;
            indices[attacker][attacked][0] = excluded ? FullThreats::Dimensions : feature;
            indices[attacker][attacked][1] =
              excluded || semi_excluded ? FullThreats::Dimensions : feature;
        }
    }

    return indices;
}

// The final index is calculated from summing data found in these two LUTs, as well
// as offsets[attacker][from]

// [attacker][attacked][from < to]
constexpr auto index_lut1 = init_index_luts();
// [attacker][from][to]
// Padded struct: i32gather reads 4 bytes per lane at byte offsets, so the last
// element (offset 65535) reads 3 bytes past the 65536-byte array end.
struct IndexLut2Padded {
    std::array<std::array<std::array<uint8_t, SQUARE_NB>, SQUARE_NB>, PIECE_NB> lut;
    uint8_t                                                                     _pad[4]{};

    constexpr const auto& operator[](std::size_t i) const { return lut[i]; }
};
constexpr IndexLut2Padded index_lut2 = {index_lut2_array()};

// Index of a feature for a given king position and another piece on some square
inline sf_always_inline IndexType FullThreats::make_index(
  Color perspective, Piece attacker, Square from, Square to, Piece attacked, Square ksq) {
    const std::int8_t orientation   = OrientTBL[ksq] ^ (56 * perspective);
    unsigned          from_oriented = uint8_t(from) ^ orientation;
    unsigned          to_oriented   = uint8_t(to) ^ orientation;

    std::int8_t swap              = 8 * perspective;
    unsigned    attacker_oriented = attacker ^ swap;
    unsigned    attacked_oriented = attacked ^ swap;

    return index_lut1[attacker_oriented][attacked_oriented][from_oriented < to_oriented]
         + offsets[attacker_oriented][from_oriented]
         + index_lut2[attacker_oriented][from_oriented][to_oriented];
}

// Get a list of indices for active features in ascending order

void FullThreats::append_active_indices(Color perspective, const Position& pos, IndexList& active) {
    Square   ksq      = pos.square<KING>(perspective);
    Bitboard occupied = pos.pieces();

    for (Color color : {WHITE, BLACK})
    {
        for (PieceType pt = PAWN; pt <= KING; ++pt)
        {
            Color    c        = Color(perspective ^ color);
            Piece    attacker = make_piece(c, pt);
            Bitboard bb       = pos.pieces(c, pt);

            if (pt == PAWN)
            {
                auto right = (c == WHITE) ? NORTH_EAST : SOUTH_WEST;
                auto left  = (c == WHITE) ? NORTH_WEST : SOUTH_EAST;
                auto attacks_left =
                  ((c == WHITE) ? shift<NORTH_EAST>(bb) : shift<SOUTH_WEST>(bb)) & occupied;
                auto attacks_right =
                  ((c == WHITE) ? shift<NORTH_WEST>(bb) : shift<SOUTH_EAST>(bb)) & occupied;

                while (attacks_left)
                {
                    Square    to       = pop_lsb(attacks_left);
                    Square    from     = to - right;
                    Piece     attacked = pos.piece_on(to);
                    IndexType index    = make_index(perspective, attacker, from, to, attacked, ksq);

                    if (index < Dimensions)
                        active.push_back(index);
                }

                while (attacks_right)
                {
                    Square    to       = pop_lsb(attacks_right);
                    Square    from     = to - left;
                    Piece     attacked = pos.piece_on(to);
                    IndexType index    = make_index(perspective, attacker, from, to, attacked, ksq);

                    if (index < Dimensions)
                        active.push_back(index);
                }
            }
            else
            {
                while (bb)
                {
                    Square   from    = pop_lsb(bb);
                    Bitboard attacks = (attacks_bb(pt, from, occupied)) & occupied;

                    while (attacks)
                    {
                        Square    to       = pop_lsb(attacks);
                        Piece     attacked = pos.piece_on(to);
                        IndexType index =
                          make_index(perspective, attacker, from, to, attacked, ksq);

                        if (index < Dimensions)
                            active.push_back(index);
                    }
                }
            }
        }
    }
}

// Get a list of indices for recently changed features

void FullThreats::append_changed_indices(Color            perspective,
                                         Square           ksq,
                                         const DiffType&  diff,
                                         IndexList&       removed,
                                         IndexList&       added,
                                         FusedUpdateData* fusedData,
                                         bool             first) {

#ifdef USE_AVX512
    if (!fusedData)
    {
        const std::int8_t orientation = OrientTBL[ksq] ^ (56 * perspective);
        const std::int8_t swap        = 8 * perspective;
        const int         n           = diff.list.ssize();

        const __m512i orient_v = _mm512_set1_epi32(orientation);
        const __m512i swap_v   = _mm512_set1_epi32(swap);
        const __m512i dim_v    = _mm512_set1_epi32(Dimensions);
        const __m512i mask8    = _mm512_set1_epi32(0xFF);
        const __m512i mask4    = _mm512_set1_epi32(0xF);
        const __m512i bit31_v  = _mm512_set1_epi32(1u << 31);
        const __m512i ones     = _mm512_set1_epi32(1);

        const void* lut1_base = &index_lut1[0][0][0];
        const void* lut2_base = &index_lut2.lut[0][0][0];
        const void* off_base  = &offsets[0][0];

        int i = 0;
        for (; i + 16 <= n; i += 16)
        {
            // Load 16 DirtyThreat raw uint32 values
            __m512i raw = _mm512_loadu_si512(diff.list.begin() + i);

            // Extract fields: from(0:7), to(8:15), attacked(16:19), attacker(20:23)
            __m512i from     = _mm512_and_si512(raw, mask8);
            __m512i to       = _mm512_and_si512(_mm512_srli_epi32(raw, 8), mask8);
            __m512i attacked = _mm512_and_si512(_mm512_srli_epi32(raw, 16), mask4);
            __m512i attacker = _mm512_and_si512(_mm512_srli_epi32(raw, 20), mask4);

            // Apply orientation and color swap
            __m512i from_o     = _mm512_xor_si512(from, orient_v);
            __m512i to_o       = _mm512_xor_si512(to, orient_v);
            __m512i attacker_o = _mm512_xor_si512(attacker, swap_v);
            __m512i attacked_o = _mm512_xor_si512(attacked, swap_v);

            // lut1_idx = attacker_o*32 + attacked_o*2 + (from_o < to_o)
            __mmask16 cmp_lt = _mm512_cmplt_epu32_mask(from_o, to_o);
            __m512i   lut1_idx =
              _mm512_add_epi32(_mm512_slli_epi32(attacker_o, 5), _mm512_slli_epi32(attacked_o, 1));
            lut1_idx = _mm512_mask_add_epi32(lut1_idx, cmp_lt, lut1_idx, ones);

            // off_idx = attacker_o*64 + from_o
            __m512i off_idx = _mm512_add_epi32(_mm512_slli_epi32(attacker_o, 6), from_o);

            // lut2_idx = attacker_o*4096 + from_o*64 + to_o
            __m512i lut2_idx = _mm512_add_epi32(
              _mm512_add_epi32(_mm512_slli_epi32(attacker_o, 12), _mm512_slli_epi32(from_o, 6)),
              to_o);

            // Gather from 3 LUTs (independent, can pipeline)
            __m512i g1 = _mm512_i32gather_epi32(lut1_idx, lut1_base, 4);
            __m512i g2 = _mm512_i32gather_epi32(off_idx, off_base, 4);
            __m512i g3 = _mm512_i32gather_epi32(lut2_idx, lut2_base, 1);
            g3         = _mm512_and_si512(g3, mask8);

            // Final index = g1 + g2 + g3
            __m512i result = _mm512_add_epi32(_mm512_add_epi32(g1, g2), g3);

            // Valid entries: result < Dimensions
            __mmask16 valid = _mm512_cmplt_epu32_mask(result, dim_v);

            // Separate adds (bit 31 set) from removes
            __mmask16 is_add    = _mm512_test_epi32_mask(raw, bit31_v);
            __mmask16 add_valid = valid & is_add;
            __mmask16 rem_valid = valid & ~is_add;

            int add_count = _mm_popcnt_u32(add_valid);
            if (add_count)
            {
                auto* dst = added.make_space(add_count);
                _mm512_storeu_si512(dst, _mm512_maskz_compress_epi32(add_valid, result));
            }

            int rem_count = _mm_popcnt_u32(rem_valid);
            if (rem_count)
            {
                auto* dst = removed.make_space(rem_count);
                _mm512_storeu_si512(dst, _mm512_maskz_compress_epi32(rem_valid, result));
            }
        }

        // Scalar tail for remaining < 16 entries
        for (; i < n; i++)
        {
            const auto&     dirty  = diff.list[i];
            auto&           insert = dirty.add() ? added : removed;
            const IndexType idx    = make_index(perspective, dirty.pc(), dirty.pc_sq(),
                                                dirty.threatened_sq(), dirty.threatened_pc(), ksq);

            if (idx < Dimensions)
                insert.push_back(idx);
        }

        return;
    }
#endif

    for (const auto& dirty : diff.list)
    {
        auto attacker = dirty.pc();
        auto attacked = dirty.threatened_pc();
        auto from     = dirty.pc_sq();
        auto to       = dirty.threatened_sq();
        auto add      = dirty.add();

        if (fusedData)
        {
            if (from == fusedData->dp2removed)
            {
                if (add)
                {
                    if (first)
                    {
                        fusedData->dp2removedOriginBoard |= to;
                        continue;
                    }
                }
                else if (fusedData->dp2removedOriginBoard & to)
                    continue;
            }

            if (to != SQ_NONE && to == fusedData->dp2removed)
            {
                if (add)
                {
                    if (first)
                    {
                        fusedData->dp2removedTargetBoard |= from;
                        continue;
                    }
                }
                else if (fusedData->dp2removedTargetBoard & from)
                    continue;
            }
        }

        auto&           insert = add ? added : removed;
        const IndexType index  = make_index(perspective, attacker, from, to, attacked, ksq);

        if (index < Dimensions)
            insert.push_back(index);
    }
}

bool FullThreats::requires_refresh(const DiffType& diff, Color perspective) {
    return perspective == diff.us && (int8_t(diff.ksq) & 0b100) != (int8_t(diff.prevKsq) & 0b100);
}

}  // namespace Stockfish::Eval::NNUE::Features
