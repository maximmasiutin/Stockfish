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
constexpr auto index_lut2 = index_lut2_array();

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
        const int count = diff.list.ssize();
        if (count == 0)
            return;

        // Precompute constants (same for all entries in this call)
        const std::int8_t orientation = OrientTBL[ksq] ^ (56 * perspective);
        const std::int8_t swap_val    = 8 * perspective;

        const __m512i zmm_orient   = _mm512_set1_epi32(orientation);
        const __m512i zmm_swap     = _mm512_set1_epi32(swap_val);
        const __m512i zmm_dim      = _mm512_set1_epi32(Dimensions);
        const __m512i zmm_0xff     = _mm512_set1_epi32(0xFF);
        const __m512i zmm_0xf      = _mm512_set1_epi32(0xF);
        const __m512i zmm_one      = _mm512_set1_epi32(1);
        const __m512i zmm_sign_bit = _mm512_set1_epi32(int(0x80000000u));

        const auto* raw_ptr   = reinterpret_cast<const int32_t*>(diff.list.begin());
        const auto* lut1_base = reinterpret_cast<const int32_t*>(&index_lut1);
        const auto* off_base  = reinterpret_cast<const int32_t*>(&offsets);
        const auto* lut2_base = reinterpret_cast<const int8_t*>(&index_lut2);

        for (int i = 0; i < count; i += 16)
        {
            const int       batch    = (count - i < 16) ? count - i : 16;
            const __mmask16 k_active = (__mmask16) ((1u << batch) - 1);

            // Load 16 packed DirtyThreat entries (zeroed beyond count)
            __m512i zmm_raw = _mm512_maskz_loadu_epi32(k_active, raw_ptr + i);

            // Extract fields: from[7:0], to[15:8], attacked[19:16], attacker[23:20]
            __m512i zmm_from     = _mm512_and_epi32(zmm_raw, zmm_0xff);
            __m512i zmm_to       = _mm512_and_epi32(_mm512_srli_epi32(zmm_raw, 8), zmm_0xff);
            __m512i zmm_attacked = _mm512_and_epi32(_mm512_srli_epi32(zmm_raw, 16), zmm_0xf);
            __m512i zmm_attacker = _mm512_and_epi32(_mm512_srli_epi32(zmm_raw, 20), zmm_0xf);

            // Orient squares and pieces
            __m512i zmm_from_o = _mm512_xor_epi32(zmm_from, zmm_orient);
            __m512i zmm_to_o   = _mm512_xor_epi32(zmm_to, zmm_orient);
            __m512i zmm_atk_o  = _mm512_xor_epi32(zmm_attacker, zmm_swap);
            __m512i zmm_atkd_o = _mm512_xor_epi32(zmm_attacked, zmm_swap);

            // from_oriented < to_oriented -> 0 or 1
            __mmask16 k_cmp   = _mm512_cmplt_epu32_mask(zmm_from_o, zmm_to_o);
            __m512i   zmm_cmp = _mm512_maskz_mov_epi32(k_cmp, zmm_one);

            __m512i zmm_result;

    #ifdef USE_AVX512ICL
            // Check if all active entries share same attacker and from_sq.
            // Outgoing threats from a single write_multiple_dirties call are
            // contiguous and share these fields, enabling VPERMB/VPERMI2D.
            const __m512i key_mask  = _mm512_set1_epi32(0x00F000FF);
            __m512i       key       = _mm512_and_epi32(zmm_raw, key_mask);
            __m512i       first_key = _mm512_broadcastd_epi32(_mm512_castsi512_si128(key));
            __mmask16     k_same    = _mm512_mask_cmpeq_epi32_mask(k_active, key, first_key);

            if (k_same == k_active)
            {
                // VPERMB/VPERMI2D path: all entries share same attacker + from_sq
                unsigned atk_o  = _mm_cvtsi128_si32(_mm512_castsi512_si128(zmm_atk_o));
                unsigned from_o = _mm_cvtsi128_si32(_mm512_castsi512_si128(zmm_from_o));

                // offsets[AO][from_o]: both indices constant, scalar broadcast
                __m512i zmm_off = _mm512_set1_epi32(offsets[atk_o][from_o]);

                // index_lut2[AO][from_o][*]: load 64-byte row, VPERMB by to_oriented
                __m512i zmm_lut2_row = _mm512_loadu_si512(&index_lut2[atk_o][from_o][0]);
                __m512i zmm_lut2 =
                  _mm512_and_epi32(_mm512_permutexvar_epi8(zmm_to_o, zmm_lut2_row), zmm_0xff);

                // index_lut1[AO][*][*]: 32 dwords for this attacker, VPERMI2D
                const auto* lut1_row = reinterpret_cast<const __m512i*>(&index_lut1[atk_o][0][0]);
                __m512i     zmm_lut1_lo = _mm512_loadu_si512(lut1_row);
                __m512i     zmm_lut1_hi = _mm512_loadu_si512(lut1_row + 1);
                __m512i     zmm_idx1 = _mm512_add_epi32(_mm512_slli_epi32(zmm_atkd_o, 1), zmm_cmp);
                __m512i zmm_lut1 = _mm512_permutex2var_epi32(zmm_lut1_lo, zmm_idx1, zmm_lut1_hi);

                zmm_result = _mm512_add_epi32(_mm512_add_epi32(zmm_lut1, zmm_off), zmm_lut2);
            }
            else
    #endif
            {
                // Gather path for heterogeneous batches (mixed attacker/from_sq)
                __m512i zmm_idx1 =
                  _mm512_add_epi32(_mm512_add_epi32(_mm512_slli_epi32(zmm_atk_o, 5),
                                                    _mm512_slli_epi32(zmm_atkd_o, 1)),
                                   zmm_cmp);
                __m512i zmm_lut1 = _mm512_i32gather_epi32(zmm_idx1, lut1_base, 4);

                __m512i zmm_idx_off = _mm512_add_epi32(_mm512_slli_epi32(zmm_atk_o, 6), zmm_from_o);
                __m512i zmm_off     = _mm512_i32gather_epi32(zmm_idx_off, off_base, 4);

                __m512i zmm_idx2 =
                  _mm512_add_epi32(_mm512_add_epi32(_mm512_slli_epi32(zmm_atk_o, 12),
                                                    _mm512_slli_epi32(zmm_from_o, 6)),
                                   zmm_to_o);
                __m512i zmm_lut2 =
                  _mm512_and_epi32(_mm512_i32gather_epi32(zmm_idx2, lut2_base, 1), zmm_0xff);

                zmm_result = _mm512_add_epi32(_mm512_add_epi32(zmm_lut1, zmm_off), zmm_lut2);
            }

            // Filter: index < Dimensions, only within active lanes
            __mmask16 k_valid = _mm512_mask_cmplt_epu32_mask(k_active, zmm_result, zmm_dim);

            // Split by add flag (bit 31 of raw DirtyThreat)
            __mmask16 k_is_add    = _mm512_test_epi32_mask(zmm_raw, zmm_sign_bit);
            __mmask16 k_valid_add = k_valid & k_is_add;
            __mmask16 k_valid_rem = k_valid & ~k_is_add;

            // Compress-store valid adds
            int n_add = popcount(Bitboard(k_valid_add));
            if (n_add)
            {
                auto* ptr = added.make_space(n_add);
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr),
                                    _mm512_maskz_compress_epi32(k_valid_add, zmm_result));
            }

            // Compress-store valid removes
            int n_rem = popcount(Bitboard(k_valid_rem));
            if (n_rem)
            {
                auto* ptr = removed.make_space(n_rem);
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr),
                                    _mm512_maskz_compress_epi32(k_valid_rem, zmm_result));
            }
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
