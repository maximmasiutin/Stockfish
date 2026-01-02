/*
  Embedded Tablebase Probe - O(1) collision-free lookup for 3-4 piece endings
  Uses direct position indexing with 8-fold symmetry reduction.

  NOTE: Pawn endings only support horizontal flip (2-fold symmetry).
  Positions requiring vertical/diagonal transforms cannot be probed.
*/

#pragma once

#include <cstdint>
#include "types.h"
#include "position.h"
#include "embedded_tb.h"

namespace Stockfish {
namespace Tablebases {
namespace Embedded {

// Triangle mapping for white king (8-fold symmetry: a1-d1-d4)
constexpr int TRIANGLE[64] = {
     0,  1,  2,  3, -1, -1, -1, -1,
    -1,  4,  5,  6, -1, -1, -1, -1,
    -1, -1,  7,  8, -1, -1, -1, -1,
    -1, -1, -1,  9, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1
};

// Transform square by symmetry operation
// Bit 0: horizontal flip, Bit 1: vertical flip, Bit 2: diagonal swap
inline int transform(int sq, int t) {
    int f = sq & 7, r = sq >> 3;
    if (t & 1) f = 7 - f;
    if (t & 2) r = 7 - r;
    if (t & 4) { int tmp = f; f = r; r = tmp; }
    return r * 8 + f;
}

// Get transform needed to put white king in triangle
inline int get_transform(int sq) {
    int f = sq & 7, r = sq >> 3;
    int t = 0;
    if (f > 3) { t |= 1; f = 7 - f; }
    if (r > 3) { t |= 2; r = 7 - r; }
    if (r > f) { t |= 4; }
    return t;
}

// Check if transform is valid for pawn positions
// Pawns only allow horizontal flip (bit 0); vertical (bit 1) and diagonal (bit 2) are invalid
inline bool transform_valid_for_pawns(int t) {
    return (t & 6) == 0;  // bits 1 and 2 must be clear
}

// Calculate material value for a side (used to determine dominant side)
inline int material_value(const Position& pos, Color c) {
    return pos.count<QUEEN>(c) * 9
         + pos.count<ROOK>(c) * 5
         + pos.count<BISHOP>(c) * 3
         + pos.count<KNIGHT>(c) * 3
         + pos.count<PAWN>(c) * 1;
}

// Encode 3-piece position (KXvK)
// has_pawn: if true, rejects positions requiring vertical/diagonal transform
inline int encode3(int wk, int bk, int p, bool has_pawn = false) {
    int t = get_transform(wk);
    if (has_pawn && !transform_valid_for_pawns(t)) return -1;
    int idx = TRIANGLE[transform(wk, t)];
    if (idx < 0) return -1;
    return idx * 4096 + transform(bk, t) * 64 + transform(p, t);
}

// Encode 4-piece position (KXvKY)
// has_pawn: if true, rejects positions requiring vertical/diagonal transform
inline int encode4(int wk, int bk, int p1, int p2, bool has_pawn = false) {
    int t = get_transform(wk);
    if (has_pawn && !transform_valid_for_pawns(t)) return -1;
    int idx = TRIANGLE[transform(wk, t)];
    if (idx < 0) return -1;
    return idx * 262144 + transform(bk, t) * 4096 + transform(p1, t) * 64 + transform(p2, t);
}

// Encode 4-piece position with same pieces (KXXvK)
// has_pawn: if true, rejects positions requiring vertical/diagonal transform
inline int encode4_same(int wk, int bk, int p1, int p2, bool has_pawn = false) {
    int t = get_transform(wk);
    if (has_pawn && !transform_valid_for_pawns(t)) return -1;
    int wk_t = transform(wk, t);
    int idx = TRIANGLE[wk_t];
    if (idx < 0) return -1;
    int p1_t = transform(p1, t), p2_t = transform(p2, t);
    if (p1_t > p2_t) { int tmp = p1_t; p1_t = p2_t; p2_t = tmp; }
    int piece_idx = p1_t * 64 + p2_t - (p1_t * (p1_t + 1)) / 2;
    return idx * 64 * 2080 + transform(bk, t) * 2080 + piece_idx;
}

// WDL values: 0=invalid, 1=loss, 2=draw, 3=win
// Returns true if probe succeeded, sets wdl to result
// wdl: -1=loss, 0=draw, 1=win (from side to move perspective)
inline bool probe(const Position& pos, int& wdl) {
    if (pos.can_castle(ANY_CASTLING))
        return false;

    int piece_cnt = popcount(pos.pieces());
    if (piece_cnt < 3 || piece_cnt > 4)
        return false;

    // Determine dominated (stronger) side by material value, not piece count
    int white_mat = material_value(pos, WHITE);
    int black_mat = material_value(pos, BLACK);
    Color strong_side = (white_mat >= black_mat) ? WHITE : BLACK;
    // For symmetric BTM, swap sides so BLACK becomes strong (matches Syzygy flipColor)
    if (white_mat == black_mat && pos.side_to_move() == BLACK)
        strong_side = BLACK;
    Color weak_side = ~strong_side;

    // Tables only store positions with strong side to move.
    // Cannot probe when weak side is to move (would need different data).
    if (pos.side_to_move() != strong_side)
        return false;

    // Check kings not adjacent
    if (distance(pos.square<KING>(strong_side), pos.square<KING>(weak_side)) <= 1)
        return false;

    // Tablebases are indexed with WHITE as the strong side, WHITE to move.
    // Flip squares when:
    // 1. BLACK is stronger (to make BLACK appear as WHITE in table lookup)
    // 2. Symmetric position with BLACK to move (table only stores WHITE to move)
    bool black_stronger = (strong_side == BLACK);
    
    int flip = black_stronger ? 56 : 0;

    int strong_king = int(pos.square<KING>(strong_side)) ^ flip;
    int weak_king = int(pos.square<KING>(weak_side)) ^ flip;

    // Check for pawns (affects symmetry)
    bool has_pawns = pos.count<PAWN>(WHITE) > 0 || pos.count<PAWN>(BLACK) > 0;

    int result = 0;
    bool found = false;

    if (piece_cnt == 3) {
        // 3-piece: KXvK (strong side has the piece)
        Bitboard strong_pieces = pos.pieces(strong_side) ^ pos.pieces(strong_side, KING);
        if (!strong_pieces) return false;

        Square p = lsb(strong_pieces);
        PieceType pt = type_of(pos.piece_on(p));
        bool is_pawn = (pt == PAWN);
        int idx = encode3(strong_king, weak_king, int(p) ^ flip, is_pawn);
        if (idx < 0) return false;

        switch (pt) {
            case QUEEN:  result = probe_KQvK(idx); found = true; break;
            case ROOK:   result = probe_KRvK(idx); found = true; break;
            case BISHOP: result = probe_KBvK(idx); found = true; break;
            case KNIGHT: result = probe_KNvK(idx); found = true; break;
            case PAWN:   result = probe_KPvK(idx); found = true; break;
            default: return false;
        }
    }
    else if (piece_cnt == 4) {
        Bitboard strong_pieces = pos.pieces(strong_side) ^ pos.pieces(strong_side, KING);
        Bitboard weak_pieces = pos.pieces(weak_side) ^ pos.pieces(weak_side, KING);

        int strong_non_king = popcount(strong_pieces);
        int weak_non_king = popcount(weak_pieces);

        if (strong_non_king == 1 && weak_non_king == 1) {
            // KXvKY - one piece per side
            Square sp = lsb(strong_pieces);
            Square wp = lsb(weak_pieces);
            PieceType spt = type_of(pos.piece_on(sp));
            PieceType wpt = type_of(pos.piece_on(wp));
            int idx = encode4(strong_king, weak_king, int(sp) ^ flip, int(wp) ^ flip, has_pawns);
            if (idx < 0) return false;

            // Match against available tables (strong piece type vs weak piece type)
            if (spt == QUEEN && wpt == QUEEN)       { result = probe_KQvKQ(idx); found = true; }
            else if (spt == QUEEN && wpt == ROOK)   { result = probe_KQvKR(idx); found = true; }
            else if (spt == QUEEN && wpt == BISHOP) { result = probe_KQvKB(idx); found = true; }
            else if (spt == QUEEN && wpt == KNIGHT) { result = probe_KQvKN(idx); found = true; }
            else if (spt == ROOK && wpt == ROOK)    { result = probe_KRvKR(idx); found = true; }
            else if (spt == ROOK && wpt == BISHOP)  { result = probe_KRvKB(idx); found = true; }
            else if (spt == PAWN && wpt == PAWN)    { result = probe_KPvKP(idx); found = true; }
            else if (spt == BISHOP && wpt == PAWN)  { result = probe_KBvKP(idx); found = true; }
            else if (spt == KNIGHT && wpt == PAWN)  { result = probe_KNvKP(idx); found = true; }
        }
        else if (strong_non_king == 2 && weak_non_king == 0) {
            // KXXvK or KXYvK - two pieces on strong side
            Square p1 = lsb(strong_pieces);
            Square p2 = msb(strong_pieces);
            PieceType pt1 = type_of(pos.piece_on(p1));
            PieceType pt2 = type_of(pos.piece_on(p2));

            if (pt1 == pt2) {
                // KXXvK (same pieces)
                bool is_pawn = (pt1 == PAWN);
                int idx = encode4_same(strong_king, weak_king, int(p1) ^ flip, int(p2) ^ flip, is_pawn);
                if (idx < 0) return false;

                switch (pt1) {
                    case QUEEN:  result = probe_KQQvK(idx); found = true; break;
                    case ROOK:   result = probe_KRRvK(idx); found = true; break;
                    case BISHOP: result = probe_KBBvK(idx); found = true; break;
                    case KNIGHT: result = probe_KNNvK(idx); found = true; break;
                    case PAWN:   result = probe_KPPvK(idx); found = true; break;
                    default: return false;
                }
            }
            else {
                // KXYvK (different pieces) - ensure pt1 > pt2 for canonical order
                if (pt1 < pt2) {
                    std::swap(p1, p2);
                    std::swap(pt1, pt2);
                }
                bool is_pawn = (pt1 == PAWN || pt2 == PAWN);
                int idx = encode4(strong_king, weak_king, int(p1) ^ flip, int(p2) ^ flip, is_pawn);
                if (idx < 0) return false;

                if (pt1 == QUEEN && pt2 == ROOK)        { result = probe_KQRvK(idx); found = true; }
                else if (pt1 == QUEEN && pt2 == BISHOP) { result = probe_KQBvK(idx); found = true; }
                else if (pt1 == QUEEN && pt2 == KNIGHT) { result = probe_KQNvK(idx); found = true; }
                else if (pt1 == QUEEN && pt2 == PAWN)   { result = probe_KQPvK(idx); found = true; }
                else if (pt1 == ROOK && pt2 == BISHOP)  { result = probe_KRBvK(idx); found = true; }
                else if (pt1 == ROOK && pt2 == KNIGHT)  { result = probe_KRNvK(idx); found = true; }
                else if (pt1 == ROOK && pt2 == PAWN)    { result = probe_KRPvK(idx); found = true; }
                else if (pt1 == BISHOP && pt2 == KNIGHT){ result = probe_KBNvK(idx); found = true; }
            }
        }
    }

    if (!found || result == 0)
        return false;

    // Convert 2-bit result to WDL: 1=loss, 2=draw, 3=win
    // Tables store result from strong side's perspective (strong side to move)
    wdl = (result == 3) ? 1 : (result == 1) ? -1 : 0;
    return true;
}

}  // namespace Embedded
}  // namespace Tablebases
}  // namespace Stockfish
