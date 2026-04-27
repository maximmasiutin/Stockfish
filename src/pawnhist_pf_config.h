// pawn_entry do_move prefetch configuration knob (research-only).
//
// Selects whether the do_move prefetch of history->pawn_entry(...) is
// kept at T0 (master), dropped, or demoted to T2. The other 4 corrhist
// prefetches (pawn_correction, minor_piece_correction, nonpawn_W,
// nonpawn_B) remain at T0 unconditionally.
//
// Set via -D on the make command line, e.g.
// EXTRACXXFLAGS="-DPAWNHIST_PF_VARIANT=1"

#pragma once

// PAWNHIST_PF_VARIANT
//   0 = baseline (master): pawn_entry prefetch at T0
//   1 = drop pawn_entry prefetch entirely
//   2 = demote pawn_entry prefetch to T2 (PREFETCHT2)
#ifndef PAWNHIST_PF_VARIANT
    #define PAWNHIST_PF_VARIANT 0
#endif
