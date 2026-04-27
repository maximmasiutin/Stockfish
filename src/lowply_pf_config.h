// Lowply prefetch configuration knobs (research-only, not shipped).
//
// Selects read-side and write-side prefetch placement and locality at
// compile time. Defaults reproduce the lowply-hashed-pf-la2-rc head
// (rolling K=2 read prefetch at T0, no write-side prefetch).
//
// Set via -D on the make command line, e.g.
// make profile-build ARCH=x86-64-avx512icl COMP=mingw
// EXTRACXXFLAGS="-DLOWPLY_PF_READ_SITE=1 -DLOWPLY_PF_READ_LOCALITY=0"

#pragma once

// LOWPLY_PF_READ_SITE
//   0 = none (no read-side prefetch in MovePicker::score<QUIETS>)
//   1 = bulk uniform pre-loop sweep (single locality from
//       LOWPLY_PF_READ_LOCALITY)
//   2 = rolling K=2 packed-register cache (la2-rc default)
//   3 = bulk partial pre-loop sweep: first LOWPLY_PF_READ_HI_COUNT
//       moves at T0, next LOWPLY_PF_READ_LO_COUNT moves at T2,
//       remainder skipped
#ifndef LOWPLY_PF_READ_SITE
    #define LOWPLY_PF_READ_SITE 2
#endif

// LOWPLY_PF_READ_LOCALITY (used for SITE 1 and 2)
//   0 = T0 (PREFETCHT0, fetched into L1)
//   1 = T2 (PREFETCHT2, fetched into L2/L3, bypasses L1)
#ifndef LOWPLY_PF_READ_LOCALITY
    #define LOWPLY_PF_READ_LOCALITY 0
#endif

// LOWPLY_PF_READ_HI_COUNT (partial-bulk, SITE == 3)
//   Number of front-of-list moves to prefetch at T0.
#ifndef LOWPLY_PF_READ_HI_COUNT
    #define LOWPLY_PF_READ_HI_COUNT 4
#endif

// LOWPLY_PF_READ_LO_COUNT (partial-bulk, SITE == 3)
//   -1 = prefetch the rest of the list at T2
//    0 = no T2 prefetches after the T0 segment (cap)
//   >0 = prefetch this many additional moves at T2
#ifndef LOWPLY_PF_READ_LO_COUNT
    #define LOWPLY_PF_READ_LO_COUNT 0
#endif

// LOWPLY_PF_WRITE_SITE
//   0 = none (no write-side prefetch)
//   1 = W1 bulk sweep at top of update_all_stats over
//       quietsSearched plus bestMove
//   2 = W3 rolling K=1 inside the quietsSearched loop
//   3 = W4 single prefetch at top of update_quiet_histories
//   4 = W5 scatter at quietsSearched.push_back in the search loop
#ifndef LOWPLY_PF_WRITE_SITE
    #define LOWPLY_PF_WRITE_SITE 0
#endif

// LOWPLY_PF_WRITE_LOCALITY
//   0 = T0
//   1 = T2
#ifndef LOWPLY_PF_WRITE_LOCALITY
    #define LOWPLY_PF_WRITE_LOCALITY 0
#endif

// Hint macros derived from the locality knobs.
#if LOWPLY_PF_READ_LOCALITY == 0
    #define LOWPLY_PF_READ_HINT _MM_HINT_T0
#else
    #define LOWPLY_PF_READ_HINT _MM_HINT_T2
#endif

#if LOWPLY_PF_WRITE_LOCALITY == 0
    #define LOWPLY_PF_WRITE_HINT _MM_HINT_T0
#else
    #define LOWPLY_PF_WRITE_HINT _MM_HINT_T2
#endif
