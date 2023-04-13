/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2023 The Stockfish developers (see AUTHORS file)

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

// Definition of layer AffineTransform of NNUE evaluation function

#ifndef NNUE_LAYERS_AFFINE_TRANSFORM_H_INCLUDED
#define NNUE_LAYERS_AFFINE_TRANSFORM_H_INCLUDED

#include <iostream>
#include <algorithm>
#include <type_traits>
#include "../nnue_common.h"
#include "simd.h"

/*
  This file contains the definition for a fully connected layer (aka affine transform).
  Two approaches are employed, depending on the sizes of the transform.

  Approach 1 (a specialization for large inputs):
    - used when the PaddedInputDimensions >= 128
    - uses AVX512 if possible
    - processes inputs in batches of 2*InputSimdWidth
      - so in batches of 128 for AVX512
    - the weight blocks of size InputSimdWidth are transposed such that
      access is sequential
    - N columns of the weight matrix are processed a time, where N
      depends on the architecture (the amount of registers)
    - accumulate + hadd is used

  Approach 2 (a specialization for small inputs):
    - used when the PaddedInputDimensions < 128
    - expected use-case is for when PaddedInputDimensions == 32 and InputDimensions <= 32.
      - that's why AVX512 is hard to implement
    - expected use-case is small layers
      - not optimized as well as the approach 1
    - inputs are processed in chunks of 4, weights are respectively transposed
    - accumulation happens directly to int32s
*/

namespace Stockfish::Eval::NNUE::Layers {

// Fallback implementation for older/other architectures.
// Identical for both approaches. Requires the input to be padded to at least 16 values.
#if !defined(USE_SSSE3)
  template <IndexType InputDimensions, IndexType PaddedInputDimensions, IndexType OutputDimensions>
  static void affine_transform_non_ssse3(std::int32_t* output, const std::int8_t* weights, const std::int32_t* biases, const std::uint8_t* input)
  {
# if defined(USE_SSE2)
    // At least a multiple of 16, with SSE2.
    constexpr IndexType NumChunks = ceil_to_multiple<IndexType>(InputDimensions, 16) / 16;
    const __m128i Zeros = _mm_setzero_si128();
    const auto inputVector = reinterpret_cast<const __m128i*>(input);

# elif defined(USE_MMX)
    constexpr IndexType NumChunks = ceil_to_multiple<IndexType>(InputDimensions, 8) / 8;
    const __m64 Zeros = _mm_setzero_si64();
    const auto inputVector = reinterpret_cast<const __m64*>(input);

# elif defined(USE_NEON_DOTPROD)
    constexpr IndexType NumChunks = ceil_to_multiple<IndexType>(InputDimensions, 16) / 16;
    const auto inputVector = reinterpret_cast<const int8x16_t*>(input);

# elif defined(USE_NEON)
    constexpr IndexType NumChunks = ceil_to_multiple<IndexType>(InputDimensions, 16) / 16;
    const auto inputVector = reinterpret_cast<const int8x8_t*>(input);
# endif

    for (IndexType i = 0; i < OutputDimensions; ++i) {
      const IndexType offset = i * PaddedInputDimensions;

# if defined(USE_SSE2)
      __m128i sumLo = _mm_cvtsi32_si128(biases[i]);
      __m128i sumHi = Zeros;
      const auto row = reinterpret_cast<const __m128i*>(&weights[offset]);
      for (IndexType j = 0; j < NumChunks; ++j) {
        __m128i row_j = _mm_load_si128(&row[j]);
        __m128i input_j = _mm_load_si128(&inputVector[j]);
        __m128i extendedRowLo = _mm_srai_epi16(_mm_unpacklo_epi8(row_j, row_j), 8);
        __m128i extendedRowHi = _mm_srai_epi16(_mm_unpackhi_epi8(row_j, row_j), 8);
        __m128i extendedInputLo = _mm_unpacklo_epi8(input_j, Zeros);
        __m128i extendedInputHi = _mm_unpackhi_epi8(input_j, Zeros);
        __m128i productLo = _mm_madd_epi16(extendedRowLo, extendedInputLo);
        __m128i productHi = _mm_madd_epi16(extendedRowHi, extendedInputHi);
        sumLo = _mm_add_epi32(sumLo, productLo);
        sumHi = _mm_add_epi32(sumHi, productHi);
      }
      __m128i sum = _mm_add_epi32(sumLo, sumHi);
      __m128i sumHigh_64 = _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2));
      sum = _mm_add_epi32(sum, sumHigh_64);
      __m128i sum_second_32 = _mm_shufflelo_epi16(sum, _MM_SHUFFLE(1, 0, 3, 2));
      sum = _mm_add_epi32(sum, sum_second_32);
      output[i] = _mm_cvtsi128_si32(sum);

# elif defined(USE_MMX)
      __m64 sumLo = _mm_cvtsi32_si64(biases[i]);
      __m64 sumHi = Zeros;
      const auto row = reinterpret_cast<const __m64*>(&weights[offset]);
      for (IndexType j = 0; j < NumChunks; ++j) {
        __m64 row_j = row[j];
        __m64 input_j = inputVector[j];
        __m64 extendedRowLo = _mm_srai_pi16(_mm_unpacklo_pi8(row_j, row_j), 8);
        __m64 extendedRowHi = _mm_srai_pi16(_mm_unpackhi_pi8(row_j, row_j), 8);
        __m64 extendedInputLo = _mm_unpacklo_pi8(input_j, Zeros);
        __m64 extendedInputHi = _mm_unpackhi_pi8(input_j, Zeros);
        __m64 productLo = _mm_madd_pi16(extendedRowLo, extendedInputLo);
        __m64 productHi = _mm_madd_pi16(extendedRowHi, extendedInputHi);
        sumLo = _mm_add_pi32(sumLo, productLo);
        sumHi = _mm_add_pi32(sumHi, productHi);
      }
      __m64 sum = _mm_add_pi32(sumLo, sumHi);
      sum = _mm_add_pi32(sum, _mm_unpackhi_pi32(sum, sum));
      output[i] = _mm_cvtsi64_si32(sum);

# elif defined(USE_NEON_DOTPROD)
      int32x4_t sum = {biases[i]};
      const auto row = reinterpret_cast<const int8x16_t*>(&weights[offset]);
      for (IndexType j = 0; j < NumChunks; ++j) {
        sum = vdotq_s32(sum, inputVector[j], row[j]);
      }
      output[i] = vaddvq_s32(sum);

# elif defined(USE_NEON)
      int32x4_t sum = {biases[i]};
      const auto row = reinterpret_cast<const int8x8_t*>(&weights[offset]);
      for (IndexType j = 0; j < NumChunks; ++j) {
        int16x8_t product = vmull_s8(inputVector[j * 2], row[j * 2]);
        product = vmlal_s8(product, inputVector[j * 2 + 1], row[j * 2 + 1]);
        sum = vpadalq_s16(sum, product);
      }
      output[i] = sum[0] + sum[1] + sum[2] + sum[3];

# else
      std::int32_t sum = biases[i];
      for (IndexType j = 0; j < InputDimensions; ++j) {
        sum += weights[offset + j] * input[j];
      }
      output[i] = sum;
# endif
    }

# if defined(USE_MMX)
    _mm_empty();
# endif
  }
#endif

  template <IndexType InDims, IndexType OutDims, typename Enabled = void>
  class AffineTransform;

#if defined (USE_AVX512)
  constexpr IndexType LargeInputSize = 2 * 64;
#else
  constexpr IndexType LargeInputSize = std::numeric_limits<IndexType>::max();
#endif

  // A specialization for large inputs
  template <IndexType InDims, IndexType OutDims>
  class AffineTransform<InDims, OutDims, std::enable_if_t<(ceil_to_multiple<IndexType>(InDims, MaxSimdWidth) >= LargeInputSize)>> {
   public:
    // Input/output type
    using InputType = std::uint8_t;
    using OutputType = std::int32_t;

    // Number of input/output dimensions
    static constexpr IndexType InputDimensions = InDims;
    static constexpr IndexType OutputDimensions = OutDims;

    static constexpr IndexType PaddedInputDimensions =
      ceil_to_multiple<IndexType>(InputDimensions, MaxSimdWidth);
    static constexpr IndexType PaddedOutputDimensions =
      ceil_to_multiple<IndexType>(OutputDimensions, MaxSimdWidth);

    using OutputBuffer = OutputType[PaddedOutputDimensions];

    static_assert(PaddedInputDimensions >= LargeInputSize, "Something went wrong. This specialization (for large inputs) should not have been chosen.");

#if defined (USE_AVX512)
    static constexpr IndexType InputSimdWidth = 64;
    static constexpr IndexType MaxNumOutputRegs = 16;
#elif defined (USE_AVX2)
    static constexpr IndexType InputSimdWidth = 32;
    static constexpr IndexType MaxNumOutputRegs = 8;
#elif defined (USE_SSSE3)
    static constexpr IndexType InputSimdWidth = 16;
    static constexpr IndexType MaxNumOutputRegs = 8;
#elif defined (USE_NEON_DOTPROD)
    static constexpr IndexType InputSimdWidth = 16;
    static constexpr IndexType MaxNumOutputRegs = 8;
#elif defined (USE_NEON)
    static constexpr IndexType InputSimdWidth = 8;
    static constexpr IndexType MaxNumOutputRegs = 8;
#else
    // The fallback implementation will not have permuted weights.
    // We define these to avoid a lot of ifdefs later.
    static constexpr IndexType InputSimdWidth = 1;
    static constexpr IndexType MaxNumOutputRegs = 1;
#endif

    // A big block is a region in the weight matrix of the size [PaddedInputDimensions, NumOutputRegs].
    // A small block is a region of size [InputSimdWidth, 1]

    static constexpr IndexType NumOutputRegs = std::min(MaxNumOutputRegs, OutputDimensions);
    static constexpr IndexType SmallBlockSize = InputSimdWidth;
    static constexpr IndexType BigBlockSize = NumOutputRegs * PaddedInputDimensions;
    static constexpr IndexType NumSmallBlocksInBigBlock = BigBlockSize / SmallBlockSize;
    static constexpr IndexType NumSmallBlocksPerOutput = PaddedInputDimensions / SmallBlockSize;
    static constexpr IndexType NumBigBlocks = OutputDimensions / NumOutputRegs;

    static_assert(OutputDimensions % NumOutputRegs == 0);

    // Hash value embedded in the evaluation file
    static constexpr std::uint32_t get_hash_value(std::uint32_t prevHash) {
      std::uint32_t hashValue = 0xCC03DAE4u;
      hashValue += OutputDimensions;
      hashValue ^= prevHash >> 1;
      hashValue ^= prevHash << 31;
      return hashValue;
    }

    /*
      Transposes the small blocks within a block.
      Effectively means that weights can be traversed sequentially during inference.
    */
    static IndexType get_weight_index(IndexType i)
    {
      const IndexType smallBlock = (i / SmallBlockSize) % NumSmallBlocksInBigBlock;
      const IndexType smallBlockCol = smallBlock / NumSmallBlocksPerOutput;
      const IndexType smallBlockRow = smallBlock % NumSmallBlocksPerOutput;
      const IndexType bigBlock   = i / BigBlockSize;
      const IndexType rest       = i % SmallBlockSize;

      const IndexType idx =
          bigBlock * BigBlockSize
        + smallBlockRow * SmallBlockSize * NumOutputRegs
        + smallBlockCol * SmallBlockSize
        + rest;

      return idx;
    }

    // Read network parameters
    bool read_parameters(std::istream& stream) {
      read_little_endian<BiasType>(stream, biases, OutputDimensions);

      for (IndexType i = 0; i < OutputDimensions * PaddedInputDimensions; ++i)
        weights[get_weight_index(i)] = read_little_endian<WeightType>(stream);

      return !stream.fail();
    }

    // Write network parameters
    bool write_parameters(std::ostream& stream) const {
      write_little_endian<BiasType>(stream, biases, OutputDimensions);

      for (IndexType i = 0; i < OutputDimensions * PaddedInputDimensions; ++i)
        write_little_endian<WeightType>(stream, weights[get_weight_index(i)]);

      return !stream.fail();
    }

    // Forward propagation
    const OutputType* propagate(
        const InputType* input, OutputType* output) const {

#if defined (USE_AVX512)
      using acc_vec_t = __m512i;
      using bias_vec_t = __m128i;
      using weight_vec_t = __m512i;
      using in_vec_t = __m512i;
      #define vec_zero _mm512_setzero_si512()
      #define vec_add_dpbusd_32x2 Simd::m512_add_dpbusd_epi32x2
      #define vec_hadd Simd::m512_hadd
      #define vec_haddx4 Simd::m512_haddx4
#elif defined (USE_AVX2)
      using acc_vec_t = __m256i;
      using bias_vec_t = __m128i;
      using weight_vec_t = __m256i;
      using in_vec_t = __m256i;
      #define vec_zero _mm256_setzero_si256()
      #define vec_add_dpbusd_32x2 Simd::m256_add_dpbusd_epi32x2
      #define vec_hadd Simd::m256_hadd
      #define vec_haddx4 Simd::m256_haddx4
#elif defined (USE_SSSE3)
      using acc_vec_t = __m128i;
      using bias_vec_t = __m128i;
      using weight_vec_t = __m128i;
      using in_vec_t = __m128i;
      #define vec_zero _mm_setzero_si128()
      #define vec_add_dpbusd_32x2 Simd::m128_add_dpbusd_epi32x2
      #define vec_hadd Simd::m128_hadd
      #define vec_haddx4 Simd::m128_haddx4
#elif defined (USE_NEON_DOTPROD)
      using acc_vec_t = int32x4_t;
      using bias_vec_t = int32x4_t;
      using weight_vec_t = int8x16_t;
      using in_vec_t = int8x16_t;
      #define vec_zero {0}
      #define vec_add_dpbusd_32x2 Simd::dotprod_m128_add_dpbusd_epi32x2
      #define vec_hadd Simd::neon_m128_hadd
      #define vec_haddx4 Simd::neon_m128_haddx4
#elif defined (USE_NEON)
      using acc_vec_t = int32x4_t;
      using bias_vec_t = int32x4_t;
      using weight_vec_t = int8x8_t;
      using in_vec_t = int8x8_t;
      #define vec_zero {0}
      #define vec_add_dpbusd_32x2 Simd::neon_m128_add_dpbusd_epi32x2
      #define vec_hadd Simd::neon_m128_hadd
      #define vec_haddx4 Simd::neon_m128_haddx4
#endif

#if defined (USE_SSSE3) || defined (USE_NEON)
      const in_vec_t* invec = reinterpret_cast<const in_vec_t*>(input);

#if defined (USE_AVX512)
      if ((NumSmallBlocksPerOutput == 16) && (NumBigBlocks == 1))
      {
          // A special case for large inputs with the loops unrolled

          // Sequental load to registers gives implicit prefetch hint
          // and happens faster than scattered loads mixed with other loads (such as from invec[] and weightvec[])

          const in_vec_t ina0 = invec[0 * 2 + 0];
          const in_vec_t inb0 = invec[0 * 2 + 1];
          const in_vec_t ina1 = invec[1 * 2 + 0];
          const in_vec_t inb1 = invec[1 * 2 + 1];
          const in_vec_t ina2 = invec[2 * 2 + 0];
          const in_vec_t inb2 = invec[2 * 2 + 1];
          const in_vec_t ina3 = invec[3 * 2 + 0];
          const in_vec_t inb3 = invec[3 * 2 + 1];
          const in_vec_t ina4 = invec[4 * 2 + 0];
          const in_vec_t inb4 = invec[4 * 2 + 1];
          const in_vec_t ina5 = invec[5 * 2 + 0];
          const in_vec_t inb5 = invec[5 * 2 + 1];
          const in_vec_t ina6 = invec[6 * 2 + 0];
          const in_vec_t inb6 = invec[6 * 2 + 1];
          const in_vec_t ina7 = invec[7 * 2 + 0];
          const in_vec_t inb7 = invec[7 * 2 + 1];

          acc_vec_t acc0 = { vec_zero };
          acc_vec_t acc1 = { vec_zero };
          acc_vec_t acc2 = { vec_zero };
          acc_vec_t acc3 = { vec_zero };
          acc_vec_t acc4 = { vec_zero };
          acc_vec_t acc5 = { vec_zero };
          acc_vec_t acc6 = { vec_zero };
          acc_vec_t acc7 = { vec_zero };
          acc_vec_t acc8 = { vec_zero };
          acc_vec_t acc9 = { vec_zero };
          acc_vec_t acc10 = { vec_zero };
          acc_vec_t acc11 = { vec_zero };
          acc_vec_t acc12 = { vec_zero };
          acc_vec_t acc13 = { vec_zero };
          acc_vec_t acc14 = { vec_zero };
          acc_vec_t acc15 = { vec_zero };

          const weight_vec_t* weightvec0 = reinterpret_cast<const weight_vec_t*>(weights + 0 * 2 * SmallBlockSize * NumOutputRegs);
          const weight_vec_t* weightvec1 = reinterpret_cast<const weight_vec_t*>(weights + 1 * 2 * SmallBlockSize * NumOutputRegs);
          const weight_vec_t* weightvec2 = reinterpret_cast<const weight_vec_t*>(weights + 2 * 2 * SmallBlockSize * NumOutputRegs);
          const weight_vec_t* weightvec3 = reinterpret_cast<const weight_vec_t*>(weights + 3 * 2 * SmallBlockSize * NumOutputRegs);
          const weight_vec_t* weightvec4 = reinterpret_cast<const weight_vec_t*>(weights + 4 * 2 * SmallBlockSize * NumOutputRegs);
          const weight_vec_t* weightvec5 = reinterpret_cast<const weight_vec_t*>(weights + 5 * 2 * SmallBlockSize * NumOutputRegs);
          const weight_vec_t* weightvec6 = reinterpret_cast<const weight_vec_t*>(weights + 6 * 2 * SmallBlockSize * NumOutputRegs);
          const weight_vec_t* weightvec7 = reinterpret_cast<const weight_vec_t*>(weights + 7 * 2 * SmallBlockSize * NumOutputRegs);

          acc0 = _mm512_dpbusd_epi32(acc0, ina0, weightvec0[0]);
          acc1 = _mm512_dpbusd_epi32(acc1, ina0, weightvec0[1]);
          acc2 = _mm512_dpbusd_epi32(acc2, ina0, weightvec0[2]);
          acc3 = _mm512_dpbusd_epi32(acc3, ina0, weightvec0[3]);
          acc4 = _mm512_dpbusd_epi32(acc4, ina0, weightvec0[4]);
          acc5 = _mm512_dpbusd_epi32(acc5, ina0, weightvec0[5]);
          acc6 = _mm512_dpbusd_epi32(acc6, ina0, weightvec0[6]);
          acc7 = _mm512_dpbusd_epi32(acc7, ina0, weightvec0[7]);
          acc8 = _mm512_dpbusd_epi32(acc8, ina0, weightvec0[8]);
          acc9 = _mm512_dpbusd_epi32(acc9, ina0, weightvec0[9]);
          acc10 = _mm512_dpbusd_epi32(acc10, ina0, weightvec0[10]);
          acc11 = _mm512_dpbusd_epi32(acc11, ina0, weightvec0[11]);
          acc12 = _mm512_dpbusd_epi32(acc12, ina0, weightvec0[12]);
          acc13 = _mm512_dpbusd_epi32(acc13, ina0, weightvec0[13]);
          acc14 = _mm512_dpbusd_epi32(acc14, ina0, weightvec0[14]);
          acc15 = _mm512_dpbusd_epi32(acc15, ina0, weightvec0[15]);
          acc0 = _mm512_dpbusd_epi32(acc0, ina1, weightvec1[0]);
          acc1 = _mm512_dpbusd_epi32(acc1, ina1, weightvec1[1]);
          acc2 = _mm512_dpbusd_epi32(acc2, ina1, weightvec1[2]);
          acc3 = _mm512_dpbusd_epi32(acc3, ina1, weightvec1[3]);
          acc4 = _mm512_dpbusd_epi32(acc4, ina1, weightvec1[4]);
          acc5 = _mm512_dpbusd_epi32(acc5, ina1, weightvec1[5]);
          acc6 = _mm512_dpbusd_epi32(acc6, ina1, weightvec1[6]);
          acc7 = _mm512_dpbusd_epi32(acc7, ina1, weightvec1[7]);
          acc8 = _mm512_dpbusd_epi32(acc8, ina1, weightvec1[8]);
          acc9 = _mm512_dpbusd_epi32(acc9, ina1, weightvec1[9]);
          acc10 = _mm512_dpbusd_epi32(acc10, ina1, weightvec1[10]);
          acc11 = _mm512_dpbusd_epi32(acc11, ina1, weightvec1[11]);
          acc12 = _mm512_dpbusd_epi32(acc12, ina1, weightvec1[12]);
          acc13 = _mm512_dpbusd_epi32(acc13, ina1, weightvec1[13]);
          acc14 = _mm512_dpbusd_epi32(acc14, ina1, weightvec1[14]);
          acc15 = _mm512_dpbusd_epi32(acc15, ina1, weightvec1[15]);
          acc0 = _mm512_dpbusd_epi32(acc0, ina2, weightvec2[0]);
          acc1 = _mm512_dpbusd_epi32(acc1, ina2, weightvec2[1]);
          acc2 = _mm512_dpbusd_epi32(acc2, ina2, weightvec2[2]);
          acc3 = _mm512_dpbusd_epi32(acc3, ina2, weightvec2[3]);
          acc4 = _mm512_dpbusd_epi32(acc4, ina2, weightvec2[4]);
          acc5 = _mm512_dpbusd_epi32(acc5, ina2, weightvec2[5]);
          acc6 = _mm512_dpbusd_epi32(acc6, ina2, weightvec2[6]);
          acc7 = _mm512_dpbusd_epi32(acc7, ina2, weightvec2[7]);
          acc8 = _mm512_dpbusd_epi32(acc8, ina2, weightvec2[8]);
          acc9 = _mm512_dpbusd_epi32(acc9, ina2, weightvec2[9]);
          acc10 = _mm512_dpbusd_epi32(acc10, ina2, weightvec2[10]);
          acc11 = _mm512_dpbusd_epi32(acc11, ina2, weightvec2[11]);
          acc12 = _mm512_dpbusd_epi32(acc12, ina2, weightvec2[12]);
          acc13 = _mm512_dpbusd_epi32(acc13, ina2, weightvec2[13]);
          acc14 = _mm512_dpbusd_epi32(acc14, ina2, weightvec2[14]);
          acc15 = _mm512_dpbusd_epi32(acc15, ina2, weightvec2[15]);
          acc0 = _mm512_dpbusd_epi32(acc0, ina3, weightvec3[0]);
          acc1 = _mm512_dpbusd_epi32(acc1, ina3, weightvec3[1]);
          acc2 = _mm512_dpbusd_epi32(acc2, ina3, weightvec3[2]);
          acc3 = _mm512_dpbusd_epi32(acc3, ina3, weightvec3[3]);
          acc4 = _mm512_dpbusd_epi32(acc4, ina3, weightvec3[4]);
          acc5 = _mm512_dpbusd_epi32(acc5, ina3, weightvec3[5]);
          acc6 = _mm512_dpbusd_epi32(acc6, ina3, weightvec3[6]);
          acc7 = _mm512_dpbusd_epi32(acc7, ina3, weightvec3[7]);
          acc8 = _mm512_dpbusd_epi32(acc8, ina3, weightvec3[8]);
          acc9 = _mm512_dpbusd_epi32(acc9, ina3, weightvec3[9]);
          acc10 = _mm512_dpbusd_epi32(acc10, ina3, weightvec3[10]);
          acc11 = _mm512_dpbusd_epi32(acc11, ina3, weightvec3[11]);
          acc12 = _mm512_dpbusd_epi32(acc12, ina3, weightvec3[12]);
          acc13 = _mm512_dpbusd_epi32(acc13, ina3, weightvec3[13]);
          acc14 = _mm512_dpbusd_epi32(acc14, ina3, weightvec3[14]);
          acc15 = _mm512_dpbusd_epi32(acc15, ina3, weightvec3[15]);
          acc0 = _mm512_dpbusd_epi32(acc0, ina4, weightvec4[0]);
          acc1 = _mm512_dpbusd_epi32(acc1, ina4, weightvec4[1]);
          acc2 = _mm512_dpbusd_epi32(acc2, ina4, weightvec4[2]);
          acc3 = _mm512_dpbusd_epi32(acc3, ina4, weightvec4[3]);
          acc4 = _mm512_dpbusd_epi32(acc4, ina4, weightvec4[4]);
          acc5 = _mm512_dpbusd_epi32(acc5, ina4, weightvec4[5]);
          acc6 = _mm512_dpbusd_epi32(acc6, ina4, weightvec4[6]);
          acc7 = _mm512_dpbusd_epi32(acc7, ina4, weightvec4[7]);
          acc8 = _mm512_dpbusd_epi32(acc8, ina4, weightvec4[8]);
          acc9 = _mm512_dpbusd_epi32(acc9, ina4, weightvec4[9]);
          acc10 = _mm512_dpbusd_epi32(acc10, ina4, weightvec4[10]);
          acc11 = _mm512_dpbusd_epi32(acc11, ina4, weightvec4[11]);
          acc12 = _mm512_dpbusd_epi32(acc12, ina4, weightvec4[12]);
          acc13 = _mm512_dpbusd_epi32(acc13, ina4, weightvec4[13]);
          acc14 = _mm512_dpbusd_epi32(acc14, ina4, weightvec4[14]);
          acc15 = _mm512_dpbusd_epi32(acc15, ina4, weightvec4[15]);
          acc0 = _mm512_dpbusd_epi32(acc0, ina5, weightvec5[0]);
          acc1 = _mm512_dpbusd_epi32(acc1, ina5, weightvec5[1]);
          acc2 = _mm512_dpbusd_epi32(acc2, ina5, weightvec5[2]);
          acc3 = _mm512_dpbusd_epi32(acc3, ina5, weightvec5[3]);
          acc4 = _mm512_dpbusd_epi32(acc4, ina5, weightvec5[4]);
          acc5 = _mm512_dpbusd_epi32(acc5, ina5, weightvec5[5]);
          acc6 = _mm512_dpbusd_epi32(acc6, ina5, weightvec5[6]);
          acc7 = _mm512_dpbusd_epi32(acc7, ina5, weightvec5[7]);
          acc8 = _mm512_dpbusd_epi32(acc8, ina5, weightvec5[8]);
          acc9 = _mm512_dpbusd_epi32(acc9, ina5, weightvec5[9]);
          acc10 = _mm512_dpbusd_epi32(acc10, ina5, weightvec5[10]);
          acc11 = _mm512_dpbusd_epi32(acc11, ina5, weightvec5[11]);
          acc12 = _mm512_dpbusd_epi32(acc12, ina5, weightvec5[12]);
          acc13 = _mm512_dpbusd_epi32(acc13, ina5, weightvec5[13]);
          acc14 = _mm512_dpbusd_epi32(acc14, ina5, weightvec5[14]);
          acc15 = _mm512_dpbusd_epi32(acc15, ina5, weightvec5[15]);
          acc0 = _mm512_dpbusd_epi32(acc0, ina6, weightvec6[0]);
          acc1 = _mm512_dpbusd_epi32(acc1, ina6, weightvec6[1]);
          acc2 = _mm512_dpbusd_epi32(acc2, ina6, weightvec6[2]);
          acc3 = _mm512_dpbusd_epi32(acc3, ina6, weightvec6[3]);
          acc4 = _mm512_dpbusd_epi32(acc4, ina6, weightvec6[4]);
          acc5 = _mm512_dpbusd_epi32(acc5, ina6, weightvec6[5]);
          acc6 = _mm512_dpbusd_epi32(acc6, ina6, weightvec6[6]);
          acc7 = _mm512_dpbusd_epi32(acc7, ina6, weightvec6[7]);
          acc8 = _mm512_dpbusd_epi32(acc8, ina6, weightvec6[8]);
          acc9 = _mm512_dpbusd_epi32(acc9, ina6, weightvec6[9]);
          acc10 = _mm512_dpbusd_epi32(acc10, ina6, weightvec6[10]);
          acc11 = _mm512_dpbusd_epi32(acc11, ina6, weightvec6[11]);
          acc12 = _mm512_dpbusd_epi32(acc12, ina6, weightvec6[12]);
          acc13 = _mm512_dpbusd_epi32(acc13, ina6, weightvec6[13]);
          acc14 = _mm512_dpbusd_epi32(acc14, ina6, weightvec6[14]);
          acc15 = _mm512_dpbusd_epi32(acc15, ina6, weightvec6[15]);
          acc0 = _mm512_dpbusd_epi32(acc0, ina7, weightvec7[0]);
          acc1 = _mm512_dpbusd_epi32(acc1, ina7, weightvec7[1]);
          acc2 = _mm512_dpbusd_epi32(acc2, ina7, weightvec7[2]);
          acc3 = _mm512_dpbusd_epi32(acc3, ina7, weightvec7[3]);
          acc4 = _mm512_dpbusd_epi32(acc4, ina7, weightvec7[4]);
          acc5 = _mm512_dpbusd_epi32(acc5, ina7, weightvec7[5]);
          acc6 = _mm512_dpbusd_epi32(acc6, ina7, weightvec7[6]);
          acc7 = _mm512_dpbusd_epi32(acc7, ina7, weightvec7[7]);
          acc8 = _mm512_dpbusd_epi32(acc8, ina7, weightvec7[8]);
          acc9 = _mm512_dpbusd_epi32(acc9, ina7, weightvec7[9]);
          acc10 = _mm512_dpbusd_epi32(acc10, ina7, weightvec7[10]);
          acc11 = _mm512_dpbusd_epi32(acc11, ina7, weightvec7[11]);
          acc12 = _mm512_dpbusd_epi32(acc12, ina7, weightvec7[12]);
          acc13 = _mm512_dpbusd_epi32(acc13, ina7, weightvec7[13]);
          acc14 = _mm512_dpbusd_epi32(acc14, ina7, weightvec7[14]);
          acc15 = _mm512_dpbusd_epi32(acc15, ina7, weightvec7[15]);
          acc0 = _mm512_dpbusd_epi32(acc0, inb0, weightvec0[0 + NumOutputRegs]);
          acc1 = _mm512_dpbusd_epi32(acc1, inb0, weightvec0[1 + NumOutputRegs]);
          acc2 = _mm512_dpbusd_epi32(acc2, inb0, weightvec0[2 + NumOutputRegs]);
          acc3 = _mm512_dpbusd_epi32(acc3, inb0, weightvec0[3 + NumOutputRegs]);
          acc4 = _mm512_dpbusd_epi32(acc4, inb0, weightvec0[4 + NumOutputRegs]);
          acc5 = _mm512_dpbusd_epi32(acc5, inb0, weightvec0[5 + NumOutputRegs]);
          acc6 = _mm512_dpbusd_epi32(acc6, inb0, weightvec0[6 + NumOutputRegs]);
          acc7 = _mm512_dpbusd_epi32(acc7, inb0, weightvec0[7 + NumOutputRegs]);
          acc8 = _mm512_dpbusd_epi32(acc8, inb0, weightvec0[8 + NumOutputRegs]);
          acc9 = _mm512_dpbusd_epi32(acc9, inb0, weightvec0[9 + NumOutputRegs]);
          acc10 = _mm512_dpbusd_epi32(acc10, inb0, weightvec0[10 + NumOutputRegs]);
          acc11 = _mm512_dpbusd_epi32(acc11, inb0, weightvec0[11 + NumOutputRegs]);
          acc12 = _mm512_dpbusd_epi32(acc12, inb0, weightvec0[12 + NumOutputRegs]);
          acc13 = _mm512_dpbusd_epi32(acc13, inb0, weightvec0[13 + NumOutputRegs]);
          acc14 = _mm512_dpbusd_epi32(acc14, inb0, weightvec0[14 + NumOutputRegs]);
          acc15 = _mm512_dpbusd_epi32(acc15, inb0, weightvec0[15 + NumOutputRegs]);
          acc0 = _mm512_dpbusd_epi32(acc0, inb1, weightvec1[0 + NumOutputRegs]);
          acc1 = _mm512_dpbusd_epi32(acc1, inb1, weightvec1[1 + NumOutputRegs]);
          acc2 = _mm512_dpbusd_epi32(acc2, inb1, weightvec1[2 + NumOutputRegs]);
          acc3 = _mm512_dpbusd_epi32(acc3, inb1, weightvec1[3 + NumOutputRegs]);
          acc4 = _mm512_dpbusd_epi32(acc4, inb1, weightvec1[4 + NumOutputRegs]);
          acc5 = _mm512_dpbusd_epi32(acc5, inb1, weightvec1[5 + NumOutputRegs]);
          acc6 = _mm512_dpbusd_epi32(acc6, inb1, weightvec1[6 + NumOutputRegs]);
          acc7 = _mm512_dpbusd_epi32(acc7, inb1, weightvec1[7 + NumOutputRegs]);
          acc8 = _mm512_dpbusd_epi32(acc8, inb1, weightvec1[8 + NumOutputRegs]);
          acc9 = _mm512_dpbusd_epi32(acc9, inb1, weightvec1[9 + NumOutputRegs]);
          acc10 = _mm512_dpbusd_epi32(acc10, inb1, weightvec1[10 + NumOutputRegs]);
          acc11 = _mm512_dpbusd_epi32(acc11, inb1, weightvec1[11 + NumOutputRegs]);
          acc12 = _mm512_dpbusd_epi32(acc12, inb1, weightvec1[12 + NumOutputRegs]);
          acc13 = _mm512_dpbusd_epi32(acc13, inb1, weightvec1[13 + NumOutputRegs]);
          acc14 = _mm512_dpbusd_epi32(acc14, inb1, weightvec1[14 + NumOutputRegs]);
          acc15 = _mm512_dpbusd_epi32(acc15, inb1, weightvec1[15 + NumOutputRegs]);
          acc0 = _mm512_dpbusd_epi32(acc0, inb2, weightvec2[0 + NumOutputRegs]);
          acc1 = _mm512_dpbusd_epi32(acc1, inb2, weightvec2[1 + NumOutputRegs]);
          acc2 = _mm512_dpbusd_epi32(acc2, inb2, weightvec2[2 + NumOutputRegs]);
          acc3 = _mm512_dpbusd_epi32(acc3, inb2, weightvec2[3 + NumOutputRegs]);
          acc4 = _mm512_dpbusd_epi32(acc4, inb2, weightvec2[4 + NumOutputRegs]);
          acc5 = _mm512_dpbusd_epi32(acc5, inb2, weightvec2[5 + NumOutputRegs]);
          acc6 = _mm512_dpbusd_epi32(acc6, inb2, weightvec2[6 + NumOutputRegs]);
          acc7 = _mm512_dpbusd_epi32(acc7, inb2, weightvec2[7 + NumOutputRegs]);
          acc8 = _mm512_dpbusd_epi32(acc8, inb2, weightvec2[8 + NumOutputRegs]);
          acc9 = _mm512_dpbusd_epi32(acc9, inb2, weightvec2[9 + NumOutputRegs]);
          acc10 = _mm512_dpbusd_epi32(acc10, inb2, weightvec2[10 + NumOutputRegs]);
          acc11 = _mm512_dpbusd_epi32(acc11, inb2, weightvec2[11 + NumOutputRegs]);
          acc12 = _mm512_dpbusd_epi32(acc12, inb2, weightvec2[12 + NumOutputRegs]);
          acc13 = _mm512_dpbusd_epi32(acc13, inb2, weightvec2[13 + NumOutputRegs]);
          acc14 = _mm512_dpbusd_epi32(acc14, inb2, weightvec2[14 + NumOutputRegs]);
          acc15 = _mm512_dpbusd_epi32(acc15, inb2, weightvec2[15 + NumOutputRegs]);
          acc0 = _mm512_dpbusd_epi32(acc0, inb3, weightvec3[0 + NumOutputRegs]);
          acc1 = _mm512_dpbusd_epi32(acc1, inb3, weightvec3[1 + NumOutputRegs]);
          acc2 = _mm512_dpbusd_epi32(acc2, inb3, weightvec3[2 + NumOutputRegs]);
          acc3 = _mm512_dpbusd_epi32(acc3, inb3, weightvec3[3 + NumOutputRegs]);
          acc4 = _mm512_dpbusd_epi32(acc4, inb3, weightvec3[4 + NumOutputRegs]);
          acc5 = _mm512_dpbusd_epi32(acc5, inb3, weightvec3[5 + NumOutputRegs]);
          acc6 = _mm512_dpbusd_epi32(acc6, inb3, weightvec3[6 + NumOutputRegs]);
          acc7 = _mm512_dpbusd_epi32(acc7, inb3, weightvec3[7 + NumOutputRegs]);
          acc8 = _mm512_dpbusd_epi32(acc8, inb3, weightvec3[8 + NumOutputRegs]);
          acc9 = _mm512_dpbusd_epi32(acc9, inb3, weightvec3[9 + NumOutputRegs]);
          acc10 = _mm512_dpbusd_epi32(acc10, inb3, weightvec3[10 + NumOutputRegs]);
          acc11 = _mm512_dpbusd_epi32(acc11, inb3, weightvec3[11 + NumOutputRegs]);
          acc12 = _mm512_dpbusd_epi32(acc12, inb3, weightvec3[12 + NumOutputRegs]);
          acc13 = _mm512_dpbusd_epi32(acc13, inb3, weightvec3[13 + NumOutputRegs]);
          acc14 = _mm512_dpbusd_epi32(acc14, inb3, weightvec3[14 + NumOutputRegs]);
          acc15 = _mm512_dpbusd_epi32(acc15, inb3, weightvec3[15 + NumOutputRegs]);
          acc0 = _mm512_dpbusd_epi32(acc0, inb4, weightvec4[0 + NumOutputRegs]);
          acc1 = _mm512_dpbusd_epi32(acc1, inb4, weightvec4[1 + NumOutputRegs]);
          acc2 = _mm512_dpbusd_epi32(acc2, inb4, weightvec4[2 + NumOutputRegs]);
          acc3 = _mm512_dpbusd_epi32(acc3, inb4, weightvec4[3 + NumOutputRegs]);
          acc4 = _mm512_dpbusd_epi32(acc4, inb4, weightvec4[4 + NumOutputRegs]);
          acc5 = _mm512_dpbusd_epi32(acc5, inb4, weightvec4[5 + NumOutputRegs]);
          acc6 = _mm512_dpbusd_epi32(acc6, inb4, weightvec4[6 + NumOutputRegs]);
          acc7 = _mm512_dpbusd_epi32(acc7, inb4, weightvec4[7 + NumOutputRegs]);
          acc8 = _mm512_dpbusd_epi32(acc8, inb4, weightvec4[8 + NumOutputRegs]);
          acc9 = _mm512_dpbusd_epi32(acc9, inb4, weightvec4[9 + NumOutputRegs]);
          acc10 = _mm512_dpbusd_epi32(acc10, inb4, weightvec4[10 + NumOutputRegs]);
          acc11 = _mm512_dpbusd_epi32(acc11, inb4, weightvec4[11 + NumOutputRegs]);
          acc12 = _mm512_dpbusd_epi32(acc12, inb4, weightvec4[12 + NumOutputRegs]);
          acc13 = _mm512_dpbusd_epi32(acc13, inb4, weightvec4[13 + NumOutputRegs]);
          acc14 = _mm512_dpbusd_epi32(acc14, inb4, weightvec4[14 + NumOutputRegs]);
          acc15 = _mm512_dpbusd_epi32(acc15, inb4, weightvec4[15 + NumOutputRegs]);
          acc0 = _mm512_dpbusd_epi32(acc0, inb5, weightvec5[0 + NumOutputRegs]);
          acc1 = _mm512_dpbusd_epi32(acc1, inb5, weightvec5[1 + NumOutputRegs]);
          acc2 = _mm512_dpbusd_epi32(acc2, inb5, weightvec5[2 + NumOutputRegs]);
          acc3 = _mm512_dpbusd_epi32(acc3, inb5, weightvec5[3 + NumOutputRegs]);
          acc4 = _mm512_dpbusd_epi32(acc4, inb5, weightvec5[4 + NumOutputRegs]);
          acc5 = _mm512_dpbusd_epi32(acc5, inb5, weightvec5[5 + NumOutputRegs]);
          acc6 = _mm512_dpbusd_epi32(acc6, inb5, weightvec5[6 + NumOutputRegs]);
          acc7 = _mm512_dpbusd_epi32(acc7, inb5, weightvec5[7 + NumOutputRegs]);
          acc8 = _mm512_dpbusd_epi32(acc8, inb5, weightvec5[8 + NumOutputRegs]);
          acc9 = _mm512_dpbusd_epi32(acc9, inb5, weightvec5[9 + NumOutputRegs]);
          acc10 = _mm512_dpbusd_epi32(acc10, inb5, weightvec5[10 + NumOutputRegs]);
          acc11 = _mm512_dpbusd_epi32(acc11, inb5, weightvec5[11 + NumOutputRegs]);
          acc12 = _mm512_dpbusd_epi32(acc12, inb5, weightvec5[12 + NumOutputRegs]);
          acc13 = _mm512_dpbusd_epi32(acc13, inb5, weightvec5[13 + NumOutputRegs]);
          acc14 = _mm512_dpbusd_epi32(acc14, inb5, weightvec5[14 + NumOutputRegs]);
          acc15 = _mm512_dpbusd_epi32(acc15, inb5, weightvec5[15 + NumOutputRegs]);
          acc0 = _mm512_dpbusd_epi32(acc0, inb6, weightvec6[0 + NumOutputRegs]);
          acc1 = _mm512_dpbusd_epi32(acc1, inb6, weightvec6[1 + NumOutputRegs]);
          acc2 = _mm512_dpbusd_epi32(acc2, inb6, weightvec6[2 + NumOutputRegs]);
          acc3 = _mm512_dpbusd_epi32(acc3, inb6, weightvec6[3 + NumOutputRegs]);
          acc4 = _mm512_dpbusd_epi32(acc4, inb6, weightvec6[4 + NumOutputRegs]);
          acc5 = _mm512_dpbusd_epi32(acc5, inb6, weightvec6[5 + NumOutputRegs]);
          acc6 = _mm512_dpbusd_epi32(acc6, inb6, weightvec6[6 + NumOutputRegs]);
          acc7 = _mm512_dpbusd_epi32(acc7, inb6, weightvec6[7 + NumOutputRegs]);
          acc8 = _mm512_dpbusd_epi32(acc8, inb6, weightvec6[8 + NumOutputRegs]);
          acc9 = _mm512_dpbusd_epi32(acc9, inb6, weightvec6[9 + NumOutputRegs]);
          acc10 = _mm512_dpbusd_epi32(acc10, inb6, weightvec6[10 + NumOutputRegs]);
          acc11 = _mm512_dpbusd_epi32(acc11, inb6, weightvec6[11 + NumOutputRegs]);
          acc12 = _mm512_dpbusd_epi32(acc12, inb6, weightvec6[12 + NumOutputRegs]);
          acc13 = _mm512_dpbusd_epi32(acc13, inb6, weightvec6[13 + NumOutputRegs]);
          acc14 = _mm512_dpbusd_epi32(acc14, inb6, weightvec6[14 + NumOutputRegs]);
          acc15 = _mm512_dpbusd_epi32(acc15, inb6, weightvec6[15 + NumOutputRegs]);
          acc0 = _mm512_dpbusd_epi32(acc0, inb7, weightvec7[0 + NumOutputRegs]);
          acc1 = _mm512_dpbusd_epi32(acc1, inb7, weightvec7[1 + NumOutputRegs]);
          acc2 = _mm512_dpbusd_epi32(acc2, inb7, weightvec7[2 + NumOutputRegs]);
          acc3 = _mm512_dpbusd_epi32(acc3, inb7, weightvec7[3 + NumOutputRegs]);
          acc4 = _mm512_dpbusd_epi32(acc4, inb7, weightvec7[4 + NumOutputRegs]);
          acc5 = _mm512_dpbusd_epi32(acc5, inb7, weightvec7[5 + NumOutputRegs]);
          acc6 = _mm512_dpbusd_epi32(acc6, inb7, weightvec7[6 + NumOutputRegs]);
          acc7 = _mm512_dpbusd_epi32(acc7, inb7, weightvec7[7 + NumOutputRegs]);
          acc8 = _mm512_dpbusd_epi32(acc8, inb7, weightvec7[8 + NumOutputRegs]);
          acc9 = _mm512_dpbusd_epi32(acc9, inb7, weightvec7[9 + NumOutputRegs]);
          acc10 = _mm512_dpbusd_epi32(acc10, inb7, weightvec7[10 + NumOutputRegs]);
          acc11 = _mm512_dpbusd_epi32(acc11, inb7, weightvec7[11 + NumOutputRegs]);
          acc12 = _mm512_dpbusd_epi32(acc12, inb7, weightvec7[12 + NumOutputRegs]);
          acc13 = _mm512_dpbusd_epi32(acc13, inb7, weightvec7[13 + NumOutputRegs]);
          acc14 = _mm512_dpbusd_epi32(acc14, inb7, weightvec7[14 + NumOutputRegs]);
          acc15 = _mm512_dpbusd_epi32(acc15, inb7, weightvec7[15 + NumOutputRegs]);

          bias_vec_t* outputvec = reinterpret_cast<bias_vec_t*>(output);
          const bias_vec_t* biasvec = reinterpret_cast<const bias_vec_t*>(biases);

          // sequental load is faster, but we do not prevent the compiler from rearranging and optimizing for the target CPU
          const bias_vec_t biasvec0 = biasvec[0];
          const bias_vec_t biasvec1 = biasvec[1];
          const bias_vec_t biasvec2 = biasvec[2];
          const bias_vec_t biasvec3 = biasvec[3];

          bias_vec_t outputvec0;
          bias_vec_t outputvec1;
          bias_vec_t outputvec2;
          bias_vec_t outputvec3;

          outputvec0 = vec_haddx4(acc0, acc1, acc2, acc3, biasvec0);
          outputvec1 = vec_haddx4(acc4, acc5, acc6, acc7, biasvec1);
          outputvec2 = vec_haddx4(acc8, acc9, acc10, acc11, biasvec2);
          outputvec3 = vec_haddx4(acc12, acc13, acc14, acc15, biasvec3);

          // sequental store is faster, but the compiler may rearrange to better fit the target CPU
          outputvec[0] = outputvec0;
          outputvec[1] = outputvec1;
          outputvec[2] = outputvec2;
          outputvec[3] = outputvec3;

      }
      else
#endif
      {
          // General case for large inputs implemented via nested loops

          // Perform accumulation to registers for each big block
          for (IndexType bigBlock = 0; bigBlock < NumBigBlocks; ++bigBlock)
          {
              acc_vec_t acc[NumOutputRegs] = { vec_zero };

              // Each big block has NumOutputRegs small blocks in each "row", one per register.
              // We process two small blocks at a time to save on one addition without VNNI.
              for (IndexType smallBlock = 0; smallBlock < NumSmallBlocksPerOutput; smallBlock += 2)
              {
                  const weight_vec_t* weightvec =
                      reinterpret_cast<const weight_vec_t*>(
                          weights
                          + bigBlock * BigBlockSize
                          + smallBlock * SmallBlockSize * NumOutputRegs);

                  const in_vec_t in0 = invec[smallBlock + 0];
                  const in_vec_t in1 = invec[smallBlock + 1];

                  for (IndexType k = 0; k < NumOutputRegs; ++k)
                      vec_add_dpbusd_32x2(acc[k], in0, weightvec[k], in1, weightvec[k + NumOutputRegs]);
              }

              // Horizontally add all accumulators.
              if constexpr (NumOutputRegs % 4 == 0)
              {
                  bias_vec_t* outputvec = reinterpret_cast<bias_vec_t*>(output);
                  const bias_vec_t* biasvec = reinterpret_cast<const bias_vec_t*>(biases);

                  for (IndexType k = 0; k < NumOutputRegs; k += 4)
                  {
                      const IndexType idx = (bigBlock * NumOutputRegs + k) / 4;
                      outputvec[idx] = vec_haddx4(acc[k + 0], acc[k + 1], acc[k + 2], acc[k + 3], biasvec[idx]);
                  }
              }
              else
              {
                  for (IndexType k = 0; k < NumOutputRegs; ++k)
                  {
                      const IndexType idx = (bigBlock * NumOutputRegs + k);
                      output[idx] = vec_hadd(acc[k], biases[idx]);
                  }
              }
          }
      }

# undef vec_zero
# undef vec_add_dpbusd_32x2
# undef vec_hadd
# undef vec_haddx4
#else
      // Use old implementation for the other architectures.
      affine_transform_non_ssse3<
        InputDimensions,
        PaddedInputDimensions,
        OutputDimensions>(output, weights, biases, input);

#endif

      return output;
    }

   private:
    using BiasType = OutputType;
    using WeightType = std::int8_t;

    alignas(CacheLineSize) BiasType biases[OutputDimensions];
    alignas(CacheLineSize) WeightType weights[OutputDimensions * PaddedInputDimensions];
  };

  // A specialization for small inputs
  template <IndexType InDims, IndexType OutDims>
  class AffineTransform<InDims, OutDims, std::enable_if_t<(ceil_to_multiple<IndexType>(InDims, MaxSimdWidth) < LargeInputSize)>> {
   public:
    // Input/output type
    // Input/output type
    using InputType = std::uint8_t;
    using OutputType = std::int32_t;

    // Number of input/output dimensions
    static constexpr IndexType InputDimensions = InDims;
    static constexpr IndexType OutputDimensions = OutDims;

    static constexpr IndexType PaddedInputDimensions =
      ceil_to_multiple<IndexType>(InputDimensions, MaxSimdWidth);
    static constexpr IndexType PaddedOutputDimensions =
      ceil_to_multiple<IndexType>(OutputDimensions, MaxSimdWidth);

    using OutputBuffer = OutputType[PaddedOutputDimensions];

    static_assert(PaddedInputDimensions < LargeInputSize, "Something went wrong. This specialization (for small inputs) should not have been chosen.");

    // Hash value embedded in the evaluation file
    static constexpr std::uint32_t get_hash_value(std::uint32_t prevHash) {
      std::uint32_t hashValue = 0xCC03DAE4u;
      hashValue += OutputDimensions;
      hashValue ^= prevHash >> 1;
      hashValue ^= prevHash << 31;
      return hashValue;
    }

    static IndexType get_weight_index_scrambled(IndexType i)
    {
      return
        (i / 4) % (PaddedInputDimensions / 4) * OutputDimensions * 4 +
        i / PaddedInputDimensions * 4 +
        i % 4;
    }

    static IndexType get_weight_index(IndexType i)
    {
#if defined (USE_SSSE3)
      return get_weight_index_scrambled(i);
#else
      return i;
#endif
    }

    // Read network parameters
    bool read_parameters(std::istream& stream) {
      read_little_endian<BiasType>(stream, biases, OutputDimensions);
      for (IndexType i = 0; i < OutputDimensions * PaddedInputDimensions; ++i)
        weights[get_weight_index(i)] = read_little_endian<WeightType>(stream);

      return !stream.fail();
    }

    // Write network parameters
    bool write_parameters(std::ostream& stream) const {
      write_little_endian<BiasType>(stream, biases, OutputDimensions);

      for (IndexType i = 0; i < OutputDimensions * PaddedInputDimensions; ++i)
        write_little_endian<WeightType>(stream, weights[get_weight_index(i)]);

      return !stream.fail();
    }
    // Forward propagation
    const OutputType* propagate(
        const InputType* input, OutputType* output) const {

#if defined (USE_AVX512)
      using vec_t = __m512i;
      #define vec_setzero _mm512_setzero_si512
      #define vec_set_32 _mm512_set1_epi32
      #define vec_add_dpbusd_32 Simd::m512_add_dpbusd_epi32
      #define vec_add_dpbusd_32x2 Simd::m512_add_dpbusd_epi32x2
      #define vec_hadd Simd::m512_hadd
#elif defined (USE_AVX2)
      using vec_t = __m256i;
      #define vec_setzero _mm256_setzero_si256
      #define vec_set_32 _mm256_set1_epi32
      #define vec_add_dpbusd_32 Simd::m256_add_dpbusd_epi32
      #define vec_add_dpbusd_32x2 Simd::m256_add_dpbusd_epi32x2
      #define vec_hadd Simd::m256_hadd
#elif defined (USE_SSSE3)
      using vec_t = __m128i;
      #define vec_setzero _mm_setzero_si128
      #define vec_set_32 _mm_set1_epi32
      #define vec_add_dpbusd_32 Simd::m128_add_dpbusd_epi32
      #define vec_add_dpbusd_32x2 Simd::m128_add_dpbusd_epi32x2
      #define vec_hadd Simd::m128_hadd
#endif

#if defined (USE_SSSE3)
      const auto inputVector = reinterpret_cast<const vec_t*>(input);

      static constexpr IndexType OutputSimdWidth = sizeof(vec_t) / sizeof(OutputType);

      static_assert(OutputDimensions % OutputSimdWidth == 0 || OutputDimensions == 1);

      if constexpr (OutputDimensions % OutputSimdWidth == 0)
      {
        constexpr IndexType NumChunks = ceil_to_multiple<IndexType>(InputDimensions, 8) / 4;
        constexpr IndexType NumRegs = OutputDimensions / OutputSimdWidth;

        const auto input32 = reinterpret_cast<const std::int32_t*>(input);
        const vec_t* biasvec = reinterpret_cast<const vec_t*>(biases);
        vec_t acc[NumRegs];
        for (IndexType k = 0; k < NumRegs; ++k)
          acc[k] = biasvec[k];

        for (IndexType i = 0; i < NumChunks; i += 2)
        {
          const vec_t in0 = vec_set_32(input32[i + 0]);
          const vec_t in1 = vec_set_32(input32[i + 1]);
          const auto col0 = reinterpret_cast<const vec_t*>(&weights[(i + 0) * OutputDimensions * 4]);
          const auto col1 = reinterpret_cast<const vec_t*>(&weights[(i + 1) * OutputDimensions * 4]);
          for (IndexType k = 0; k < NumRegs; ++k)
            vec_add_dpbusd_32x2(acc[k], in0, col0[k], in1, col1[k]);
        }

        vec_t* outptr = reinterpret_cast<vec_t*>(output);
        for (IndexType k = 0; k < NumRegs; ++k)
          outptr[k] = acc[k];
      }
      else if constexpr (OutputDimensions == 1)
      {
        constexpr IndexType NumChunks = PaddedInputDimensions / SimdWidth;
        vec_t sum0 = vec_setzero();
        const auto row0 = reinterpret_cast<const vec_t*>(&weights[0]);

        for (int j = 0; j < (int)NumChunks; ++j)
        {
          const vec_t in = inputVector[j];
          vec_add_dpbusd_32(sum0, in, row0[j]);
        }
        output[0] = vec_hadd(sum0, biases[0]);
      }

# undef vec_setzero
# undef vec_set_32
# undef vec_add_dpbusd_32
# undef vec_add_dpbusd_32x2
# undef vec_hadd
#else
      // Use old implementation for the other architectures.
      affine_transform_non_ssse3<
        InputDimensions,
        PaddedInputDimensions,
        OutputDimensions>(output, weights, biases, input);
#endif

      return output;
    }

   private:
    using BiasType = OutputType;
    using WeightType = std::int8_t;

    alignas(CacheLineSize) BiasType biases[OutputDimensions];
    alignas(CacheLineSize) WeightType weights[OutputDimensions * PaddedInputDimensions];
  };

}  // namespace Stockfish::Eval::NNUE::Layers

#endif // #ifndef NNUE_LAYERS_AFFINE_TRANSFORM_H_INCLUDED
