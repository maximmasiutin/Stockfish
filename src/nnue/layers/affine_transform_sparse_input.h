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

// Definition of layer AffineTransformSparseInput of NNUE evaluation function

#ifndef NNUE_LAYERS_AFFINE_TRANSFORM_SPARSE_INPUT_H_INCLUDED
#define NNUE_LAYERS_AFFINE_TRANSFORM_SPARSE_INPUT_H_INCLUDED

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>

#include "../../bitboard.h"
#include "../simd.h"
#include "../nnue_common.h"

/*
  This file contains the definition for a fully connected layer (aka affine transform) with block sparse input.
*/

namespace Stockfish::Eval::NNUE::Layers {

#if (USE_SSSE3 | (USE_NEON >= 8))
static constexpr int lsb_index64[64] = {
  0,  47, 1,  56, 48, 27, 2,  60, 57, 49, 41, 37, 28, 16, 3,  61, 54, 58, 35, 52, 50, 42,
  21, 44, 38, 32, 29, 23, 17, 11, 4,  62, 46, 55, 26, 59, 40, 36, 15, 53, 34, 51, 20, 43,
  31, 22, 10, 45, 25, 39, 14, 33, 19, 30, 9,  24, 13, 18, 8,  12, 7,  6,  5,  63};

constexpr int constexpr_lsb(uint64_t bb) {
    assert(bb != 0);
    constexpr uint64_t debruijn64 = 0x03F79D71B4CB0A89ULL;
    return lsb_index64[((bb ^ (bb - 1)) * debruijn64) >> 58];
}

alignas(CacheLineSize) static constexpr struct OffsetIndices {

    std::uint16_t offset_indices[256][8];

    constexpr OffsetIndices() :
        offset_indices() {
        for (int i = 0; i < 256; ++i)
        {
            std::uint64_t j = i, k = 0;
            while (j)
            {
                offset_indices[i][k++] = constexpr_lsb(j);
                j &= j - 1;
            }
            while (k < 8)
                offset_indices[i][k++] = 0;
        }
    }

} Lookup;

    #if defined(__GNUC__) || defined(__clang__)
        #define RESTRICT __restrict__
    #elif defined(_MSC_VER)
        #define RESTRICT __restrict
    #else
        #define RESTRICT
    #endif

// Precomputed base indices for find_nnz - eliminates loop-carried dependency
// Order matches _mm512_packus_epi32 lane-crossing behavior
alignas(64) static constexpr std::int16_t FindNnzBases[8][32] = {
  {0, 1, 2,  3,  16, 17, 18, 19, 4,  5,  6,  7,  20, 21, 22, 23,
   8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31},
  {32, 33, 34, 35, 48, 49, 50, 51, 36, 37, 38, 39, 52, 53, 54, 55,
   40, 41, 42, 43, 56, 57, 58, 59, 44, 45, 46, 47, 60, 61, 62, 63},
  {64, 65, 66, 67, 80, 81, 82, 83, 68, 69, 70, 71, 84, 85, 86, 87,
   72, 73, 74, 75, 88, 89, 90, 91, 76, 77, 78, 79, 92, 93, 94, 95},
  {96,  97,  98,  99,  112, 113, 114, 115, 100, 101, 102, 103, 116, 117, 118, 119,
   104, 105, 106, 107, 120, 121, 122, 123, 108, 109, 110, 111, 124, 125, 126, 127},
  {128, 129, 130, 131, 144, 145, 146, 147, 132, 133, 134, 135, 148, 149, 150, 151,
   136, 137, 138, 139, 152, 153, 154, 155, 140, 141, 142, 143, 156, 157, 158, 159},
  {160, 161, 162, 163, 176, 177, 178, 179, 164, 165, 166, 167, 180, 181, 182, 183,
   168, 169, 170, 171, 184, 185, 186, 187, 172, 173, 174, 175, 188, 189, 190, 191},
  {192, 193, 194, 195, 208, 209, 210, 211, 196, 197, 198, 199, 212, 213, 214, 215,
   200, 201, 202, 203, 216, 217, 218, 219, 204, 205, 206, 207, 220, 221, 222, 223},
  {224, 225, 226, 227, 240, 241, 242, 243, 228, 229, 230, 231, 244, 245, 246, 247,
   232, 233, 234, 235, 248, 249, 250, 251, 236, 237, 238, 239, 252, 253, 254, 255},
};

// Find indices of nonzero numbers in an int32_t array
template<const IndexType InputDimensions>
void find_nnz(const std::int32_t* RESTRICT input,
              std::uint16_t* RESTRICT      out,
              IndexType&                   count_out) {

    #if defined(USE_AVX512ICL)

    constexpr IndexType SimdWidthIn  = 16;  // 512 bits / 32 bits
    constexpr IndexType SimdWidthOut = 32;  // 512 bits / 16 bits
    constexpr IndexType NumChunks    = InputDimensions / SimdWidthOut;

    IndexType count = 0;

    // Process 2 chunks per iteration for better ILP
    IndexType i = 0;
    for (; i + 1 < NumChunks; i += 2)
    {
        // Prefetch next iteration's data
        _mm_prefetch(reinterpret_cast<const char*>(input + (i + 2) * 2 * SimdWidthIn), _MM_HINT_T0);

        // Load 4 vectors (2 chunks worth)
        const __m512i in0 = _mm512_load_si512(input + i * 2 * SimdWidthIn);
        const __m512i in1 = _mm512_load_si512(input + i * 2 * SimdWidthIn + SimdWidthIn);
        const __m512i in2 = _mm512_load_si512(input + (i + 1) * 2 * SimdWidthIn);
        const __m512i in3 = _mm512_load_si512(input + (i + 1) * 2 * SimdWidthIn + SimdWidthIn);

        // Load precomputed bases (no loop-carried dependency)
        const __m512i base0 = _mm512_load_si512(reinterpret_cast<const __m512i*>(FindNnzBases[i]));
        const __m512i base1 =
          _mm512_load_si512(reinterpret_cast<const __m512i*>(FindNnzBases[i + 1]));

        // Pack and test - both chunks in parallel
        const __m512i   packed0 = _mm512_packus_epi32(in0, in1);
        const __m512i   packed1 = _mm512_packus_epi32(in2, in3);
        const __mmask32 mask0   = _mm512_test_epi16_mask(packed0, packed0);
        const __mmask32 mask1   = _mm512_test_epi16_mask(packed1, packed1);

        // Compress and store first chunk
        const __m512i nnz0 = _mm512_maskz_compress_epi16(mask0, base0);
        _mm512_storeu_si512(out + count, nnz0);
        count += popcount(mask0);

        // Compress and store second chunk
        const __m512i nnz1 = _mm512_maskz_compress_epi16(mask1, base1);
        _mm512_storeu_si512(out + count, nnz1);
        count += popcount(mask1);
    }

    // Handle remaining chunk if NumChunks is odd
    for (; i < NumChunks; ++i)
    {
        const __m512i in0  = _mm512_load_si512(input + i * 2 * SimdWidthIn);
        const __m512i in1  = _mm512_load_si512(input + i * 2 * SimdWidthIn + SimdWidthIn);
        const __m512i base = _mm512_load_si512(reinterpret_cast<const __m512i*>(FindNnzBases[i]));

        const __m512i   packed  = _mm512_packus_epi32(in0, in1);
        const __mmask32 nnzMask = _mm512_test_epi16_mask(packed, packed);

        const __m512i nnz = _mm512_maskz_compress_epi16(nnzMask, base);
        _mm512_storeu_si512(out + count, nnz);
        count += popcount(nnzMask);
    }

    count_out = count;

    #elif defined(USE_AVX512)

    constexpr IndexType SimdWidth = 16;  // 512 bits / 32 bits
    constexpr IndexType NumChunks = InputDimensions / SimdWidth;
    const __m512i       increment = _mm512_set1_epi32(SimdWidth);
    __m512i base = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

    IndexType count = 0;
    for (IndexType i = 0; i < NumChunks; ++i)
    {
        const __m512i inputV = _mm512_load_si512(input + i * SimdWidth);

        // Get a bitmask and gather non zero indices
        const __mmask16 nnzMask = _mm512_test_epi32_mask(inputV, inputV);
        const __m512i   nnzV    = _mm512_maskz_compress_epi32(nnzMask, base);
        _mm512_mask_cvtepi32_storeu_epi16(out + count, 0xFFFF, nnzV);
        count += popcount(nnzMask);
        base = _mm512_add_epi32(base, increment);
    }
    count_out = count;

    #else

    using namespace SIMD;

    constexpr IndexType InputSimdWidth = sizeof(vec_uint_t) / sizeof(std::int32_t);
    // Outputs are processed 8 elements at a time, even if the SIMD width is narrower
    constexpr IndexType ChunkSize      = 8;
    constexpr IndexType NumChunks      = InputDimensions / ChunkSize;
    constexpr IndexType InputsPerChunk = ChunkSize / InputSimdWidth;

    static_assert(InputsPerChunk > 0 && "SIMD width too wide");

    const auto     inputVector = reinterpret_cast<const vec_uint_t*>(input);
    IndexType      count       = 0;
    vec128_t       base        = vec128_zero;
    const vec128_t increment   = vec128_set_16(8);
    for (IndexType i = 0; i < NumChunks; ++i)
    {
        // bitmask of nonzero values in this chunk
        unsigned nnz = 0;
        for (IndexType j = 0; j < InputsPerChunk; ++j)
        {
            const vec_uint_t inputChunk = inputVector[i * InputsPerChunk + j];
            nnz |= unsigned(vec_nnz(inputChunk)) << (j * InputSimdWidth);
        }
        const vec128_t offsets =
          vec128_load(reinterpret_cast<const vec128_t*>(&Lookup.offset_indices[nnz]));
        vec128_storeu(reinterpret_cast<vec128_t*>(out + count), vec128_add(base, offsets));
        count += popcount(nnz);
        base = vec128_add(base, increment);
    }
    count_out = count;
    #endif
}

#endif

// Sparse input implementation
template<IndexType InDims, IndexType OutDims>
class AffineTransformSparseInput {
   public:
    // Input/output type
    using InputType  = std::uint8_t;
    using OutputType = std::int32_t;

    // Number of input/output dimensions
    static constexpr IndexType InputDimensions  = InDims;
    static constexpr IndexType OutputDimensions = OutDims;

    static_assert(OutputDimensions % 16 == 0,
                  "Only implemented for OutputDimensions divisible by 16.");

    static constexpr IndexType PaddedInputDimensions =
      ceil_to_multiple<IndexType>(InputDimensions, MaxSimdWidth);
    static constexpr IndexType PaddedOutputDimensions =
      ceil_to_multiple<IndexType>(OutputDimensions, MaxSimdWidth);

#if (USE_SSSE3 | (USE_NEON >= 8))
    static constexpr IndexType ChunkSize = 4;
#else
    static constexpr IndexType ChunkSize = 1;
#endif

    using OutputBuffer = OutputType[PaddedOutputDimensions];

    // Hash value embedded in the evaluation file
    static constexpr std::uint32_t get_hash_value(std::uint32_t prevHash) {
        std::uint32_t hashValue = 0xCC03DAE4u;
        hashValue += OutputDimensions;
        hashValue ^= prevHash >> 1;
        hashValue ^= prevHash << 31;
        return hashValue;
    }

    static constexpr IndexType get_weight_index_scrambled(IndexType i) {
        return (i / ChunkSize) % (PaddedInputDimensions / ChunkSize) * OutputDimensions * ChunkSize
             + i / PaddedInputDimensions * ChunkSize + i % ChunkSize;
    }

    static constexpr IndexType get_weight_index(IndexType i) {
#if (USE_SSSE3 | (USE_NEON >= 8))
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

    std::size_t get_content_hash() const {
        std::size_t h = 0;
        hash_combine(h, get_raw_data_hash(biases));
        hash_combine(h, get_raw_data_hash(weights));
        hash_combine(h, get_hash_value(0));
        return h;
    }

    // Forward propagation
    void propagate(const InputType* input, OutputType* output) const {

#if (USE_SSSE3 | (USE_NEON >= 8))
    #if defined(USE_AVX512)
        using invec_t  = __m512i;
        using outvec_t = __m512i;
        #define vec_add_32 _mm512_add_epi32
        #define vec_set_32 _mm512_set1_epi32
        #define vec_add_dpbusd_32 SIMD::m512_add_dpbusd_epi32
    #elif defined(USE_AVX2)
        using invec_t  = __m256i;
        using outvec_t = __m256i;
        #define vec_add_32 _mm256_add_epi32
        #define vec_set_32 _mm256_set1_epi32
        #define vec_add_dpbusd_32 SIMD::m256_add_dpbusd_epi32
    #elif defined(USE_SSSE3)
        using invec_t  = __m128i;
        using outvec_t = __m128i;
        #define vec_set_32 _mm_set1_epi32
        #define vec_add_dpbusd_32 SIMD::m128_add_dpbusd_epi32
    #elif defined(USE_NEON_DOTPROD)
        using invec_t  = int8x16_t;
        using outvec_t = int32x4_t;
        #define vec_set_32(a) vreinterpretq_s8_u32(vdupq_n_u32(a))
        #define vec_add_dpbusd_32 SIMD::dotprod_m128_add_dpbusd_epi32
    #elif defined(USE_NEON)
        using invec_t  = int8x16_t;
        using outvec_t = int32x4_t;
        #define vec_set_32(a) vreinterpretq_s8_u32(vdupq_n_u32(a))
        #define vec_add_dpbusd_32 SIMD::neon_m128_add_dpbusd_epi32
    #endif
        constexpr IndexType OutputSimdWidth = sizeof(outvec_t) / sizeof(OutputType);
        constexpr IndexType NumChunks = ceil_to_multiple<IndexType>(InputDimensions, 8) / ChunkSize;
        constexpr IndexType NumAccums = OutputDimensions / OutputSimdWidth;
        // If we're using high-latency dot product instructions, split the accumulators
        // to create 3 separate dependency chains and merge at the end
        constexpr IndexType NumRegs =
    #if defined(USE_VNNI)
          3 * NumAccums;
    #else
          NumAccums;
    #endif
        std::uint16_t nnz[NumChunks];
        IndexType     count;

        const auto input32 = reinterpret_cast<const std::int32_t*>(input);

        // Find indices of nonzero 32-bit blocks
        find_nnz<NumChunks>(input32, nnz, count);

        const outvec_t* biasvec = reinterpret_cast<const outvec_t*>(biases);
        outvec_t        acc[NumRegs];
        for (IndexType k = 0; k < NumAccums; ++k)
            acc[k] = biasvec[k];

        const auto* start = nnz;
        const auto* end   = nnz + count;

        // convince GCC to not do weird pointer arithmetic in the following loop
        const std::int8_t* weights_cp = weights;
    #if defined(USE_VNNI)
        for (IndexType k = NumAccums; k < NumRegs; ++k)
            acc[k] = vec_zero();

        while (start < end - 2)
        {
            const std::ptrdiff_t i0  = *start++;
            const std::ptrdiff_t i1  = *start++;
            const std::ptrdiff_t i2  = *start++;
            const invec_t        in0 = vec_set_32(input32[i0]);
            const invec_t        in1 = vec_set_32(input32[i1]);
            const invec_t        in2 = vec_set_32(input32[i2]);
            const auto           col0 =
              reinterpret_cast<const invec_t*>(&weights_cp[i0 * OutputDimensions * ChunkSize]);
            const auto col1 =
              reinterpret_cast<const invec_t*>(&weights_cp[i1 * OutputDimensions * ChunkSize]);
            const auto col2 =
              reinterpret_cast<const invec_t*>(&weights_cp[i2 * OutputDimensions * ChunkSize]);
            for (IndexType k = 0; k < NumAccums; ++k)
            {
                vec_add_dpbusd_32(acc[k], in0, col0[k]);
                vec_add_dpbusd_32(acc[k + NumAccums], in1, col1[k]);
                vec_add_dpbusd_32(acc[k + 2 * NumAccums], in2, col2[k]);
            }
        }
        for (IndexType k = 0; k < NumAccums; ++k)
            acc[k] = vec_add_32(vec_add_32(acc[k], acc[k + NumAccums]), acc[k + 2 * NumAccums]);
    #endif
        while (start < end)
        {
            const std::ptrdiff_t i  = *start++;
            const invec_t        in = vec_set_32(input32[i]);
            const auto           col =
              reinterpret_cast<const invec_t*>(&weights_cp[i * OutputDimensions * ChunkSize]);
            for (IndexType k = 0; k < NumAccums; ++k)
                vec_add_dpbusd_32(acc[k], in, col[k]);
        }

        outvec_t* outptr = reinterpret_cast<outvec_t*>(output);
        for (IndexType k = 0; k < NumAccums; ++k)
            outptr[k] = acc[k];

    #undef vec_set_32
    #undef vec_add_dpbusd_32
    #ifdef vec_add_32
        #undef vec_add_32
    #endif
#else
        // Use dense implementation for the other architectures.
        affine_transform_non_ssse3<InputDimensions, PaddedInputDimensions, OutputDimensions>(
          output, weights, biases, input);
#endif
    }

   private:
    using BiasType   = OutputType;
    using WeightType = std::int8_t;

    alignas(CacheLineSize) BiasType biases[OutputDimensions];
    alignas(CacheLineSize) WeightType weights[OutputDimensions * PaddedInputDimensions];
};

}  // namespace Stockfish::Eval::NNUE::Layers

#endif  // #ifndef NNUE_LAYERS_AFFINE_TRANSFORM_SPARSE_INPUT_H_INCLUDED
