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

// Definition of layer ClippedReLU of NNUE evaluation function

#ifndef NNUE_LAYERS_SQR_CLIPPED_RELU_H_INCLUDED
#define NNUE_LAYERS_SQR_CLIPPED_RELU_H_INCLUDED

#include <algorithm>
#include <cstdint>
#include <iosfwd>

#include "../nnue_common.h"

namespace Stockfish::Eval::NNUE::Layers {

// Clipped ReLU
template<IndexType InDims>
class SqrClippedReLU {
   public:
    // Input/output type
    using InputType  = std::int32_t;
    using OutputType = std::uint8_t;

    // Number of input/output dimensions
    static constexpr IndexType InputDimensions  = InDims;
    static constexpr IndexType OutputDimensions = InputDimensions;
    static constexpr IndexType PaddedOutputDimensions =
      ceil_to_multiple<IndexType>(OutputDimensions, 32);

    using OutputBuffer = OutputType[PaddedOutputDimensions];

    // Hash value embedded in the evaluation file
    static constexpr std::uint32_t get_hash_value(std::uint32_t prevHash) {
        std::uint32_t hashValue = 0x538D24C7u;
        hashValue += prevHash;
        return hashValue;
    }

    // Read network parameters
    bool read_parameters(std::istream&) { return true; }

    // Write network parameters
    bool write_parameters(std::ostream&) const { return true; }

    std::size_t get_content_hash() const {
        std::size_t h = 0;
        hash_combine(h, get_hash_value(0));
        return h;
    }

    // Forward propagation
    void propagate(const InputType* input, OutputType* output) const {

        // SqrClippedReLU computation:
        // We shift by WeightScaleBits * 2 = 12 and divide by 128
        // which is an additional shift-right of 7, meaning 19 in total.
        // MulHi strips the lower 16 bits so we need to shift out 3 more to match.
        static_assert(WeightScaleBits == 6);

#if defined(USE_AVX512)
        // AVX-512 path: process 64 elements per iteration
        if constexpr (InputDimensions % 64 == 0)
        {
            constexpr IndexType NumChunks = InputDimensions / 64;
            // Fix for 512-bit pack lane interleaving: {0,2,4,6,1,3,5,7} in 64-bit elements
            const __m512i Offsets = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
            const auto    in      = reinterpret_cast<const __m512i*>(input);
            const auto    out     = reinterpret_cast<__m512i*>(output);
            for (IndexType i = 0; i < NumChunks; ++i)
            {
                // Pack int32 -> int16 with signed saturation
                __m512i words0 = _mm512_packs_epi32(_mm512_load_si512(&in[i * 4 + 0]),
                                                    _mm512_load_si512(&in[i * 4 + 1]));
                __m512i words1 = _mm512_packs_epi32(_mm512_load_si512(&in[i * 4 + 2]),
                                                    _mm512_load_si512(&in[i * 4 + 3]));

                // Square using mulhi (gives (a*a) >> 16), then shift right by 3
                words0 = _mm512_srli_epi16(_mm512_mulhi_epi16(words0, words0), 3);
                words1 = _mm512_srli_epi16(_mm512_mulhi_epi16(words1, words1), 3);

                // Pack int16 -> int8 with signed saturation, then fix lane ordering
                _mm512_store_si512(
                  &out[i], _mm512_permutexvar_epi64(Offsets, _mm512_packs_epi16(words0, words1)));
            }
        }
        else if constexpr (InputDimensions % 32 == 0)
        {
            // Fall back to 256-bit processing for 32-element aligned dimensions
            constexpr IndexType NumChunks = InputDimensions / 32;
            const __m256i       Offsets   = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
            const auto          in        = reinterpret_cast<const __m256i*>(input);
            const auto          out       = reinterpret_cast<__m256i*>(output);
            for (IndexType i = 0; i < NumChunks; ++i)
            {
                __m256i words0 = _mm256_packs_epi32(_mm256_load_si256(&in[i * 4 + 0]),
                                                    _mm256_load_si256(&in[i * 4 + 1]));
                __m256i words1 = _mm256_packs_epi32(_mm256_load_si256(&in[i * 4 + 2]),
                                                    _mm256_load_si256(&in[i * 4 + 3]));

                words0 = _mm256_srli_epi16(_mm256_mulhi_epi16(words0, words0), 3);
                words1 = _mm256_srli_epi16(_mm256_mulhi_epi16(words1, words1), 3);

                _mm256_store_si256(&out[i], _mm256_permutevar8x32_epi32(
                                              _mm256_packs_epi16(words0, words1), Offsets));
            }
        }
        else
        {
            // Fall back to 128-bit processing for smaller dimensions
            constexpr IndexType NumChunks = InputDimensions / 16;
            const auto          in        = reinterpret_cast<const __m128i*>(input);
            const auto          out       = reinterpret_cast<__m128i*>(output);
            for (IndexType i = 0; i < NumChunks; ++i)
            {
                __m128i words0 =
                  _mm_packs_epi32(_mm_load_si128(&in[i * 4 + 0]), _mm_load_si128(&in[i * 4 + 1]));
                __m128i words1 =
                  _mm_packs_epi32(_mm_load_si128(&in[i * 4 + 2]), _mm_load_si128(&in[i * 4 + 3]));

                words0 = _mm_srli_epi16(_mm_mulhi_epi16(words0, words0), 3);
                words1 = _mm_srli_epi16(_mm_mulhi_epi16(words1, words1), 3);

                _mm_store_si128(&out[i], _mm_packs_epi16(words0, words1));
            }
        }
        constexpr IndexType Start = InputDimensions % 64 == 0 ? InputDimensions
                                  : InputDimensions % 32 == 0 ? InputDimensions
                                                              : (InputDimensions / 16) * 16;

#elif defined(USE_AVX2)
        // AVX2 path: process 32 elements per iteration
        if constexpr (InputDimensions % 32 == 0)
        {
            constexpr IndexType NumChunks = InputDimensions / 32;
            const __m256i       Offsets   = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
            const auto          in        = reinterpret_cast<const __m256i*>(input);
            const auto          out       = reinterpret_cast<__m256i*>(output);
            for (IndexType i = 0; i < NumChunks; ++i)
            {
                __m256i words0 = _mm256_packs_epi32(_mm256_load_si256(&in[i * 4 + 0]),
                                                    _mm256_load_si256(&in[i * 4 + 1]));
                __m256i words1 = _mm256_packs_epi32(_mm256_load_si256(&in[i * 4 + 2]),
                                                    _mm256_load_si256(&in[i * 4 + 3]));

                words0 = _mm256_srli_epi16(_mm256_mulhi_epi16(words0, words0), 3);
                words1 = _mm256_srli_epi16(_mm256_mulhi_epi16(words1, words1), 3);

                _mm256_store_si256(&out[i], _mm256_permutevar8x32_epi32(
                                              _mm256_packs_epi16(words0, words1), Offsets));
            }
        }
        else
        {
            // Fall back to 128-bit processing for smaller dimensions
            constexpr IndexType NumChunks = InputDimensions / 16;
            const auto          in        = reinterpret_cast<const __m128i*>(input);
            const auto          out       = reinterpret_cast<__m128i*>(output);
            for (IndexType i = 0; i < NumChunks; ++i)
            {
                __m128i words0 =
                  _mm_packs_epi32(_mm_load_si128(&in[i * 4 + 0]), _mm_load_si128(&in[i * 4 + 1]));
                __m128i words1 =
                  _mm_packs_epi32(_mm_load_si128(&in[i * 4 + 2]), _mm_load_si128(&in[i * 4 + 3]));

                words0 = _mm_srli_epi16(_mm_mulhi_epi16(words0, words0), 3);
                words1 = _mm_srli_epi16(_mm_mulhi_epi16(words1, words1), 3);

                _mm_store_si128(&out[i], _mm_packs_epi16(words0, words1));
            }
        }
        constexpr IndexType Start =
          InputDimensions % 32 == 0 ? InputDimensions : (InputDimensions / 16) * 16;

#elif defined(USE_SSE2)
        constexpr IndexType NumChunks = InputDimensions / 16;

        const auto in  = reinterpret_cast<const __m128i*>(input);
        const auto out = reinterpret_cast<__m128i*>(output);
        for (IndexType i = 0; i < NumChunks; ++i)
        {
            __m128i words0 =
              _mm_packs_epi32(_mm_load_si128(&in[i * 4 + 0]), _mm_load_si128(&in[i * 4 + 1]));
            __m128i words1 =
              _mm_packs_epi32(_mm_load_si128(&in[i * 4 + 2]), _mm_load_si128(&in[i * 4 + 3]));

            words0 = _mm_srli_epi16(_mm_mulhi_epi16(words0, words0), 3);
            words1 = _mm_srli_epi16(_mm_mulhi_epi16(words1, words1), 3);

            _mm_store_si128(&out[i], _mm_packs_epi16(words0, words1));
        }
        constexpr IndexType Start = NumChunks * 16;

#else
        constexpr IndexType Start = 0;
#endif

        for (IndexType i = Start; i < InputDimensions; ++i)
        {
            output[i] = static_cast<OutputType>(
              // Really should be /127 but we need to make it fast so we right-shift
              // by an extra 7 bits instead. Needs to be accounted for in the trainer.
              std::min(127ll, ((long long) (input[i]) * input[i]) >> (2 * WeightScaleBits + 7)));
        }
    }
};

}  // namespace Stockfish::Eval::NNUE::Layers

#endif  // NNUE_LAYERS_SQR_CLIPPED_RELU_H_INCLUDED
