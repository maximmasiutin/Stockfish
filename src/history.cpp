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

#include "history.h"

#include <cstdint>

namespace Stockfish {

std::uint16_t rol16(std::uint16_t x, unsigned n) {
    n &= 15u;
    return static_cast<std::uint16_t>((x << n) | (x >> ((-n) & 15u)));
}

std::uint16_t magic_index(std::uint16_t raw) {
    std::uint32_t r32;
    std::uint16_t r;
    r32 = rol16(raw, 10);
    r32 = r32 ^ (r32 >> 10);
    r   = rol16(static_cast<std::uint16_t>(r32), 7);
    r   = static_cast<std::uint16_t>(r ^ 0x2CA9u);
    return r;
}

}  // namespace Stockfish
