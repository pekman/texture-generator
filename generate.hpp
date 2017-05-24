#pragma once

#include <stdexcept>

void generate(
    const std::string &cloudfile,
    const std::string &polygonfile,
    const std::string &outfile_base,
    unsigned texture_size,
    unsigned points_per_texel,
    float max_sqr_dist);
