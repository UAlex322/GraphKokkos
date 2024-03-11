#pragma once

#include "matrix.h"
#include <random>
#include <unordered_set>

spMtx<int> generate_mask(const size_t n, const size_t min_deg, const size_t max_deg) {
    std::unordered_set<int> positions;
    std::uniform_int_distribution<int> deg_distr(min_deg, max_deg);
    std::uniform_int_distribution<int> col_distr(0, n - 1);
    std::mt19937 generator{std::random_device{}()};

    spMtx<int> mask;
    mask.m = mask.n = n;
    mask.resizeRows(mask.m + 1);

    mask.Rst(0) = 0;
    for (size_t i = 0; i < n; ++i)
        mask.Rst(i+1) = mask.Rst(i) + deg_distr(generator);
    mask.nz = mask.Rst(n);
    Kokkos::resize(mask.Col, mask.nz);

    size_t j = 0;
    for (size_t i = 0; i < mask.n; ++i) {
        size_t deg = mask.Rst(i+1) - mask.Rst(i);
        while (positions.size() < deg)
            positions.insert(col_distr(generator));
        for (const int column : positions)
            mask.Col(j++) = column;
        positions.clear();
    }

    return mask;
}

template<typename T>
void full_mask(spMtx<T> &mask, const size_t n) {
    mask.m = n;
    mask.nz = n*n;
    Kokkos::resize(mask.Rst, n + 1);
    Kokkos::resize(mask.Col, n * n);

    for (size_t i = 0; i <= n; ++i)
        mask.Rst(i) = i*n;
    size_t curr_pos = 0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j)
            mask.Col(curr_pos++) = j;
    }
}

spMtx<int> generate_adjacency_matrix(const size_t n, const size_t min_deg, const size_t max_deg) {
    spMtx<int> Res = generate_mask(n, min_deg, max_deg);

    Kokkos::resize(Res.Val, Res.nz);
    for (size_t j = 0; j < Res.nz; ++j)
        Res.Val(j) = 1;

    return build_symm_from_lower(extract_lower_triangle(Res));
}