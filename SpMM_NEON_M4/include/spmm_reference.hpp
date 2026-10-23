#pragma once

#include "csr_matrix.hpp"
#include <cstring>

// Scalar reference implementation for CSR SpMM: C = A * B
// A: CSRMatrix (M x K)
// B: Dense matrix (K x N), row-major
// C: Dense matrix (M x N), row-major
inline void spmm_reference(
    const CSRMatrix& A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    // Zero out output buffer
    std::memset(C, 0, A.M * N * sizeof(float));

    for (int i = 0; i < A.M; ++i) {
        int row_start = A.row_ptr[i];
        int row_end   = A.row_ptr[i + 1];
        float* C_row = C + i * N;

        for (int p = row_start; p < row_end; ++p) {
            int k = A.col_indices[p];
            float a_val = A.values[p];
            const float* B_row = B + k * N;

            for (int j = 0; j < N; ++j) {
                C_row[j] += a_val * B_row[j];
            }
        }
    }
}
