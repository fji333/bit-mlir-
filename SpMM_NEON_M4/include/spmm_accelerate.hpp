#pragma once

#include "csr_matrix.hpp"
#include <Accelerate/Accelerate.h>
#include <vector>

// Wrapper around Apple Accelerate framework Sparse BLAS (sparse_matrix_product_dense_float)
class AccelerateSpMMRunner {
private:
    sparse_matrix_float A_acc;
    int M;
    int K;
    bool is_initialized;

public:
    AccelerateSpMMRunner() : A_acc(nullptr), M(0), K(0), is_initialized(false) {}

    ~AccelerateSpMMRunner() {
        if (is_initialized && A_acc) {
            sparse_matrix_destroy(A_acc);
        }
    }

    void load_csr(const CSRMatrix& A) {
        if (is_initialized && A_acc) {
            sparse_matrix_destroy(A_acc);
            A_acc = nullptr;
            is_initialized = false;
        }

        M = A.M;
        K = A.K;
        A_acc = sparse_matrix_create_float(M, K);

        // Apple Accelerate uses int64_t for sparse_index
        std::vector<sparse_index> temp_cols;

        for (int i = 0; i < M; ++i) {
            int nz = A.row_ptr[i + 1] - A.row_ptr[i];
            if (nz > 0) {
                int offset = A.row_ptr[i];
                temp_cols.resize(nz);
                for (int j = 0; j < nz; ++j) {
                    temp_cols[j] = static_cast<sparse_index>(A.col_indices[offset + j]);
                }

                sparse_insert_row_float(
                    A_acc,
                    i,
                    nz,
                    &A.values[offset],
                    temp_cols.data()
                );
            }
        }

        sparse_commit(A_acc);
        is_initialized = true;
    }

    void multiply(const float* __restrict__ B, float* __restrict__ C, int N) {
        if (!is_initialized || !A_acc) return;

        // C = 1.0 * A * B (order=CblasRowMajor, transa=CblasNoTrans)
        sparse_matrix_product_dense_float(
            CblasRowMajor,
            CblasNoTrans,
            N,
            1.0f,
            A_acc,
            B,
            N,
            C,
            N
        );
    }
};
