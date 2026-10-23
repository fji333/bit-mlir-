#pragma once

#include <vector>
#include <random>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

#if defined(__APPLE__)
#include <pthread.h>
#include <pthread/qos.h>
#endif

// QoS Pinning to Apple Silicon High-Performance Cores (P-Core)
inline void bind_thread_to_pcore() {
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
}

// 64-byte / 16KB cacheline & page aligned memory allocation
inline float* allocate_aligned_dense(size_t rows, size_t cols, size_t alignment = 64) {
    size_t bytes = rows * cols * sizeof(float);
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, bytes) != 0) {
        throw std::bad_alloc();
    }
    std::memset(ptr, 0, bytes);
    return reinterpret_cast<float*>(ptr);
}

inline void free_aligned(void* ptr) {
    if (ptr) {
        free(ptr);
    }
}

// Compressed Sparse Row (CSR) Representation for Matrix A (M x K)
struct CSRMatrix {
    int M; // Number of rows
    int K; // Number of columns
    int nnz; // Number of non-zero elements
    std::vector<int> row_ptr;     // Size M + 1
    std::vector<int> col_indices; // Size nnz
    std::vector<float> values;    // Size nnz

    float sparsity() const {
        double total = static_cast<double>(M) * static_cast<double>(K);
        return static_cast<float>(1.0 - (static_cast<double>(nnz) / total));
    }
};

// Generate Synthetic Random CSR Matrix with target sparsity (e.g. 0.95 = 95% zeros)
inline CSRMatrix generate_random_csr(int M, int K, float sparsity, unsigned int seed = 42) {
    CSRMatrix mat;
    mat.M = M;
    mat.K = K;
    mat.row_ptr.resize(M + 1, 0);

    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> val_dist(0.1f, 1.0f);

    float target_density = 1.0f - sparsity;
    if (target_density <= 0.0f) target_density = 0.001f;

    // First pass: generate non-zeros per row
    std::vector<std::vector<std::pair<int, float>>> temp_rows(M);
    int total_nnz = 0;

    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            if (prob_dist(gen) < target_density) {
                temp_rows[i].push_back({k, val_dist(gen)});
                total_nnz++;
            }
        }
        // Ensure at least 1 non-zero per row to prevent degenerative empty rows
        if (temp_rows[i].empty()) {
            int k_rand = gen() % K;
            temp_rows[i].push_back({k_rand, val_dist(gen)});
            total_nnz++;
        }
    }

    mat.nnz = total_nnz;
    mat.col_indices.reserve(total_nnz);
    mat.values.reserve(total_nnz);

    mat.row_ptr[0] = 0;
    for (int i = 0; i < M; ++i) {
        for (const auto& item : temp_rows[i]) {
            mat.col_indices.push_back(item.first);
            mat.values.push_back(item.second);
        }
        mat.row_ptr[i + 1] = mat.row_ptr[i] + static_cast<int>(temp_rows[i].size());
    }

    return mat;
}
