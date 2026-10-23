#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
#include "csr_matrix.hpp"
#include "spmm_reference.hpp"
#include "spmm_advanced.hpp"

bool compare_matrices(const float* ref, const float* test, int size, float tol = 1e-4f, const std::string& name = "") {
    float max_diff = 0.0f;
    double l2_err = 0.0;
    for (int i = 0; i < size; ++i) {
        float diff = std::abs(ref[i] - test[i]);
        if (diff > max_diff) max_diff = diff;
        l2_err += diff * diff;
    }
    l2_err = std::sqrt(l2_err / size);

    if (max_diff > tol) {
        std::cout << "  [FAIL] " << name << " | Max Diff: " << max_diff << ", RMSE: " << l2_err << std::endl;
        return false;
    } else {
        std::cout << "  [PASS] " << name << " | Max Diff: " << max_diff << ", RMSE: " << l2_err << std::endl;
        return true;
    }
}

void run_test(int M, int K, int N, float sparsity) {
    std::cout << ">>> Testing: M=" << M << ", K=" << K << ", N=" << N 
              << ", Sparsity=" << (sparsity * 100.0f) << "%" << std::endl;

    CSRMatrix A = generate_random_csr(M, K, sparsity, 42);
    float* B = allocate_aligned_dense(K, N);
    float* C_ref = allocate_aligned_dense(M, N);
    float* C_u2 = allocate_aligned_dense(M, N);
    float* C_u4 = allocate_aligned_dense(M, N);
    float* C_u6 = allocate_aligned_dense(M, N);
    float* C_2x16 = allocate_aligned_dense(M, N);
    float* C_4x16 = allocate_aligned_dense(M, N);
    float* C_8x16 = allocate_aligned_dense(M, N);

    for (int i = 0; i < K * N; ++i) {
        B[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
    }

    spmm_reference(A, B, C_ref, N);

    spmm_neon_u2_single_thread(A, B, C_u2, N);
    bool pass_u2 = compare_matrices(C_ref, C_u2, M * N, 1e-4f, "NEON U=2");

    spmm_neon_u4_single_thread(A, B, C_u4, N);
    bool pass_u4 = compare_matrices(C_ref, C_u4, M * N, 1e-4f, "NEON U=4");

    spmm_neon_u6_single_thread(A, B, C_u6, N);
    bool pass_u6 = compare_matrices(C_ref, C_u6, M * N, 1e-4f, "NEON U=6");

    spmm_neon_2x16_single_thread(A, B, C_2x16, N);
    bool pass_2x16 = compare_matrices(C_ref, C_2x16, M * N, 1e-4f, "NEON Tile 2x16");

    spmm_neon_4x16_single_thread(A, B, C_4x16, N);
    bool pass_4x16 = compare_matrices(C_ref, C_4x16, M * N, 1e-4f, "NEON Tile 4x16");

    spmm_neon_8x16_single_thread(A, B, C_8x16, N);
    bool pass_8x16 = compare_matrices(C_ref, C_8x16, M * N, 1e-4f, "NEON Tile 8x16");

    free_aligned(B);
    free_aligned(C_ref);
    free_aligned(C_u2);
    free_aligned(C_u4);
    free_aligned(C_u6);
    free_aligned(C_2x16);
    free_aligned(C_4x16);
    free_aligned(C_8x16);

    if (!pass_u2 || !pass_u4 || !pass_u6 || !pass_2x16 || !pass_4x16 || !pass_8x16) {
        std::cerr << "Verification failed!" << std::endl;
        std::exit(1);
    }
}

int main() {
    std::cout << "=== Advanced Optimization Variants Correctness Tests ===" << std::endl;
    run_test(64, 64, 16, 0.90f);
    run_test(128, 128, 32, 0.95f);
    run_test(256, 256, 64, 0.98f);
    run_test(512, 512, 128, 0.99f);
    std::cout << "\nAll advanced correctness tests passed.\n" << std::endl;
    return 0;
}
