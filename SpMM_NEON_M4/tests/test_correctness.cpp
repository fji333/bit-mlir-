#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
#include "csr_matrix.hpp"
#include "spmm_reference.hpp"
#include "spmm_neon.hpp"
#include "spmm_accelerate.hpp"

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
    std::cout << ">>> Running Test: M=" << M << ", K=" << K << ", N=" << N 
              << ", Sparsity=" << (sparsity * 100.0f) << "%" << std::endl;

    CSRMatrix A = generate_random_csr(M, K, sparsity, 42);
    float* B = allocate_aligned_dense(K, N);
    float* C_ref = allocate_aligned_dense(M, N);
    float* C_neon_st = allocate_aligned_dense(M, N);
    float* C_neon_mt = allocate_aligned_dense(M, N);
    float* C_acc = allocate_aligned_dense(M, N);

    // Initialize B with random floats
    for (int i = 0; i < K * N; ++i) {
        B[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
    }

    // 1. Golden Reference
    spmm_reference(A, B, C_ref, N);

    // 2. NEON Single-Thread
    spmm_neon_single_thread(A, B, C_neon_st, N);
    bool pass_st = compare_matrices(C_ref, C_neon_st, M * N, 1e-4f, "NEON Single-Thread");

    // 3. NEON Multi-Thread
    spmm_neon_multithread(A, B, C_neon_mt, N, 4);
    bool pass_mt = compare_matrices(C_ref, C_neon_mt, M * N, 1e-4f, "NEON Multi-Thread (4P)");

    // 4. Apple Accelerate
    AccelerateSpMMRunner acc_runner;
    acc_runner.load_csr(A);
    acc_runner.multiply(B, C_acc, N);
    bool pass_acc = compare_matrices(C_ref, C_acc, M * N, 1e-4f, "Apple Accelerate");

    free_aligned(B);
    free_aligned(C_ref);
    free_aligned(C_neon_st);
    free_aligned(C_neon_mt);
    free_aligned(C_acc);

    if (!pass_st || !pass_mt || !pass_acc) {
        std::cerr << "Verification failed for size " << M << "x" << K << "x" << N << std::endl;
        std::exit(1);
    }
}

int main() {
    std::cout << "=== SpMM Correctness Tests ===" << std::endl;

    run_test(64, 64, 16, 0.90f);
    run_test(128, 128, 32, 0.95f);
    run_test(256, 256, 64, 0.98f);
    run_test(512, 512, 128, 0.99f);
    run_test(1024, 1024, 64, 0.995f);

    std::cout << "\nAll correctness tests passed." << std::endl;
    return 0;
}
