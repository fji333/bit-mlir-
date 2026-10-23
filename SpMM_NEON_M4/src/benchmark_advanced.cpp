#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cmath>
#include <algorithm>
#include "csr_matrix.hpp"
#include "spmm_reference.hpp"
#include "spmm_neon.hpp"
#include "spmm_advanced.hpp"
#include "spmm_accelerate.hpp"

template <typename Func>
double time_kernel(Func&& func, int warmup = 3, int iterations = 10) {
    for (int i = 0; i < warmup; ++i) {
        func();
    }
    std::vector<double> times;
    times.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        times.push_back(elapsed.count());
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

void run_ablation_case(int M, int K, int N, float sparsity) {
    CSRMatrix A = generate_random_csr(M, K, sparsity, 2026);
    float* B = allocate_aligned_dense(K, N);
    float* C = allocate_aligned_dense(M, N);

    for (int i = 0; i < K * N; ++i) {
        B[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
    }

    double total_flops = 2.0 * static_cast<double>(A.nnz) * static_cast<double>(N);

    // 1. Apple Accelerate
    AccelerateSpMMRunner acc_runner;
    acc_runner.load_csr(A);
    double acc_ms = time_kernel([&]() {
        acc_runner.multiply(B, C, N);
    }, 2, 7);

    // 2. Baseline 1x16 (U=1)
    double t_u1 = time_kernel([&]() {
        spmm_neon_single_thread(A, B, C, N);
    }, 3, 10);

    // 3. U=2
    double t_u2 = time_kernel([&]() {
        spmm_neon_u2_single_thread(A, B, C, N);
    }, 3, 10);

    // 4. U=4
    double t_u4 = time_kernel([&]() {
        spmm_neon_u4_single_thread(A, B, C, N);
    }, 3, 10);

    // 5. U=6
    double t_u6 = time_kernel([&]() {
        spmm_neon_u6_single_thread(A, B, C, N);
    }, 3, 10);

    // 6. Tile 2x16
    double t_2x16 = time_kernel([&]() {
        spmm_neon_2x16_single_thread(A, B, C, N);
    }, 3, 10);

    // 7. Tile 4x16
    double t_4x16 = time_kernel([&]() {
        spmm_neon_4x16_single_thread(A, B, C, N);
    }, 3, 10);

    // 8. Tile 8x16
    double t_8x16 = time_kernel([&]() {
        spmm_neon_8x16_single_thread(A, B, C, N);
    }, 3, 10);

    free_aligned(B);
    free_aligned(C);

    auto gflops = [&](double ms) {
        return (total_flops / (ms * 1e-3)) / 1e9;
    };

    std::cout << "| " << M << "x" << K << ", N=" << N << ", s=" << (int)(sparsity * 100) << "%"
              << " | " << std::fixed << std::setprecision(2) << acc_ms << " ms"
              << " | " << std::fixed << std::setprecision(2) << t_u1   << " ms (" << gflops(t_u1)   << "G)"
              << " | " << std::fixed << std::setprecision(2) << t_u2   << " ms (" << gflops(t_u2)   << "G)"
              << " | " << std::fixed << std::setprecision(2) << t_u4   << " ms (" << gflops(t_u4)   << "G)"
              << " | " << std::fixed << std::setprecision(2) << t_u6   << " ms (" << gflops(t_u6)   << "G)"
              << " | " << std::fixed << std::setprecision(2) << t_2x16 << " ms (" << gflops(t_2x16) << "G)"
              << " | " << std::fixed << std::setprecision(2) << t_4x16 << " ms (" << gflops(t_4x16) << "G)"
              << " | " << std::fixed << std::setprecision(2) << t_8x16 << " ms (" << gflops(t_8x16) << "G) |"
              << std::endl;
}

int main() {
    bind_thread_to_pcore();

    std::cout << "========================================================================================================================================" << std::endl;
    std::cout << "  Comprehensive Ablation Study on Apple M4: U=1, U=2, U=4, U=6, Tile 2x16, Tile 4x16, Tile 8x16 vs Apple Accelerate                      " << std::endl;
    std::cout << "========================================================================================================================================" << std::endl;

    std::cout << "\n| Config | Accelerate | U=1 (Base) | U=2 | U=4 | U=6 | Tile 2x16 | Tile 4x16 | Tile 8x16 |" << std::endl;
    std::cout << "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |" << std::endl;

    run_ablation_case(1024, 1024, 64, 0.90f);
    run_ablation_case(1024, 1024, 128, 0.95f);
    run_ablation_case(2048, 2048, 64, 0.95f);
    run_ablation_case(2048, 2048, 128, 0.98f);
    run_ablation_case(4096, 4096, 64, 0.98f);
    run_ablation_case(4096, 4096, 128, 0.99f);
    run_ablation_case(8192, 8192, 128, 0.99f);
    run_ablation_case(8192, 8192, 256, 0.995f);

    std::cout << "========================================================================================================================================" << std::endl;
    return 0;
}
