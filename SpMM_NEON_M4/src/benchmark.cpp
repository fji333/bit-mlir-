#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cmath>
#include <algorithm>
#include "csr_matrix.hpp"
#include "spmm_reference.hpp"
#include "spmm_neon.hpp"
#include "spmm_accelerate.hpp"

struct BenchmarkResult {
    int M, K, N;
    float sparsity;
    int nnz;
    double ref_ms;
    double acc_ms;
    double neon_st_ms;
    double neon_mt_ms;
    double neon_st_gflops;
    double neon_mt_gflops;
    double speedup_over_ref;
    double speedup_over_acc;
};

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
    return times[times.size() / 2]; // Median latency
}

BenchmarkResult run_benchmark_case(int M, int K, int N, float sparsity) {
    CSRMatrix A = generate_random_csr(M, K, sparsity, 2026);
    float* B = allocate_aligned_dense(K, N);
    float* C = allocate_aligned_dense(M, N);

    for (int i = 0; i < K * N; ++i) {
        B[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
    }

    double total_flops = 2.0 * static_cast<double>(A.nnz) * static_cast<double>(N);

    // 1. Naive Scalar Reference (only run if matrix is not too massive)
    double ref_ms = 0.0;
    if (A.M <= 2048) {
        ref_ms = time_kernel([&]() {
            spmm_reference(A, B, C, N);
        }, 1, 3);
    } else {
        ref_ms = -1.0;
    }

    // 2. Apple Accelerate Sparse BLAS
    AccelerateSpMMRunner acc_runner;
    acc_runner.load_csr(A);
    double acc_ms = time_kernel([&]() {
        acc_runner.multiply(B, C, N);
    }, 2, 7);

    // 3. NEON Single-Thread (1 P-Core)
    double neon_st_ms = time_kernel([&]() {
        spmm_neon_single_thread(A, B, C, N);
    }, 3, 10);

    // 4. NEON Multi-Thread (4 P-Cores)
    double neon_mt_ms = time_kernel([&]() {
        spmm_neon_multithread(A, B, C, N, 4);
    }, 3, 10);

    free_aligned(B);
    free_aligned(C);

    BenchmarkResult res;
    res.M = M; res.K = K; res.N = N;
    res.sparsity = sparsity;
    res.nnz = A.nnz;
    res.ref_ms = ref_ms;
    res.acc_ms = acc_ms;
    res.neon_st_ms = neon_st_ms;
    res.neon_mt_ms = neon_mt_ms;

    res.neon_st_gflops = (total_flops / (neon_st_ms * 1e-3)) / 1e9;
    res.neon_mt_gflops = (total_flops / (neon_mt_ms * 1e-3)) / 1e9;
    res.speedup_over_ref = (ref_ms > 0) ? (ref_ms / neon_st_ms) : 0.0;
    res.speedup_over_acc = acc_ms / neon_mt_ms;

    return res;
}

int main() {
    bind_thread_to_pcore();

    std::cout << "======================================================================================================" << std::endl;
    std::cout << "        High-Performance CSR SpMM Benchmark on Apple M4 (NEON vs Apple Accelerate Sparse BLAS)       " << std::endl;
    std::cout << "======================================================================================================" << std::endl;

    std::vector<std::tuple<int, int, int, float>> test_configs = {
        // M,     K,    N,   Sparsity
        {1024, 1024,   64, 0.90f},
        {1024, 1024,  128, 0.95f},
        {2048, 2048,   64, 0.95f},
        {2048, 2048,  128, 0.98f},
        {4096, 4096,   64, 0.98f},
        {4096, 4096,  128, 0.99f},
        {8192, 8192,  128, 0.99f},
        {8192, 8192,  256, 0.995f}
    };

    std::cout << "\n| M x K | N (Dim) | Sparsity | NNZ | Ref (ms) | Accelerate (ms) | NEON 1P (ms) | NEON 4P (ms) | NEON 4P (GFLOPS) | vs Accel Speedup |" << std::endl;
    std::cout << "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |" << std::endl;

    for (const auto& cfg : test_configs) {
        int M = std::get<0>(cfg);
        int K = std::get<1>(cfg);
        int N = std::get<2>(cfg);
        float s = std::get<3>(cfg);

        BenchmarkResult r = run_benchmark_case(M, K, N, s);

        std::cout << "| " << r.M << "x" << r.K 
                  << " | " << std::setw(3) << r.N 
                  << " | " << std::fixed << std::setprecision(1) << (r.sparsity * 100.0f) << "%"
                  << " | " << r.nnz
                  << " | " << (r.ref_ms > 0 ? std::to_string(r.ref_ms).substr(0, 6) : "N/A")
                  << " | " << std::fixed << std::setprecision(2) << r.acc_ms
                  << " | " << std::fixed << std::setprecision(2) << r.neon_st_ms
                  << " | " << std::fixed << std::setprecision(2) << r.neon_mt_ms
                  << " | " << std::fixed << std::setprecision(2) << r.neon_mt_gflops
                  << " | " << std::fixed << std::setprecision(2) << r.speedup_over_acc << "x |"
                  << std::endl;
    }

    std::cout << "======================================================================================================" << std::endl;
    return 0;
}
