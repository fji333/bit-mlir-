#pragma once

#include "csr_matrix.hpp"
#include <arm_neon.h>
#include <thread>
#include <vector>
#include <algorithm>

// NEON vector implementation for SpMM (C = A * B)
// Vectorizes along the dense dimension N and keeps accumulators in registers during row reduction.

// Core Single-Threaded NEON Micro-Kernel
inline void spmm_neon_single_thread(
    const CSRMatrix& A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    bind_thread_to_pcore();

    const int* __restrict__ row_ptr = A.row_ptr.data();
    const int* __restrict__ col_ind = A.col_indices.data();
    const float* __restrict__ vals  = A.values.data();

    // Process row by row
    for (int i = 0; i < A.M; ++i) {
        int r_start = row_ptr[i];
        int r_end   = row_ptr[i + 1];

        float* __restrict__ c_row = C + i * N;

        // Tile N in chunks of 16 floats (4 NEON registers)
        int j = 0;
        for (; j + 15 < N; j += 16) {
            float32x4_t c0 = vdupq_n_f32(0.0f);
            float32x4_t c1 = vdupq_n_f32(0.0f);
            float32x4_t c2 = vdupq_n_f32(0.0f);
            float32x4_t c3 = vdupq_n_f32(0.0f);

            // Accumulate across all non-zeros for these 16 columns in registers
            for (int p = r_start; p < r_end; ++p) {
                int k = col_ind[p];
                float32x4_t v_a = vdupq_n_f32(vals[p]);
                const float* b_ptr = B + k * N + j;

                float32x4_t b0 = vld1q_f32(b_ptr);
                float32x4_t b1 = vld1q_f32(b_ptr + 4);
                float32x4_t b2 = vld1q_f32(b_ptr + 8);
                float32x4_t b3 = vld1q_f32(b_ptr + 12);

                c0 = vfmaq_f32(c0, b0, v_a);
                c1 = vfmaq_f32(c1, b1, v_a);
                c2 = vfmaq_f32(c2, b2, v_a);
                c3 = vfmaq_f32(c3, b3, v_a);
            }

            // Single write back to DRAM / Cache
            vst1q_f32(c_row + j,      c0);
            vst1q_f32(c_row + j + 4,  c1);
            vst1q_f32(c_row + j + 8,  c2);
            vst1q_f32(c_row + j + 12, c3);
        }

        // Remainder chunk of 8 floats
        for (; j + 7 < N; j += 8) {
            float32x4_t c0 = vdupq_n_f32(0.0f);
            float32x4_t c1 = vdupq_n_f32(0.0f);

            for (int p = r_start; p < r_end; ++p) {
                int k = col_ind[p];
                float32x4_t v_a = vdupq_n_f32(vals[p]);
                const float* b_ptr = B + k * N + j;

                float32x4_t b0 = vld1q_f32(b_ptr);
                float32x4_t b1 = vld1q_f32(b_ptr + 4);

                c0 = vfmaq_f32(c0, b0, v_a);
                c1 = vfmaq_f32(c1, b1, v_a);
            }

            vst1q_f32(c_row + j,     c0);
            vst1q_f32(c_row + j + 4, c1);
        }

        // Remainder chunk of 4 floats
        for (; j + 3 < N; j += 4) {
            float32x4_t c0 = vdupq_n_f32(0.0f);

            for (int p = r_start; p < r_end; ++p) {
                int k = col_ind[p];
                float32x4_t v_a = vdupq_n_f32(vals[p]);
                const float* b_ptr = B + k * N + j;
                float32x4_t b0 = vld1q_f32(b_ptr);
                c0 = vfmaq_f32(c0, b0, v_a);
            }

            vst1q_f32(c_row + j, c0);
        }

        // Scalar remainder for j < N
        for (; j < N; ++j) {
            float acc = 0.0f;
            for (int p = r_start; p < r_end; ++p) {
                int k = col_ind[p];
                acc += vals[p] * B[k * N + j];
            }
            c_row[j] = acc;
        }
    }
}

// Multi-Threaded NEON SpMM Engine across P-Cores
inline void spmm_neon_multithread(
    const CSRMatrix& A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N,
    int num_threads = 4
) {
    if (num_threads <= 1 || A.M < num_threads * 16) {
        spmm_neon_single_thread(A, B, C, N);
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    int rows_per_thread = (A.M + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        int r_begin = t * rows_per_thread;
        int r_end = std::min(r_begin + rows_per_thread, A.M);
        if (r_begin >= r_end) break;

        workers.emplace_back([&, r_begin, r_end]() {
            bind_thread_to_pcore();

            const int* __restrict__ row_ptr = A.row_ptr.data();
            const int* __restrict__ col_ind = A.col_indices.data();
            const float* __restrict__ vals  = A.values.data();

            for (int i = r_begin; i < r_end; ++i) {
                int rs = row_ptr[i];
                int re = row_ptr[i + 1];
                float* __restrict__ c_row = C + i * N;

                int j = 0;
                for (; j + 15 < N; j += 16) {
                    float32x4_t c0 = vdupq_n_f32(0.0f);
                    float32x4_t c1 = vdupq_n_f32(0.0f);
                    float32x4_t c2 = vdupq_n_f32(0.0f);
                    float32x4_t c3 = vdupq_n_f32(0.0f);

                    for (int p = rs; p < re; ++p) {
                        int k = col_ind[p];
                        float32x4_t v_a = vdupq_n_f32(vals[p]);
                        const float* b_ptr = B + k * N + j;

                        float32x4_t b0 = vld1q_f32(b_ptr);
                        float32x4_t b1 = vld1q_f32(b_ptr + 4);
                        float32x4_t b2 = vld1q_f32(b_ptr + 8);
                        float32x4_t b3 = vld1q_f32(b_ptr + 12);

                        c0 = vfmaq_f32(c0, b0, v_a);
                        c1 = vfmaq_f32(c1, b1, v_a);
                        c2 = vfmaq_f32(c2, b2, v_a);
                        c3 = vfmaq_f32(c3, b3, v_a);
                    }

                    vst1q_f32(c_row + j,      c0);
                    vst1q_f32(c_row + j + 4,  c1);
                    vst1q_f32(c_row + j + 8,  c2);
                    vst1q_f32(c_row + j + 12, c3);
                }

                for (; j + 7 < N; j += 8) {
                    float32x4_t c0 = vdupq_n_f32(0.0f);
                    float32x4_t c1 = vdupq_n_f32(0.0f);

                    for (int p = rs; p < re; ++p) {
                        int k = col_ind[p];
                        float32x4_t v_a = vdupq_n_f32(vals[p]);
                        const float* b_ptr = B + k * N + j;

                        float32x4_t b0 = vld1q_f32(b_ptr);
                        float32x4_t b1 = vld1q_f32(b_ptr + 4);

                        c0 = vfmaq_f32(c0, b0, v_a);
                        c1 = vfmaq_f32(c1, b1, v_a);
                    }

                    vst1q_f32(c_row + j,     c0);
                    vst1q_f32(c_row + j + 4, c1);
                }

                for (; j + 3 < N; j += 4) {
                    float32x4_t c0 = vdupq_n_f32(0.0f);
                    for (int p = rs; p < re; ++p) {
                        int k = col_ind[p];
                        float32x4_t v_a = vdupq_n_f32(vals[p]);
                        const float* b_ptr = B + k * N + j;
                        float32x4_t b0 = vld1q_f32(b_ptr);
                        c0 = vfmaq_f32(c0, b0, v_a);
                    }
                    vst1q_f32(c_row + j, c0);
                }

                for (; j < N; ++j) {
                    float acc = 0.0f;
                    for (int p = rs; p < re; ++p) {
                        int k = col_ind[p];
                        acc += vals[p] * B[k * N + j];
                    }
                    c_row[j] = acc;
                }
            }
        });
    }

    for (auto& w : workers) {
        if (w.joinable()) {
            w.join();
        }
    }
}
