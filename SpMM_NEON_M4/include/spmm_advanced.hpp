#pragma once

#include "csr_matrix.hpp"
#include <arm_neon.h>
#include <thread>
#include <vector>
#include <algorithm>

// ==============================================================================
// Advanced NEON SpMM Implementations:
// 1. spmm_neon_u1: Baseline (1x16 tile, U=1 unroll)
// 2. spmm_neon_u2: 1x16 tile with 2-way non-zero unrolling (U=2)
// 3. spmm_neon_u4: 1x16 tile with 4-way non-zero unrolling (U=4)
// 4. spmm_neon_2x16: 2x16 two-row register tile (8 accumulators, alternating RAW bypass)
// ==============================================================================

// ------------------------------------------------------------------------------
// Variant 2: U = 2 Non-Zero Unroll (1x16)
// ------------------------------------------------------------------------------
inline void spmm_neon_u2_single_thread(
    const CSRMatrix& A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    bind_thread_to_pcore();

    const int* __restrict__ row_ptr = A.row_ptr.data();
    const int* __restrict__ col_ind = A.col_indices.data();
    const float* __restrict__ vals  = A.values.data();

    for (int i = 0; i < A.M; ++i) {
        int r_start = row_ptr[i];
        int r_end   = row_ptr[i + 1];
        float* __restrict__ c_row = C + i * N;

        int j = 0;
        for (; j + 15 < N; j += 16) {
            float32x4_t c0 = vdupq_n_f32(0.0f);
            float32x4_t c1 = vdupq_n_f32(0.0f);
            float32x4_t c2 = vdupq_n_f32(0.0f);
            float32x4_t c3 = vdupq_n_f32(0.0f);

            int p = r_start;
            // 2-way unroll
            for (; p + 1 < r_end; p += 2) {
                int k0 = col_ind[p];
                int k1 = col_ind[p + 1];

                float32x4_t va0 = vdupq_n_f32(vals[p]);
                float32x4_t va1 = vdupq_n_f32(vals[p + 1]);

                const float* b_ptr0 = B + k0 * N + j;
                const float* b_ptr1 = B + k1 * N + j;

                float32x4_t b0_0 = vld1q_f32(b_ptr0);
                float32x4_t b0_1 = vld1q_f32(b_ptr0 + 4);
                float32x4_t b0_2 = vld1q_f32(b_ptr0 + 8);
                float32x4_t b0_3 = vld1q_f32(b_ptr0 + 12);

                float32x4_t b1_0 = vld1q_f32(b_ptr1);
                float32x4_t b1_1 = vld1q_f32(b_ptr1 + 4);
                float32x4_t b1_2 = vld1q_f32(b_ptr1 + 8);
                float32x4_t b1_3 = vld1q_f32(b_ptr1 + 12);

                // Interleaved computation
                c0 = vfmaq_f32(c0, b0_0, va0);
                c0 = vfmaq_f32(c0, b1_0, va1);

                c1 = vfmaq_f32(c1, b0_1, va0);
                c1 = vfmaq_f32(c1, b1_1, va1);

                c2 = vfmaq_f32(c2, b0_2, va0);
                c2 = vfmaq_f32(c2, b1_2, va1);

                c3 = vfmaq_f32(c3, b0_3, va0);
                c3 = vfmaq_f32(c3, b1_3, va1);
            }

            // Remainder 1 non-zero
            for (; p < r_end; ++p) {
                int k = col_ind[p];
                float32x4_t va = vdupq_n_f32(vals[p]);
                const float* b_ptr = B + k * N + j;

                c0 = vfmaq_f32(c0, vld1q_f32(b_ptr), va);
                c1 = vfmaq_f32(c1, vld1q_f32(b_ptr + 4), va);
                c2 = vfmaq_f32(c2, vld1q_f32(b_ptr + 8), va);
                c3 = vfmaq_f32(c3, vld1q_f32(b_ptr + 12), va);
            }

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
                float32x4_t va = vdupq_n_f32(vals[p]);
                const float* b_ptr = B + k * N + j;
                c0 = vfmaq_f32(c0, vld1q_f32(b_ptr), va);
                c1 = vfmaq_f32(c1, vld1q_f32(b_ptr + 4), va);
            }

            vst1q_f32(c_row + j,     c0);
            vst1q_f32(c_row + j + 4, c1);
        }

        // Remainder chunk of 4 floats
        for (; j + 3 < N; j += 4) {
            float32x4_t c0 = vdupq_n_f32(0.0f);
            for (int p = r_start; p < r_end; ++p) {
                int k = col_ind[p];
                float32x4_t va = vdupq_n_f32(vals[p]);
                const float* b_ptr = B + k * N + j;
                c0 = vfmaq_f32(c0, vld1q_f32(b_ptr), va);
            }
            vst1q_f32(c_row + j, c0);
        }

        // Scalar remainder
        for (; j < N; ++j) {
            float acc = 0.0f;
            for (int p = r_start; p < r_end; ++p) {
                acc += vals[p] * B[col_ind[p] * N + j];
            }
            c_row[j] = acc;
        }
    }
}

// ------------------------------------------------------------------------------
// Variant 3: U = 4 Non-Zero Unroll (1x16)
// ------------------------------------------------------------------------------
inline void spmm_neon_u4_single_thread(
    const CSRMatrix& A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    bind_thread_to_pcore();

    const int* __restrict__ row_ptr = A.row_ptr.data();
    const int* __restrict__ col_ind = A.col_indices.data();
    const float* __restrict__ vals  = A.values.data();

    for (int i = 0; i < A.M; ++i) {
        int r_start = row_ptr[i];
        int r_end   = row_ptr[i + 1];
        float* __restrict__ c_row = C + i * N;

        int j = 0;
        for (; j + 15 < N; j += 16) {
            float32x4_t c0 = vdupq_n_f32(0.0f);
            float32x4_t c1 = vdupq_n_f32(0.0f);
            float32x4_t c2 = vdupq_n_f32(0.0f);
            float32x4_t c3 = vdupq_n_f32(0.0f);

            int p = r_start;
            // 4-way unroll
            for (; p + 3 < r_end; p += 4) {
                int k0 = col_ind[p];
                int k1 = col_ind[p + 1];
                int k2 = col_ind[p + 2];
                int k3 = col_ind[p + 3];

                float32x4_t va0 = vdupq_n_f32(vals[p]);
                float32x4_t va1 = vdupq_n_f32(vals[p + 1]);
                float32x4_t va2 = vdupq_n_f32(vals[p + 2]);
                float32x4_t va3 = vdupq_n_f32(vals[p + 3]);

                const float* b0 = B + k0 * N + j;
                const float* b1 = B + k1 * N + j;
                const float* b2 = B + k2 * N + j;
                const float* b3 = B + k3 * N + j;

                c0 = vfmaq_f32(c0, vld1q_f32(b0), va0);
                c0 = vfmaq_f32(c0, vld1q_f32(b1), va1);
                c0 = vfmaq_f32(c0, vld1q_f32(b2), va2);
                c0 = vfmaq_f32(c0, vld1q_f32(b3), va3);

                c1 = vfmaq_f32(c1, vld1q_f32(b0 + 4), va0);
                c1 = vfmaq_f32(c1, vld1q_f32(b1 + 4), va1);
                c1 = vfmaq_f32(c1, vld1q_f32(b2 + 4), va2);
                c1 = vfmaq_f32(c1, vld1q_f32(b3 + 4), va3);

                c2 = vfmaq_f32(c2, vld1q_f32(b0 + 8), va0);
                c2 = vfmaq_f32(c2, vld1q_f32(b1 + 8), va1);
                c2 = vfmaq_f32(c2, vld1q_f32(b2 + 8), va2);
                c2 = vfmaq_f32(c2, vld1q_f32(b3 + 8), va3);

                c3 = vfmaq_f32(c3, vld1q_f32(b0 + 12), va0);
                c3 = vfmaq_f32(c3, vld1q_f32(b1 + 12), va1);
                c3 = vfmaq_f32(c3, vld1q_f32(b2 + 12), va2);
                c3 = vfmaq_f32(c3, vld1q_f32(b3 + 12), va3);
            }

            for (; p < r_end; ++p) {
                int k = col_ind[p];
                float32x4_t va = vdupq_n_f32(vals[p]);
                const float* b = B + k * N + j;
                c0 = vfmaq_f32(c0, vld1q_f32(b), va);
                c1 = vfmaq_f32(c1, vld1q_f32(b + 4), va);
                c2 = vfmaq_f32(c2, vld1q_f32(b + 8), va);
                c3 = vfmaq_f32(c3, vld1q_f32(b + 12), va);
            }

            vst1q_f32(c_row + j,      c0);
            vst1q_f32(c_row + j + 4,  c1);
            vst1q_f32(c_row + j + 8,  c2);
            vst1q_f32(c_row + j + 12, c3);
        }

        for (; j + 7 < N; j += 8) {
            float32x4_t c0 = vdupq_n_f32(0.0f);
            float32x4_t c1 = vdupq_n_f32(0.0f);
            for (int p = r_start; p < r_end; ++p) {
                int k = col_ind[p];
                float32x4_t va = vdupq_n_f32(vals[p]);
                const float* b = B + k * N + j;
                c0 = vfmaq_f32(c0, vld1q_f32(b), va);
                c1 = vfmaq_f32(c1, vld1q_f32(b + 4), va);
            }
            vst1q_f32(c_row + j,     c0);
            vst1q_f32(c_row + j + 4, c1);
        }

        for (; j + 3 < N; j += 4) {
            float32x4_t c0 = vdupq_n_f32(0.0f);
            for (int p = r_start; p < r_end; ++p) {
                int k = col_ind[p];
                float32x4_t va = vdupq_n_f32(vals[p]);
                c0 = vfmaq_f32(c0, vld1q_f32(B + k * N + j), va);
            }
            vst1q_f32(c_row + j, c0);
        }

        for (; j < N; ++j) {
            float acc = 0.0f;
            for (int p = r_start; p < r_end; ++p) {
                acc += vals[p] * B[col_ind[p] * N + j];
            }
            c_row[j] = acc;
        }
    }
}

// ------------------------------------------------------------------------------
// Variant 4: 2x16 Two-Row Register Tile (8 Accumulator Registers)
// Alternates accumulation destinations to break RAW pipeline bubbles.
// ------------------------------------------------------------------------------
inline void spmm_neon_2x16_single_thread(
    const CSRMatrix& A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    bind_thread_to_pcore();

    const int* __restrict__ row_ptr = A.row_ptr.data();
    const int* __restrict__ col_ind = A.col_indices.data();
    const float* __restrict__ vals  = A.values.data();

    int i = 0;
    // Process 2 rows concurrently
    for (; i + 1 < A.M; i += 2) {
        int rs0 = row_ptr[i];
        int re0 = row_ptr[i + 1];
        int rs1 = row_ptr[i + 1];
        int re1 = row_ptr[i + 2];

        float* __restrict__ c_row0 = C + i * N;
        float* __restrict__ c_row1 = C + (i + 1) * N;

        int j = 0;
        for (; j + 15 < N; j += 16) {
            // Row 0 accumulators (4 registers)
            float32x4_t c0_0 = vdupq_n_f32(0.0f);
            float32x4_t c0_1 = vdupq_n_f32(0.0f);
            float32x4_t c0_2 = vdupq_n_f32(0.0f);
            float32x4_t c0_3 = vdupq_n_f32(0.0f);

            // Row 1 accumulators (4 registers)
            float32x4_t c1_0 = vdupq_n_f32(0.0f);
            float32x4_t c1_1 = vdupq_n_f32(0.0f);
            float32x4_t c1_2 = vdupq_n_f32(0.0f);
            float32x4_t c1_3 = vdupq_n_f32(0.0f);

            int p0 = rs0;
            int p1 = rs1;

            // Interleave computation of row 0 and row 1
            while (p0 < re0 && p1 < re1) {
                int k0 = col_ind[p0];
                int k1 = col_ind[p1];

                float32x4_t va0 = vdupq_n_f32(vals[p0]);
                float32x4_t va1 = vdupq_n_f32(vals[p1]);

                const float* b0 = B + k0 * N + j;
                const float* b1 = B + k1 * N + j;

                // Load row 0 & row 1 vectors
                float32x4_t v_b0_0 = vld1q_f32(b0);
                float32x4_t v_b1_0 = vld1q_f32(b1);
                c0_0 = vfmaq_f32(c0_0, v_b0_0, va0);
                c1_0 = vfmaq_f32(c1_0, v_b1_0, va1);

                float32x4_t v_b0_1 = vld1q_f32(b0 + 4);
                float32x4_t v_b1_1 = vld1q_f32(b1 + 4);
                c0_1 = vfmaq_f32(c0_1, v_b0_1, va0);
                c1_1 = vfmaq_f32(c1_1, v_b1_1, va1);

                float32x4_t v_b0_2 = vld1q_f32(b0 + 8);
                float32x4_t v_b1_2 = vld1q_f32(b1 + 8);
                c0_2 = vfmaq_f32(c0_2, v_b0_2, va0);
                c1_2 = vfmaq_f32(c1_2, v_b1_2, va1);

                float32x4_t v_b0_3 = vld1q_f32(b0 + 12);
                float32x4_t v_b1_3 = vld1q_f32(b1 + 12);
                c0_3 = vfmaq_f32(c0_3, v_b0_3, va0);
                c1_3 = vfmaq_f32(c1_3, v_b1_3, va1);

                p0++;
                p1++;
            }

            // Remainder for row 0
            while (p0 < re0) {
                int k = col_ind[p0];
                float32x4_t va = vdupq_n_f32(vals[p0]);
                const float* b = B + k * N + j;
                c0_0 = vfmaq_f32(c0_0, vld1q_f32(b), va);
                c0_1 = vfmaq_f32(c0_1, vld1q_f32(b + 4), va);
                c0_2 = vfmaq_f32(c0_2, vld1q_f32(b + 8), va);
                c0_3 = vfmaq_f32(c0_3, vld1q_f32(b + 12), va);
                p0++;
            }

            // Remainder for row 1
            while (p1 < re1) {
                int k = col_ind[p1];
                float32x4_t va = vdupq_n_f32(vals[p1]);
                const float* b = B + k * N + j;
                c1_0 = vfmaq_f32(c1_0, vld1q_f32(b), va);
                c1_1 = vfmaq_f32(c1_1, vld1q_f32(b + 4), va);
                c1_2 = vfmaq_f32(c1_2, vld1q_f32(b + 8), va);
                c1_3 = vfmaq_f32(c1_3, vld1q_f32(b + 12), va);
                p1++;
            }

            // Write back row 0 & row 1
            vst1q_f32(c_row0 + j,      c0_0);
            vst1q_f32(c_row0 + j + 4,  c0_1);
            vst1q_f32(c_row0 + j + 8,  c0_2);
            vst1q_f32(c_row0 + j + 12, c0_3);

            vst1q_f32(c_row1 + j,      c1_0);
            vst1q_f32(c_row1 + j + 4,  c1_1);
            vst1q_f32(c_row1 + j + 8,  c1_2);
            vst1q_f32(c_row1 + j + 12, c1_3);
        }

        // Remainder chunks for row 0 & 1
        for (; j < N; ++j) {
            float acc0 = 0.0f, acc1 = 0.0f;
            for (int p = rs0; p < re0; ++p) acc0 += vals[p] * B[col_ind[p] * N + j];
            for (int p = rs1; p < re1; ++p) acc1 += vals[p] * B[col_ind[p] * N + j];
            c_row0[j] = acc0;
            c_row1[j] = acc1;
        }
    }

    // Remainder row if M is odd
    for (; i < A.M; ++i) {
        int rs = row_ptr[i];
        int re = row_ptr[i + 1];
        float* c_row = C + i * N;
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int p = rs; p < re; ++p) acc += vals[p] * B[col_ind[p] * N + j];
            c_row[j] = acc;
        }
    }
}

// ------------------------------------------------------------------------------
// Generic Multi-Threaded Wrapper for any Single-Threaded Kernel
// ------------------------------------------------------------------------------
template <typename KernelFunc>
inline void spmm_multithread_runner(
    KernelFunc&& kernel,
    const CSRMatrix& A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N,
    int num_threads = 4
) {
    if (num_threads <= 1 || A.M < num_threads * 16) {
        kernel(A, B, C, N);
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

            CSRMatrix sub_A;
            sub_A.M = r_end - r_begin;
            sub_A.K = A.K;
            sub_A.row_ptr.resize(sub_A.M + 1);
            int base_offset = A.row_ptr[r_begin];

            for (int i = 0; i <= sub_A.M; ++i) {
                sub_A.row_ptr[i] = A.row_ptr[r_begin + i] - base_offset;
            }

            int sub_nnz = A.row_ptr[r_end] - base_offset;
            sub_A.nnz = sub_nnz;
            sub_A.col_indices.assign(A.col_indices.begin() + base_offset, A.col_indices.begin() + base_offset + sub_nnz);
            sub_A.values.assign(A.values.begin() + base_offset, A.values.begin() + base_offset + sub_nnz);

            kernel(sub_A, B, C + r_begin * N, N);
        });
    }

    for (auto& w : workers) {
        if (w.joinable()) w.join();
    }
}


// ------------------------------------------------------------------------------
// Variant 5: 2x8 Two-Row Tile (4 Accumulator Registers)
// ------------------------------------------------------------------------------
inline void spmm_neon_2x8_single_thread(
    const CSRMatrix& A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    bind_thread_to_pcore();
    const int* __restrict__ row_ptr = A.row_ptr.data();
    const int* __restrict__ col_ind = A.col_indices.data();
    const float* __restrict__ vals  = A.values.data();

    int i = 0;
    for (; i + 1 < A.M; i += 2) {
        int rs0 = row_ptr[i], re0 = row_ptr[i + 1];
        int rs1 = row_ptr[i + 1], re1 = row_ptr[i + 2];
        float* __restrict__ c_row0 = C + i * N;
        float* __restrict__ c_row1 = C + (i + 1) * N;

        int j = 0;
        for (; j + 7 < N; j += 8) {
            float32x4_t c0_0 = vdupq_n_f32(0.0f), c0_1 = vdupq_n_f32(0.0f);
            float32x4_t c1_0 = vdupq_n_f32(0.0f), c1_1 = vdupq_n_f32(0.0f);

            int p0 = rs0, p1 = rs1;
            while (p0 < re0 && p1 < re1) {
                int k0 = col_ind[p0], k1 = col_ind[p1];
                float32x4_t va0 = vdupq_n_f32(vals[p0]);
                float32x4_t va1 = vdupq_n_f32(vals[p1]);
                const float* b0 = B + k0 * N + j;
                const float* b1 = B + k1 * N + j;

                c0_0 = vfmaq_f32(c0_0, vld1q_f32(b0), va0);
                c1_0 = vfmaq_f32(c1_0, vld1q_f32(b1), va1);
                c0_1 = vfmaq_f32(c0_1, vld1q_f32(b0 + 4), va0);
                c1_1 = vfmaq_f32(c1_1, vld1q_f32(b1 + 4), va1);
                p0++; p1++;
            }
            while (p0 < re0) {
                float32x4_t va = vdupq_n_f32(vals[p0]);
                const float* b = B + col_ind[p0] * N + j;
                c0_0 = vfmaq_f32(c0_0, vld1q_f32(b), va);
                c0_1 = vfmaq_f32(c0_1, vld1q_f32(b + 4), va);
                p0++;
            }
            while (p1 < re1) {
                float32x4_t va = vdupq_n_f32(vals[p1]);
                const float* b = B + col_ind[p1] * N + j;
                c1_0 = vfmaq_f32(c1_0, vld1q_f32(b), va);
                c1_1 = vfmaq_f32(c1_1, vld1q_f32(b + 4), va);
                p1++;
            }
            vst1q_f32(c_row0 + j,     c0_0);
            vst1q_f32(c_row0 + j + 4, c0_1);
            vst1q_f32(c_row1 + j,     c1_0);
            vst1q_f32(c_row1 + j + 4, c1_1);
        }

        for (; j < N; ++j) {
            float a0 = 0.0f, a1 = 0.0f;
            for (int p = rs0; p < re0; ++p) a0 += vals[p] * B[col_ind[p] * N + j];
            for (int p = rs1; p < re1; ++p) a1 += vals[p] * B[col_ind[p] * N + j];
            c_row0[j] = a0;
            c_row1[j] = a1;
        }
    }
    for (; i < A.M; ++i) {
        int rs = row_ptr[i], re = row_ptr[i + 1];
        float* c_row = C + i * N;
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int p = rs; p < re; ++p) acc += vals[p] * B[col_ind[p] * N + j];
            c_row[j] = acc;
        }
    }
}

// ------------------------------------------------------------------------------
// Variant 6: 4x8 Four-Row Tile (8 Accumulator Registers)
// ------------------------------------------------------------------------------
inline void spmm_neon_4x8_single_thread(
    const CSRMatrix& A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    bind_thread_to_pcore();
    const int* __restrict__ row_ptr = A.row_ptr.data();
    const int* __restrict__ col_ind = A.col_indices.data();
    const float* __restrict__ vals  = A.values.data();

    int i = 0;
    for (; i + 3 < A.M; i += 4) {
        int rs0 = row_ptr[i], re0 = row_ptr[i + 1];
        int rs1 = row_ptr[i + 1], re1 = row_ptr[i + 2];
        int rs2 = row_ptr[i + 2], re2 = row_ptr[i + 3];
        int rs3 = row_ptr[i + 3], re3 = row_ptr[i + 4];

        float* __restrict__ c0 = C + i * N;
        float* __restrict__ c1 = C + (i + 1) * N;
        float* __restrict__ c2 = C + (i + 2) * N;
        float* __restrict__ c3 = C + (i + 3) * N;

        int j = 0;
        for (; j + 7 < N; j += 8) {
            float32x4_t r0_0 = vdupq_n_f32(0.0f), r0_1 = vdupq_n_f32(0.0f);
            float32x4_t r1_0 = vdupq_n_f32(0.0f), r1_1 = vdupq_n_f32(0.0f);
            float32x4_t r2_0 = vdupq_n_f32(0.0f), r2_1 = vdupq_n_f32(0.0f);
            float32x4_t r3_0 = vdupq_n_f32(0.0f), r3_1 = vdupq_n_f32(0.0f);

            int p0 = rs0, p1 = rs1, p2 = rs2, p3 = rs3;
            while (p0 < re0 && p1 < re1 && p2 < re2 && p3 < re3) {
                float32x4_t va0 = vdupq_n_f32(vals[p0]);
                float32x4_t va1 = vdupq_n_f32(vals[p1]);
                float32x4_t va2 = vdupq_n_f32(vals[p2]);
                float32x4_t va3 = vdupq_n_f32(vals[p3]);

                const float* b0 = B + col_ind[p0] * N + j;
                const float* b1 = B + col_ind[p1] * N + j;
                const float* b2 = B + col_ind[p2] * N + j;
                const float* b3 = B + col_ind[p3] * N + j;

                r0_0 = vfmaq_f32(r0_0, vld1q_f32(b0), va0);
                r1_0 = vfmaq_f32(r1_0, vld1q_f32(b1), va1);
                r2_0 = vfmaq_f32(r2_0, vld1q_f32(b2), va2);
                r3_0 = vfmaq_f32(r3_0, vld1q_f32(b3), va3);

                r0_1 = vfmaq_f32(r0_1, vld1q_f32(b0 + 4), va0);
                r1_1 = vfmaq_f32(r1_1, vld1q_f32(b1 + 4), va1);
                r2_1 = vfmaq_f32(r2_1, vld1q_f32(b2 + 4), va2);
                r3_1 = vfmaq_f32(r3_1, vld1q_f32(b3 + 4), va3);

                p0++; p1++; p2++; p3++;
            }
            while (p0 < re0) {
                float32x4_t va = vdupq_n_f32(vals[p0]);
                const float* b = B + col_ind[p0] * N + j;
                r0_0 = vfmaq_f32(r0_0, vld1q_f32(b), va);
                r0_1 = vfmaq_f32(r0_1, vld1q_f32(b + 4), va);
                p0++;
            }
            while (p1 < re1) {
                float32x4_t va = vdupq_n_f32(vals[p1]);
                const float* b = B + col_ind[p1] * N + j;
                r1_0 = vfmaq_f32(r1_0, vld1q_f32(b), va);
                r1_1 = vfmaq_f32(r1_1, vld1q_f32(b + 4), va);
                p1++;
            }
            while (p2 < re2) {
                float32x4_t va = vdupq_n_f32(vals[p2]);
                const float* b = B + col_ind[p2] * N + j;
                r2_0 = vfmaq_f32(r2_0, vld1q_f32(b), va);
                r2_1 = vfmaq_f32(r2_1, vld1q_f32(b + 4), va);
                p2++;
            }
            while (p3 < re3) {
                float32x4_t va = vdupq_n_f32(vals[p3]);
                const float* b = B + col_ind[p3] * N + j;
                r3_0 = vfmaq_f32(r3_0, vld1q_f32(b), va);
                r3_1 = vfmaq_f32(r3_1, vld1q_f32(b + 4), va);
                p3++;
            }

            vst1q_f32(c0 + j,     r0_0); vst1q_f32(c0 + j + 4, r0_1);
            vst1q_f32(c1 + j,     r1_0); vst1q_f32(c1 + j + 4, r1_1);
            vst1q_f32(c2 + j,     r2_0); vst1q_f32(c2 + j + 4, r2_1);
            vst1q_f32(c3 + j,     r3_0); vst1q_f32(c3 + j + 4, r3_1);
        }

        for (; j < N; ++j) {
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            for (int p = rs0; p < re0; ++p) a0 += vals[p] * B[col_ind[p] * N + j];
            for (int p = rs1; p < re1; ++p) a1 += vals[p] * B[col_ind[p] * N + j];
            for (int p = rs2; p < re2; ++p) a2 += vals[p] * B[col_ind[p] * N + j];
            for (int p = rs3; p < re3; ++p) a3 += vals[p] * B[col_ind[p] * N + j];
            c0[j] = a0; c1[j] = a1; c2[j] = a2; c3[j] = a3;
        }
    }
    for (; i < A.M; ++i) {
        int rs = row_ptr[i], re = row_ptr[i + 1];
        float* cr = C + i * N;
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int p = rs; p < re; ++p) acc += vals[p] * B[col_ind[p] * N + j];
            cr[j] = acc;
        }
    }
}

// ------------------------------------------------------------------------------
// Variant 7: 4x16 Four-Row Tile (16 Accumulator Registers, saturating 32 registers)
// ------------------------------------------------------------------------------
inline void spmm_neon_4x16_single_thread(
    const CSRMatrix& A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    bind_thread_to_pcore();
    const int* __restrict__ row_ptr = A.row_ptr.data();
    const int* __restrict__ col_ind = A.col_indices.data();
    const float* __restrict__ vals  = A.values.data();

    int i = 0;
    for (; i + 3 < A.M; i += 4) {
        int rs0 = row_ptr[i], re0 = row_ptr[i + 1];
        int rs1 = row_ptr[i + 1], re1 = row_ptr[i + 2];
        int rs2 = row_ptr[i + 2], re2 = row_ptr[i + 3];
        int rs3 = row_ptr[i + 3], re3 = row_ptr[i + 4];

        float* __restrict__ c0 = C + i * N;
        float* __restrict__ c1 = C + (i + 1) * N;
        float* __restrict__ c2 = C + (i + 2) * N;
        float* __restrict__ c3 = C + (i + 3) * N;

        int j = 0;
        for (; j + 15 < N; j += 16) {
            float32x4_t r0_0 = vdupq_n_f32(0.0f), r0_1 = vdupq_n_f32(0.0f), r0_2 = vdupq_n_f32(0.0f), r0_3 = vdupq_n_f32(0.0f);
            float32x4_t r1_0 = vdupq_n_f32(0.0f), r1_1 = vdupq_n_f32(0.0f), r1_2 = vdupq_n_f32(0.0f), r1_3 = vdupq_n_f32(0.0f);
            float32x4_t r2_0 = vdupq_n_f32(0.0f), r2_1 = vdupq_n_f32(0.0f), r2_2 = vdupq_n_f32(0.0f), r2_3 = vdupq_n_f32(0.0f);
            float32x4_t r3_0 = vdupq_n_f32(0.0f), r3_1 = vdupq_n_f32(0.0f), r3_2 = vdupq_n_f32(0.0f), r3_3 = vdupq_n_f32(0.0f);

            int p0 = rs0, p1 = rs1, p2 = rs2, p3 = rs3;
            while (p0 < re0 && p1 < re1 && p2 < re2 && p3 < re3) {
                float32x4_t va0 = vdupq_n_f32(vals[p0]);
                float32x4_t va1 = vdupq_n_f32(vals[p1]);
                float32x4_t va2 = vdupq_n_f32(vals[p2]);
                float32x4_t va3 = vdupq_n_f32(vals[p3]);

                const float* b0 = B + col_ind[p0] * N + j;
                const float* b1 = B + col_ind[p1] * N + j;
                const float* b2 = B + col_ind[p2] * N + j;
                const float* b3 = B + col_ind[p3] * N + j;

                r0_0 = vfmaq_f32(r0_0, vld1q_f32(b0), va0);
                r1_0 = vfmaq_f32(r1_0, vld1q_f32(b1), va1);
                r2_0 = vfmaq_f32(r2_0, vld1q_f32(b2), va2);
                r3_0 = vfmaq_f32(r3_0, vld1q_f32(b3), va3);

                r0_1 = vfmaq_f32(r0_1, vld1q_f32(b0 + 4), va0);
                r1_1 = vfmaq_f32(r1_1, vld1q_f32(b1 + 4), va1);
                r2_1 = vfmaq_f32(r2_1, vld1q_f32(b2 + 4), va2);
                r3_1 = vfmaq_f32(r3_1, vld1q_f32(b3 + 4), va3);

                r0_2 = vfmaq_f32(r0_2, vld1q_f32(b0 + 8), va0);
                r1_2 = vfmaq_f32(r1_2, vld1q_f32(b1 + 8), va1);
                r2_2 = vfmaq_f32(r2_2, vld1q_f32(b2 + 8), va2);
                r3_2 = vfmaq_f32(r3_2, vld1q_f32(b3 + 8), va3);

                r0_3 = vfmaq_f32(r0_3, vld1q_f32(b0 + 12), va0);
                r1_3 = vfmaq_f32(r1_3, vld1q_f32(b1 + 12), va1);
                r2_3 = vfmaq_f32(r2_3, vld1q_f32(b2 + 12), va2);
                r3_3 = vfmaq_f32(r3_3, vld1q_f32(b3 + 12), va3);

                p0++; p1++; p2++; p3++;
            }
            while (p0 < re0) {
                float32x4_t va = vdupq_n_f32(vals[p0]);
                const float* b = B + col_ind[p0] * N + j;
                r0_0 = vfmaq_f32(r0_0, vld1q_f32(b), va);
                r0_1 = vfmaq_f32(r0_1, vld1q_f32(b + 4), va);
                r0_2 = vfmaq_f32(r0_2, vld1q_f32(b + 8), va);
                r0_3 = vfmaq_f32(r0_3, vld1q_f32(b + 12), va);
                p0++;
            }
            while (p1 < re1) {
                float32x4_t va = vdupq_n_f32(vals[p1]);
                const float* b = B + col_ind[p1] * N + j;
                r1_0 = vfmaq_f32(r1_0, vld1q_f32(b), va);
                r1_1 = vfmaq_f32(r1_1, vld1q_f32(b + 4), va);
                r1_2 = vfmaq_f32(r1_2, vld1q_f32(b + 8), va);
                r1_3 = vfmaq_f32(r1_3, vld1q_f32(b + 12), va);
                p1++;
            }
            while (p2 < re2) {
                float32x4_t va = vdupq_n_f32(vals[p2]);
                const float* b = B + col_ind[p2] * N + j;
                r2_0 = vfmaq_f32(r2_0, vld1q_f32(b), va);
                r2_1 = vfmaq_f32(r2_1, vld1q_f32(b + 4), va);
                r2_2 = vfmaq_f32(r2_2, vld1q_f32(b + 8), va);
                r2_3 = vfmaq_f32(r2_3, vld1q_f32(b + 12), va);
                p2++;
            }
            while (p3 < re3) {
                float32x4_t va = vdupq_n_f32(vals[p3]);
                const float* b = B + col_ind[p3] * N + j;
                r3_0 = vfmaq_f32(r3_0, vld1q_f32(b), va);
                r3_1 = vfmaq_f32(r3_1, vld1q_f32(b + 4), va);
                r3_2 = vfmaq_f32(r3_2, vld1q_f32(b + 8), va);
                r3_3 = vfmaq_f32(r3_3, vld1q_f32(b + 12), va);
                p3++;
            }

            vst1q_f32(c0 + j,      r0_0); vst1q_f32(c0 + j + 4,  r0_1);
            vst1q_f32(c0 + j + 8,  r0_2); vst1q_f32(c0 + j + 12, r0_3);

            vst1q_f32(c1 + j,      r1_0); vst1q_f32(c1 + j + 4,  r1_1);
            vst1q_f32(c1 + j + 8,  r1_2); vst1q_f32(c1 + j + 12, r1_3);

            vst1q_f32(c2 + j,      r2_0); vst1q_f32(c2 + j + 4,  r2_1);
            vst1q_f32(c2 + j + 8,  r2_2); vst1q_f32(c2 + j + 12, r2_3);

            vst1q_f32(c3 + j,      r3_0); vst1q_f32(c3 + j + 4,  r3_1);
            vst1q_f32(c3 + j + 8,  r3_2); vst1q_f32(c3 + j + 12, r3_3);
        }

        for (; j < N; ++j) {
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            for (int p = rs0; p < re0; ++p) a0 += vals[p] * B[col_ind[p] * N + j];
            for (int p = rs1; p < re1; ++p) a1 += vals[p] * B[col_ind[p] * N + j];
            for (int p = rs2; p < re2; ++p) a2 += vals[p] * B[col_ind[p] * N + j];
            for (int p = rs3; p < re3; ++p) a3 += vals[p] * B[col_ind[p] * N + j];
            c0[j] = a0; c1[j] = a1; c2[j] = a2; c3[j] = a3;
        }
    }
    for (; i < A.M; ++i) {
        int rs = row_ptr[i], re = row_ptr[i + 1];
        float* cr = C + i * N;
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int p = rs; p < re; ++p) acc += vals[p] * B[col_ind[p] * N + j];
            cr[j] = acc;
        }
    }
}


// ------------------------------------------------------------------------------
// Variant 8: U = 6 Non-Zero Unroll (1x16)
// ------------------------------------------------------------------------------
inline void spmm_neon_u6_single_thread(
    const CSRMatrix& A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    bind_thread_to_pcore();
    const int* __restrict__ row_ptr = A.row_ptr.data();
    const int* __restrict__ col_ind = A.col_indices.data();
    const float* __restrict__ vals  = A.values.data();

    for (int i = 0; i < A.M; ++i) {
        int r_start = row_ptr[i];
        int r_end   = row_ptr[i + 1];
        float* __restrict__ c_row = C + i * N;

        int j = 0;
        for (; j + 15 < N; j += 16) {
            float32x4_t c0 = vdupq_n_f32(0.0f);
            float32x4_t c1 = vdupq_n_f32(0.0f);
            float32x4_t c2 = vdupq_n_f32(0.0f);
            float32x4_t c3 = vdupq_n_f32(0.0f);

            int p = r_start;
            // 6-way unroll
            for (; p + 5 < r_end; p += 6) {
                int k0 = col_ind[p];
                int k1 = col_ind[p + 1];
                int k2 = col_ind[p + 2];
                int k3 = col_ind[p + 3];
                int k4 = col_ind[p + 4];
                int k5 = col_ind[p + 5];

                float32x4_t va0 = vdupq_n_f32(vals[p]);
                float32x4_t va1 = vdupq_n_f32(vals[p + 1]);
                float32x4_t va2 = vdupq_n_f32(vals[p + 2]);
                float32x4_t va3 = vdupq_n_f32(vals[p + 3]);
                float32x4_t va4 = vdupq_n_f32(vals[p + 4]);
                float32x4_t va5 = vdupq_n_f32(vals[p + 5]);

                const float* b0 = B + k0 * N + j;
                const float* b1 = B + k1 * N + j;
                const float* b2 = B + k2 * N + j;
                const float* b3 = B + k3 * N + j;
                const float* b4 = B + k4 * N + j;
                const float* b5 = B + k5 * N + j;

                c0 = vfmaq_f32(c0, vld1q_f32(b0), va0);
                c0 = vfmaq_f32(c0, vld1q_f32(b1), va1);
                c0 = vfmaq_f32(c0, vld1q_f32(b2), va2);
                c0 = vfmaq_f32(c0, vld1q_f32(b3), va3);
                c0 = vfmaq_f32(c0, vld1q_f32(b4), va4);
                c0 = vfmaq_f32(c0, vld1q_f32(b5), va5);

                c1 = vfmaq_f32(c1, vld1q_f32(b0 + 4), va0);
                c1 = vfmaq_f32(c1, vld1q_f32(b1 + 4), va1);
                c1 = vfmaq_f32(c1, vld1q_f32(b2 + 4), va2);
                c1 = vfmaq_f32(c1, vld1q_f32(b3 + 4), va3);
                c1 = vfmaq_f32(c1, vld1q_f32(b4 + 4), va4);
                c1 = vfmaq_f32(c1, vld1q_f32(b5 + 4), va5);

                c2 = vfmaq_f32(c2, vld1q_f32(b0 + 8), va0);
                c2 = vfmaq_f32(c2, vld1q_f32(b1 + 8), va1);
                c2 = vfmaq_f32(c2, vld1q_f32(b2 + 8), va2);
                c2 = vfmaq_f32(c2, vld1q_f32(b3 + 8), va3);
                c2 = vfmaq_f32(c2, vld1q_f32(b4 + 8), va4);
                c2 = vfmaq_f32(c2, vld1q_f32(b5 + 8), va5);

                c3 = vfmaq_f32(c3, vld1q_f32(b0 + 12), va0);
                c3 = vfmaq_f32(c3, vld1q_f32(b1 + 12), va1);
                c3 = vfmaq_f32(c3, vld1q_f32(b2 + 12), va2);
                c3 = vfmaq_f32(c3, vld1q_f32(b3 + 12), va3);
                c3 = vfmaq_f32(c3, vld1q_f32(b4 + 12), va4);
                c3 = vfmaq_f32(c3, vld1q_f32(b5 + 12), va5);
            }

            for (; p < r_end; ++p) {
                int k = col_ind[p];
                float32x4_t va = vdupq_n_f32(vals[p]);
                const float* b = B + k * N + j;
                c0 = vfmaq_f32(c0, vld1q_f32(b), va);
                c1 = vfmaq_f32(c1, vld1q_f32(b + 4), va);
                c2 = vfmaq_f32(c2, vld1q_f32(b + 8), va);
                c3 = vfmaq_f32(c3, vld1q_f32(b + 12), va);
            }

            vst1q_f32(c_row + j,      c0);
            vst1q_f32(c_row + j + 4,  c1);
            vst1q_f32(c_row + j + 8,  c2);
            vst1q_f32(c_row + j + 12, c3);
        }

        for (; j + 7 < N; j += 8) {
            float32x4_t c0 = vdupq_n_f32(0.0f);
            float32x4_t c1 = vdupq_n_f32(0.0f);
            for (int p = r_start; p < r_end; ++p) {
                int k = col_ind[p];
                float32x4_t va = vdupq_n_f32(vals[p]);
                const float* b = B + k * N + j;
                c0 = vfmaq_f32(c0, vld1q_f32(b), va);
                c1 = vfmaq_f32(c1, vld1q_f32(b + 4), va);
            }
            vst1q_f32(c_row + j,     c0);
            vst1q_f32(c_row + j + 4, c1);
        }

        for (; j + 3 < N; j += 4) {
            float32x4_t c0 = vdupq_n_f32(0.0f);
            for (int p = r_start; p < r_end; ++p) {
                int k = col_ind[p];
                float32x4_t va = vdupq_n_f32(vals[p]);
                c0 = vfmaq_f32(c0, vld1q_f32(B + k * N + j), va);
            }
            vst1q_f32(c_row + j, c0);
        }

        for (; j < N; ++j) {
            float acc = 0.0f;
            for (int p = r_start; p < r_end; ++p) {
                acc += vals[p] * B[col_ind[p] * N + j];
            }
            c_row[j] = acc;
        }
    }
}

// ------------------------------------------------------------------------------
// Variant 9: 8x16 Eight-Row Tile (32 Accumulator Registers)
// ------------------------------------------------------------------------------
inline void spmm_neon_8x16_single_thread(
    const CSRMatrix& A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    bind_thread_to_pcore();
    const int* __restrict__ row_ptr = A.row_ptr.data();
    const int* __restrict__ col_ind = A.col_indices.data();
    const float* __restrict__ vals  = A.values.data();

    int i = 0;
    for (; i + 7 < A.M; i += 8) {
        int rs[8], re[8];
        float* cr[8];
        for (int r = 0; r < 8; ++r) {
            rs[r] = row_ptr[i + r];
            re[r] = row_ptr[i + r + 1];
            cr[r] = C + (i + r) * N;
        }

        int j = 0;
        for (; j + 15 < N; j += 16) {
            // 32 accumulator registers (8 rows x 4 vectors)
            float32x4_t acc[8][4];
            for (int r = 0; r < 8; ++r) {
                acc[r][0] = vdupq_n_f32(0.0f);
                acc[r][1] = vdupq_n_f32(0.0f);
                acc[r][2] = vdupq_n_f32(0.0f);
                acc[r][3] = vdupq_n_f32(0.0f);
            }

            int p[8];
            for (int r = 0; r < 8; ++r) p[r] = rs[r];

            // Interleaved computation across all 8 rows
            bool any_active = true;
            while (any_active) {
                any_active = false;
                for (int r = 0; r < 8; ++r) {
                    if (p[r] < re[r]) {
                        any_active = true;
                        int k = col_ind[p[r]];
                        float32x4_t va = vdupq_n_f32(vals[p[r]]);
                        const float* b = B + k * N + j;
                        acc[r][0] = vfmaq_f32(acc[r][0], vld1q_f32(b), va);
                        acc[r][1] = vfmaq_f32(acc[r][1], vld1q_f32(b + 4), va);
                        acc[r][2] = vfmaq_f32(acc[r][2], vld1q_f32(b + 8), va);
                        acc[r][3] = vfmaq_f32(acc[r][3], vld1q_f32(b + 12), va);
                        p[r]++;
                    }
                }
            }

            for (int r = 0; r < 8; ++r) {
                vst1q_f32(cr[r] + j,      acc[r][0]);
                vst1q_f32(cr[r] + j + 4,  acc[r][1]);
                vst1q_f32(cr[r] + j + 8,  acc[r][2]);
                vst1q_f32(cr[r] + j + 12, acc[r][3]);
            }
        }

        for (; j < N; ++j) {
            for (int r = 0; r < 8; ++r) {
                float a = 0.0f;
                for (int p = rs[r]; p < re[r]; ++p) a += vals[p] * B[col_ind[p] * N + j];
                cr[r][j] = a;
            }
        }
    }

    // Remainder rows
    for (; i < A.M; ++i) {
        int rs = row_ptr[i], re = row_ptr[i + 1];
        float* cr = C + i * N;
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int p = rs; p < re; ++p) acc += vals[p] * B[col_ind[p] * N + j];
            cr[j] = acc;
        }
    }
}
