# Experiment Report: SpMM on Apple M4 with NEON

**Author:** Donglin Shao
**Platform:** Apple M4 CPU, macOS Sequoia
**Target:** Multiply a CSR sparse matrix by a dense matrix ($C = A \times B$) and compare with Apple Accelerate.

---

## 1. Overview

SpMM calculates:

$$
C = A \times B
$$

Where matrix $A$ is sparse ($M \times K$, CSR format), and matrices $B$ ($K \times N$) and $C$ ($M \times N$) are dense.

We implemented this operation using C++ and ARM NEON intrinsics on an Apple M4 CPU, and compared performance with Apple's official Accelerate framework (`sparse_matrix_product_dense_float`).

---

## 2. Implementation

1. **CSR Representation**:Matrix $A$ is stored using three standard arrays: `values` (non-zero floats), `col_indices` (column index for each non-zero), and `row_ptr` (row offsets).
2. **Vectorization**:Because the sparse dimension has irregular non-zero positions, we vectorize along the dense dimension ($N$). For each non-zero element $A[i, k]$, we broadcast it with `vdupq_n_f32`, load 4 consecutive floats from row $k$ of matrix $B$ using `vld1q_f32`, and multiply-add with `vfmaq_f32`.
3. **Register Accumulation**:In a basic loop, values in $C$ are read and written for each non-zero. Here, we process 16 columns at a time and hold the running sum in 4 NEON registers (`c0`, `c1`, `c2`, `c3`). We write back to matrix $C$ only once after all non-zeros in the row are processed.
4. **Multi-Threading**:
   Rows are divided among 4 threads using `std::thread`, with each thread pinned to a high-performance core (P-Core).

---

## 3. Correctness

We verified output against a simple scalar loop and Apple Accelerate across multiple matrix sizes ($64 \times 64$ to $1024 \times 1024$, sparsity from 90% to 99.5%).

* Max difference: `0.0000`
* RMSE: `0.0000`

All results matched with zero difference.

---

## 4. Benchmark Results (Multi-Core vs Apple Accelerate)

Measured on an Apple M4 Mac (4 P-Cores, median of 10 runs):

| Matrix ($M \times K$) | $N$ | Sparsity | Non-Zeros | Scalar Ref (ms) | Apple Accelerate (ms) | NEON 1 Core Base (ms) | NEON 1 Core Best (ms) | NEON 4 Cores (ms) | NEON 4 Cores (GFLOPS) | Speedup vs Accelerate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1024 x 1024 | 64  | 90.0% | 105,490 | 1.21 | 0.23 | 0.59 | 0.41 (Tile 4x16) | 0.18 | 73.80  | 1.26x |
| 1024 x 1024 | 128 | 95.0% | 52,982  | 0.70 | 0.18 | 0.46 | 0.34 (U=6)        | 0.15 | 90.67  | 1.18x |
| 2048 x 2048 | 64  | 95.0% | 210,889 | 1.24 | 0.38 | 0.75 | 0.74 (U=6)        | 0.25 | 106.98 | 1.50x |
| 2048 x 2048 | 128 | 98.0% | 84,462  | 0.80 | 0.25 | 0.59 | 0.56 (U=6)        | 0.22 | 100.37 | 1.17x |
| 4096 x 4096 | 64  | 98.0% | 335,255 | -    | 0.59 | 1.27 | 1.19 (U=6)        | 0.39 | 108.69 | 1.50x |
| 4096 x 4096 | 128 | 99.0% | 167,773 | -    | 0.46 | 1.40 | 1.19 (U=4)        | 0.43 | 99.97  | 1.07x |
| 8192 x 8192 | 128 | 99.0% | 670,964 | -    | 1.79 | 5.95 | 5.36 (U=6)        | 1.62 | 105.79 | 1.10x |
| 8192 x 8192 | 256 | 99.5% | 335,422 | -    | 1.98 | 5.23 | 4.90 (U=4)        | 1.57 | 109.04 | 1.26x |

Our implementation runs **1.07x to 1.50x faster** than Apple Accelerate Sparse BLAS.

---

## 5. Further Tests: Unroll Factors and Tile Shapes

In `include/spmm_advanced.hpp`, we tested unrolling by $U=1, 2, 4, 6$ and grouping rows into tiles ($2\times16, 4\times16, 8\times16$) on a single core:

| Matrix & Sparsity | Accelerate | U=1 (Base) | U=2 | U=4 | U=6 | Tile 2x16 | Tile 4x16 | Tile 8x16 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1024x1024, N=64, s=90% | 0.24 ms | 0.59 ms (23.0G) | 0.47 ms (28.4G) | 0.45 ms (30.3G) | **0.41 ms (33.0G)** | 0.41 ms (33.0G) | 0.41 ms (33.3G) | 0.43 ms (31.5G) |
| 1024x1024, N=128, s=95% | 0.17 ms | 0.37 ms (36.4G) | 0.36 ms (37.6G) | 0.36 ms (37.5G) | **0.35 ms (38.4G)** | **0.35 ms (38.6G)** | 0.37 ms (37.0G) | 0.42 ms (32.5G) |
| 2048x2048, N=64, s=95% | 0.38 ms | 0.75 ms (35.9G) | 0.75 ms (35.8G) | 0.74 ms (36.6G) | **0.72 ms (37.5G)** | 0.73 ms (37.2G) | 0.74 ms (36.7G) | 0.82 ms (33.0G) |
| 2048x2048, N=128, s=98% | 0.26 ms | 0.58 ms (37.5G) | 0.58 ms (37.6G) | 0.57 ms (37.9G) | **0.56 ms (38.4G)** | 0.60 ms (35.8G) | 0.61 ms (35.3G) | 0.68 ms (32.0G) |
| 4096x4096, N=64, s=98% | 0.63 ms | 1.35 ms (31.9G) | 1.35 ms (31.8G) | 1.32 ms (32.5G) | **1.26 ms (34.0G)** | 1.29 ms (33.4G) | 1.27 ms (33.7G) | 1.44 ms (29.8G) |
| 4096x4096, N=128, s=99% | 0.48 ms | 1.21 ms (35.5G) | 1.21 ms (35.5G) | **1.19 ms (36.2G)** | 1.20 ms (35.9G) | 1.26 ms (34.1G) | 1.27 ms (33.9G) | 1.39 ms (30.9G) |
| 8192x8192, N=128, s=99% | 1.77 ms | 6.17 ms (27.8G) | 5.82 ms (29.5G) | 5.57 ms (30.8G) | **5.36 ms (32.0G)** | 5.73 ms (30.0G) | 6.03 ms (28.5G) | 6.52 ms (26.4G) |
| 8192x8192, N=256, s=99.5% | 2.42 ms | 4.90 ms (35.1G) | 4.96 ms (34.6G) | **4.90 ms (35.1G)** | 5.00 ms (34.3G) | 5.41 ms (31.8G) | 5.73 ms (30.0G) | 6.72 ms (25.6G) |

### Observations:

1. **$U=6$ unrolling**: Works well when rows have around 100 non-zeros (at 90% sparsity), reducing latency from 0.59 ms to 0.41 ms by cutting loop overhead. When rows are very sparse (98%-99%), each row has only 10 to 20 elements, so $U=2$ or $U=4$ is enough.
2. **$2\times16$ and $4\times16$ tiles**: Running 2 or 4 rows together allows independent instructions to be issued back-to-back, reducing pipeline wait times.
3. **$8\times16$ tile**: Performance drops in all cases. An $8\times16$ tile requires 32 registers just to hold output accumulators ($8 \text{ rows} \times 4 \text{ vectors}$). Since ARMv9 only has 32 vector registers in total, there are no spare registers to load input data. The compiler is forced to spill values to the stack, which makes it slower.

---

## 6. Conclusion

* The hand-written NEON SpMM kernel matches Apple Accelerate with zero difference.
* By vectorizing along the continuous dense dimension and keeping accumulators in registers, it runs 1.07x to 1.50x faster than Apple's library on 4 cores.
* Unrolling up to $U=6$ helps when rows have enough non-zeros.
* Register tiles of $2\times16$ and $4\times16$ work best, while $8\times16$ exceeds the CPU's 32-register limit and causes stack spills.
