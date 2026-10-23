# SpMM on Apple M4 with NEON

This project implements sparse matrix times dense matrix multiplication (SpMM) on Apple Silicon (M4 CPU) using C++ and ARM NEON vector instructions.

Formula:
$$C = A \times B$$
* $A$: Sparse matrix stored in CSR format.
* $B$: Dense matrix stored in row-major order.
* $C$: Output dense matrix.

We compare this implementation with Apple's Accelerate framework (`sparse_matrix_product_dense_float`).

---

## File Structure

* `include/csr_matrix.hpp`: CSR matrix data structure and random matrix generator.
* `include/spmm_reference.hpp`: Simple scalar loop used to check correctness.
* `include/spmm_neon.hpp`: Main NEON vector implementation (single-thread and 4-core multi-thread).
* `include/spmm_accelerate.hpp`: Wrapper around Apple Accelerate Sparse BLAS.
* `include/spmm_advanced.hpp`: Advanced tests with different loop unroll factors (U=2, 4, 6) and multi-row tiles (2x16, 4x16, 8x16).
* `src/benchmark.cpp`: Main benchmark comparing our kernel against Apple Accelerate.
* `src/benchmark_advanced.cpp`: Benchmark comparing different unroll and tile configurations.
* `tests/test_correctness.cpp`: Verifies output against scalar code and Apple Accelerate.
* `scripts/run_all.sh`: Script to compile and run main tests and benchmarks.
* `scripts/run_advanced.sh`: Script to run the unroll and tile tests.
* `reports/`: Markdown experiment reports.

---

## How It Works

1. **Vectorizing along $N$**:  
   Matrix $A$ is sparse, but each row of matrix $B$ is stored contiguously in memory. We vectorize along the $N$ dimension using NEON registers (`float32x4_t`), loading and computing 4 floats per instruction.

2. **Accumulating in Registers**:  
   Instead of writing back to memory for every non-zero element, we keep the sum in 4 NEON registers (`c0`, `c1`, `c2`, `c3`) for 16 columns at a time. The result is written back to matrix $C$ only once at the end of each row.

---

## How to Run

### Main benchmark (vs Apple Accelerate):
```bash
./scripts/run_all.sh
```

### Unroll and Tile tests ($U=6$, $8\times16$, etc.):
```bash
./scripts/run_advanced.sh
```

---

## Benchmark Results (Apple M4)

### 1. Main Benchmark (Multi-threaded on 4 P-Cores)

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

Output difference against Apple Accelerate is 0.0 (exact match).

### 2. Unroll and Tile Results (Single Core)

Tested on single P-Core with different unroll factors ($U=1, 2, 4, 6$) and tile shapes ($2\times16, 4\times16, 8\times16$):

* **Loop unrolling ($U=2, 4, 6$)**: Unrolling by 6 works best when rows have many non-zeros (around 100 per row at 90% sparsity), dropping latency from 0.59 ms to 0.41 ms. When rows are very sparse (98%-99%), rows only have 10-20 elements, so $U=2$ or $U=4$ is enough.
* **$2\times16$ and $4\times16$ tiles**: Computing 2 or 4 rows together allows independent instructions to run side-by-side, giving the lowest latency (0.39 ms at 90% sparsity).
* **$8\times16$ tile**: Performance drops across all matrix sizes. An $8\times16$ tile takes 32 registers just for accumulators ($8 \text{ rows} \times 4 \text{ vectors}$). Since ARMv9 only has 32 vector registers total, the CPU runs out of registers and has to use the stack, which slows it down.
