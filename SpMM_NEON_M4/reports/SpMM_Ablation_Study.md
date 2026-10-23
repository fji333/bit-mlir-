# SpMM Optimization Ablation Study: Unroll Factors and Tile Shapes

This report tests different unroll factors ($U=1, 2, 4, 6$) and multi-row register tiles ($1\times16, 2\times16, 4\times16, 8\times16$) on an Apple M4 CPU.

All test code is kept in separate files (`include/spmm_advanced.hpp`, `src/benchmark_advanced.cpp`, `tests/test_advanced_correctness.cpp`) so the main codebase remains untouched.

---

## Benchmark Data (Apple M4 Single Performance Core)

Data measured using single-thread execution on an Apple M4 performance core. Latency is the median of 10 runs (in milliseconds).

| Matrix & Sparsity | Accelerate | U=1 (Base) | U=2 | U=4 | U=6 | Tile 2x16 | Tile 4x16 | Tile 8x16 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1024x1024, N=64, s=90% | 0.25 ms | 0.53 ms (25.6G) | 0.47 ms (28.5G) | 0.44 ms (30.4G) | 0.41 ms (32.8G) | **0.39 ms (34.5G)** | **0.39 ms (34.4G)** | 0.42 ms (32.4G) |
| 1024x1024, N=128, s=95% | 0.18 ms | 0.36 ms (37.9G) | **0.34 ms (39.7G)** | **0.34 ms (39.6G)** | 0.35 ms (38.9G) | 0.35 ms (38.3G) | 0.36 ms (37.5G) | 0.40 ms (33.7G) |
| 2048x2048, N=64, s=95% | 0.40 ms | 0.75 ms (36.0G) | 0.78 ms (34.8G) | **0.74 ms (36.7G)** | **0.74 ms (36.3G)** | **0.74 ms (36.6G)** | **0.74 ms (36.6G)** | 0.84 ms (32.2G) |
| 2048x2048, N=128, s=98% | 0.26 ms | 0.58 ms (37.4G) | 0.58 ms (37.4G) | **0.57 ms (37.9G)** | 0.59 ms (37.0G) | 0.60 ms (36.0G) | 0.61 ms (35.4G) | 0.68 ms (32.0G) |
| 4096x4096, N=64, s=98% | 0.59 ms | 1.35 ms (31.8G) | 1.34 ms (31.9G) | 1.30 ms (33.0G) | 1.24 ms (34.5G) | **1.21 ms (35.5G)** | **1.21 ms (35.6G)** | 1.34 ms (32.0G) |
| 4096x4096, N=128, s=99% | 0.48 ms | 1.26 ms (34.1G) | 1.26 ms (34.1G) | **1.23 ms (35.0G)** | 1.24 ms (34.6G) | 1.33 ms (32.3G) | 1.30 ms (33.0G) | 1.44 ms (29.7G) |
| 8192x8192, N=128, s=99% | 1.77 ms | 6.17 ms (27.8G) | 5.82 ms (29.5G) | 5.57 ms (30.8G) | **5.36 ms (32.0G)** | 5.73 ms (30.0G) | 6.03 ms (28.5G) | 6.52 ms (26.4G) |
| 8192x8192, N=256, s=99.5% | 2.42 ms | 4.90 ms (35.1G) | 4.96 ms (34.6G) | **4.90 ms (35.1G)** | 5.00 ms (34.3G) | 5.41 ms (31.8G) | 5.73 ms (30.0G) | 6.72 ms (25.6G) |

---

## Analysis of Results

### 1. Unroll Factors: $U=1$ vs $U=2$ vs $U=4$ vs $U=6$
* When rows have enough non-zeros (such as $s=90\%$, about 100 non-zeros per row), increasing unroll from $U=1$ to $U=6$ cuts time from 0.53 ms to 0.41 ms (throughput increases from 25.6 to 32.8 GFLOPS, about +28% speedup).
* When rows are very sparse ($s=98\%$ or $99\%$, with only 10 to 20 non-zeros per row), $U=4$ is the sweet spot. $U=6$ does not add much because the loop finishes in only 1 to 2 unrolled iterations.

### 2. Multi-Row Tile Shapes: $2\times16$ and $4\times16$ vs $8\times16$
* **$2\times16$ (8 accumulators)** and **$4\times16$ (16 accumulators)** perform the best. Because consecutive instructions update different row registers, there is no waiting on previous results, and the processor can issue instructions smoothly.
* **$8\times16$ (32 accumulators) slows down across all test cases**:
  * An $8\times16$ tile in FP32 needs 32 vector registers just for accumulators ($8 \text{ rows} \times 4 \text{ vectors}$).
  * Since ARMv9 has only 32 vector registers in total, there are no free registers left to load matrix values. The compiler is forced to save and restore values from stack memory, causing extra memory traffic and slowing down execution.
