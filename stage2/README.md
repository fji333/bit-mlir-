# Triton CPU on Apple Silicon: Fused 4-Bit Matrix Multiplication

This repository contains our work on porting the Triton compiler to Apple Silicon CPUs (tested on Apple M4) and creating a fast, fused 4-bit weight matrix multiplication kernel for large language model text generation.

---

## 1. What This Project Does

When large language models generate text, the CPU spends most of its time reading model weights from main memory instead of doing calculations. To save memory, weights are packed into 4 bits. 

Normally, PyTorch unpacks these 4-bit numbers into 16-bit floats, writes them back to main memory, and reads them again to multiply. This wastes memory bandwidth.

In this project:
1. We modified the Triton compiler to build and run on Apple Silicon CPU (ARM64 / NEON).
2. We wrote a kernel that unpacks 4-bit weights directly inside the CPU registers using simple bit shifts and masks, multiplying them right away.
3. We studied how compiler tile sizes affect CPU registers and memory caches. We found that smaller tiles ($16 \times 16$) prevent register spilling and cut generated assembly code by 88.7%.
4. On language model text generation workloads, our fused kernel runs up to **31.32 times faster** than PyTorch by cutting out 87.7% of main memory traffic.

---

## 2. Environment and Requirements

### Hardware
* **Tested Machine**: Apple Silicon Mac with M4 chip (also works on M1, M2, and M3 Macs).
* **Architecture**: AArch64 (ARMv9.2-A with 32 128-bit NEON vector registers).

### Software
* **Operating System**: macOS (tested on macOS Sequoia 15).
* **Python**: Python 3.12 (using Anaconda at `/opt/anaconda3/bin/python`).
* **Required Libraries**:
  * `torch` (version 2.4 or newer)
  * `cmake`, `ninja` (for building C++ code)
  * `matplotlib` (for generating charts)
* **Custom Compiler Build**:
  * We built Triton from source using a custom LLVM 19 / MLIR toolchain.
  * The compiled library is linked to this folder using Python editable mode (`pip install -e`).

---

## 3. How to Set Up and Configure

### Step 1: Set the Environment Variable
Triton needs an environment variable to know it should run on CPU instead of GPU.

In your terminal:
```bash
export TRITON_CPU_BACKEND=1
```

Or at the very top of your Python script:
```python
import os
os.environ["TRITON_CPU_BACKEND"] = "1"

import torch
import triton
import triton.language as tl
```

### Step 2: VS Code / Cursor Editor Setup
If your editor shows yellow warning lines under `import triton` or `import triton.language`, make sure:
1. Your Python interpreter is set to `/opt/anaconda3/bin/python`.
2. The folder `.vscode/settings.json` has the following lines so the editor can find the source code:
```json
{
    "python.defaultInterpreterPath": "/opt/anaconda3/bin/python",
    "python.analysis.extraPaths": [
        "${workspaceFolder}/triton-cpu/python"
    ],
    "python.autoComplete.extraPaths": [
        "${workspaceFolder}/triton-cpu/python"
    ]
}
```

---

## 4. Folder Structure and File Descriptions

```text
Project2_Triton/
├── README.md                      # This file (project guide and overview)
├── PROJECT_HANDOVER_CONTEXT.md    # Detailed notes and setup for continuing this project
├── Project2_Research_Report.md    # Full academic research paper (Chinese version)
├── Project2_Research_Report_EN.md # Full academic research paper (English version)
│
├── kernels/                       # Core Triton kernels
│   ├── fp16_gemm.py               # Standard 16-bit float matrix multiplication kernel
│   └── w4a16_gemm.py              # Fused 4-bit weight unpacking + multiplication kernel
│
├── tests/                         # Accuracy and correctness tests
│   ├── test_correctness.py        # Compares Triton FP16 output against PyTorch
│   └── test_w4a16_correctness.py  # Compares Triton 4-bit output against PyTorch reference
│
├── benchmarks/                    # Speed and performance testing
│   ├── benchmark_fp16.py          # Measures FP16 speed across different matrix sizes
│   ├── benchmark_w4a16.py         # Measures 4-bit fused speed across matrix sizes
│   └── run_massive_full_suite.py  # Runs all tests to collect real data on the M4 chip
│
├── analysis/                      # Compiler intermediate code inspection
│   ├── dump_ir.py                 # Script that saves all 7 compiler stages to files
│   └── ir_dumps/                  # Output folder containing .mlir, .ll, .s, and .so files
│
├── scripts/                       # Build and patch scripts
│   ├── build_triton_m4.sh         # Shell script to build LLVM 19 and Triton CPU from source
│   └── patch_cpu.py               # Script that fixed the C++ enum crash in src/llvm.cc
│
├── history/                       # Saved records and charts
│   ├── README.md                  # Index of historical files
│   ├── 01_code_iterations/        # Code saved at different points during development
│   ├── 02_terminal_logs/          # Raw test output logs recorded from real runs
│   ├── 03_ir_artifacts/          # Assembly comparisons (64x64 vs 16x16 tile sizes)
│   ├── 04_visual_charts/          # High-resolution charts (Figure 1 through Figure 5)
│   └── 05_performance_reports/    # Summary performance tables
│
├── triton-cpu/                    # The modified Triton source tree with our CPU patches
├── triton-upstream-official/      # Original, unmodified Triton source for comparison
└── llvm-project/                  # Custom LLVM 19 / MLIR source code used to build Triton
```

---

## 5. How to Run the Code

Make sure your terminal is inside the project directory:
```bash
cd /Users/a15583507331/Downloads/Project2_Triton
```

### 1. Run Correctness Tests
Check that the math output matches PyTorch:
```bash
# Test normal FP16 matrix multiplication
python tests/test_correctness.py

# Test fused 4-bit weight matrix multiplication
python tests/test_w4a16_correctness.py
```
Both tests should print `✅ 验证通过！` (Verification Passed).

### 2. Run Speed Benchmarks
Test execution time and compute throughput:
```bash
# Test FP16 speed
python benchmarks/benchmark_fp16.py

# Test 4-bit fused speed
python benchmarks/benchmark_w4a16.py

# Run the full test suite (matrix sweeps, tile size sweeps, and LLM decoding batches)
python benchmarks/run_massive_full_suite.py
```

### 3. Save and View Compiler Code
See what the compiler generates at each stage (from high-level Python down to ARM64 assembly):
```bash
python analysis/dump_ir.py
```
The output files will be written to `analysis/ir_dumps/`.

### 4. Re-draw Performance Charts
Generate the 5 performance figures (saved as 300 DPI images):
```bash
python history/04_visual_charts/render_publication_figures.py
```
The images are saved to `history/04_visual_charts/` (`fig01` to `fig05`).

---

## 6. Key Research Findings (Quick Summary)

1. **Hardware-Aware Tile Size ($16 \times 16 \times 32$)**:
   * The GPU-default tile size ($64 \times 64$) needs 16 KB of registers.
   * An Apple M4 core only has 32 vector registers (512 bytes total).
   * This overload caused the compiler to generate **206,586 lines of assembly code** (8.09 MB), which clogged the CPU's 192 KB instruction cache.
   * Shrinking the tile size to $16 \times 16$ reduced the assembly by **88.7%** (down to 23,306 lines), letting the hot loop fit inside the fast cache.

2. **Speedups**:
   * **Square Matrices**: Our kernel is **3.05x to 6.84x faster** than PyTorch.
   * **Language Model Text Generation** ($M=128, N=4096, K=4096$):
     * PyTorch: 5.685 seconds
     * Our fused kernel: **0.181 seconds**
     * Speedup: **31.32x faster**, because it eliminates 87.7% of main memory traffic.

3. **Accuracy**:
   * The output matches PyTorch with an average error of only $0.002193$. This tiny difference is caused purely by normal floating-point addition order.
