# Project 3: Fast 4-Bit Matrix Math on Apple Silicon with MLIR

This project makes 4-bit language models run fast on Apple Silicon CPUs (like the Apple M4). 

Normally, running 4-bit weights on a CPU is slow. Most tools first unpack all 4-bit weights into big 16-bit numbers in main memory, and then do the math. This wastes a lot of memory bandwidth and slows down the CPU. 

In this project, we use **MLIR** to build a fast compiler pipeline. It unpacks the 4-bit numbers directly inside the CPU registers and does the math right away. This cuts memory traffic by **8.53 times** and runs **2.27 times faster than OpenAI Triton on CPU**.

---

## 1. What You Need (Environment & Requirements)

To build and run this project, you need:

1. **Computer**: A Mac with an Apple Silicon chip (Apple M1, M2, M3, or M4).
2. **Operating System**: macOS.
3. **C++ Compiler**: Apple Clang (`clang++`), which comes with Xcode Command Line Tools.
4. **Python**: Python 3.10 or newer with two libraries:
   * `torch` (PyTorch)
   * `numpy`
5. **LLVM and MLIR**: You need LLVM built from source with MLIR enabled. Specifically, the project uses these tools from your LLVM build folder:
   * `mlir-opt`
   * `mlir-translate`
   * `llc`
   * `libmlir_c_runner_utils.dylib` and `libmlir_runner_utils.dylib`

---

## 2. How to Set Up (Configuration)

Before running the scripts, you must check the paths to your LLVM build and your Python binary.

### Step 1: Open the environment script
Open `scripts/env_setup.sh` in any text editor.

### Step 2: Set your paths
Make sure the paths point to your actual LLVM build folder and Python path:

```bash
# Path to your LLVM build directory
export LLVM_BUILD_DIR="/Users/a15583507331/Downloads/Project2_Triton/llvm-project/build"

# Path to your Python binary
export PYTHON_EXEC="/opt/anaconda3/bin/python"
```

If your LLVM or Python is installed in a different place, change these two lines to match your computer.

### Step 3: Load the environment
In your terminal, run:

```bash
source scripts/env_setup.sh
```

---

## 3. How to Run (Step-by-Step)

All run scripts are in the `scripts/` folder. You can run them directly from the project root folder.

### Step 1: Compile the fast library
Run this command to compile the MLIR code into a native Mac library (`.dylib`):

```bash
./scripts/compile_pipeline.sh
```

**What this script does:**
1. It takes the matrix math code from `mlir_src/w4a16_linalg.mlir`.
2. It applies our best optimization schedule (`schedule_s6_deep_optimized.mlir`).
3. It converts the code step by step into LLVM format.
4. It uses `llc` and `clang++` to make `kernels/libw4a16_mlir.dylib`.

### Step 2: Run correctness tests
Run this command to check if the math output is 100% correct:

```bash
./scripts/run_tests.sh
```

**What this script does:**
* It runs standard unit tests first.
* Then it runs all **47 stress test cases** (testing small matrices, big matrices, strange odd sizes, and real model layer sizes).
* All 47 tests must pass with zero errors.

### Step 3: Run speed benchmarks
Run this command to test how fast the code runs on your hardware:

```bash
./scripts/run_benchmarks.sh
```

**What this script does:**
* It measures the speed (in GFLOPS) and latency (in milliseconds).
* It tests different matrix sizes from $128 \times 128$ up to $4096 \times 4096$.
* On an Apple M4 CPU, the single-core speed reaches **95.41 GFLOPS**.

### Step 4: Run the cache explorer
Run this command to test how different block sizes affect memory cache:

```bash
./scripts/run_cache_explorer.sh
```

**What this script does:**
* It tests 6 different optimization strategies (from simple loops to deep register tuning).
* It saves the test data and a report into `benchmarks/artifacts/`.

### Step 5: Test with PyTorch and HuggingFace models
Run this command to test replacing real model layers with our fast library:

```bash
./scripts/run_e2e_llm.sh
```

**What this script does:**
* It takes a PyTorch linear layer.
* It packs the weights into 4-bit numbers.
* It calls our compiled `.dylib` library to do the math and compares the result with regular PyTorch.

---

## 4. Project Folder Structure and What Files Do

Here is the full map of every folder and file in this project:

```text
Project3_MLIR_Engine/
│
├── README.md               # This guide.
│
├── mlir_src/               # All MLIR source code and optimization schedules.
│   ├── w4a16_linalg.mlir   # The base math definition (4-bit unpack + matrix multiply).
│   ├── schedule_m4.mlir    # Default schedule for Apple M4.
│   ├── w4a16_scheduled.mlir# Code snapshot after applying the schedule.
│   ├── w4a16_bufferized.mlir # Code snapshot after assigning memory buffers.
│   ├── w4a16_llvm_dialect.mlir # Code snapshot lowered to LLVM dialect.
│   └── schedules/          # 6 different optimization experiments:
│       ├── schedule_s1_no_fusion.mlir          # S1: No fusion (writes to memory, slow).
│       ├── schedule_s2_l1_only.mlir            # S2: L1 cache blocks + in-loop unpack.
│       ├── schedule_s3_l2_l1_hierarchical.mlir # S3: Two-level cache blocks.
│       ├── schedule_s4_m4_specialized.mlir     # S4: Register blocks + vector instructions.
│       ├── schedule_s5_wide_neon_specialized.mlir # S5: Wider vector blocks.
│       └── schedule_s6_deep_optimized.mlir     # S6: Our best schedule (in-register sum + unroll).
│
├── kernels/                # Output folder for compiled files.
│   ├── w4a16.ll            # Text version of LLVM intermediate code.
│   ├── w4a16.o             # Machine code object file.
│   └── libw4a16_mlir.dylib # The final shared library loaded by Python.
│
├── scripts/                # One-click shell scripts.
│   ├── env_setup.sh        # Sets up environment variables.
│   ├── compile_pipeline.sh # Builds libw4a16_mlir.dylib from MLIR.
│   ├── run_tests.sh        # Runs all 47 tests.
│   ├── run_benchmarks.sh   # Runs speed tests.
│   ├── run_cache_explorer.sh # Tests 6 schedules across cache levels.
│   └── run_e2e_llm.sh      # Demos replacing a PyTorch layer.
│
├── tests/                  # Test scripts.
│   ├── test_mlir_w4a16_correctness.py # Basic unit tests.
│   ├── comprehensive_test_suite.py    # 47 full stress test cases.
│   └── artifacts/          # Saved test reports and JSON data.
│       ├── test_telemetry.json        # Test numbers in JSON format.
│       └── test_verification_report.md# Full test verification report.
│
├── benchmarks/             # Benchmark scripts.
│   ├── benchmark_w4a16_mlir.py        # Measures GFLOPS and latency.
│   ├── benchmark_huge_matrices.py     # Measures big matrices (2048 and 4096).
│   ├── cache_tile_space_explorer.py   # Compares different block sizes.
│   └── artifacts/          # Saved benchmark reports.
│       ├── cache_exploration_telemetry.json
│       └── cache_hierarchy_analysis.md
│
├── engine/                 # PyTorch integration.
│   └── llm_layer_patcher.py# Connects Python tensors to our C-ABI library.
│
└── docs/                   # Detailed academic research reports.
    ├── Project3_Research_Report.md    # Full report in Chinese.
    └── Project3_Research_Report_EN.md # Full report in English.
```

---

## 5. Main Results and Key Numbers

Here is a quick summary of the test and benchmark results on an Apple M4 CPU:

| Metric | Regular PyTorch | OpenAI Triton (CPU) | Our MLIR Engine | Hand Assembly |
| :--- | :--- | :--- | :--- | :--- |
| **Speed (GFLOPS)** | 0.75 GFLOPS | 24.61 GFLOPS | **95.41 GFLOPS** | **155.04 GFLOPS** |
| **Speedup vs Triton**| 0.03x | 1.00x (Baseline) | **2.27x faster** | 6.30x faster |
| **$4096^2$ Matrix Time** | 5685 ms | ~200 ms | **79.65 ms (71x faster)** | ~48 ms |
| **Memory Traffic** | High (Writes 32MB) | Spills to stack | **Cut by 8.53x** | Cut by 8.53x |
| **Tests Passed** | N/A | N/A | **47 / 47 (100%)** | Basic tests |

---

## 6. Common Questions & Troubleshooting

* **Problem: `libw4a16_mlir.dylib` not found.**  
  **Fix**: Run `./scripts/compile_pipeline.sh` first to build the library.
* **Problem: `mlir-opt: command not found`.**  
  **Fix**: Run `source scripts/env_setup.sh` to add your LLVM build folder to your `PATH`.
* **Problem: Missing Python packages.**  
  **Fix**: Install PyTorch and NumPy using `pip install torch numpy`.
