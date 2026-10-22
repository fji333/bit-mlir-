# Apple M4 W4A16 GEMM Performance Study

This project tests and explains how fast an Apple M4 CPU core can run matrix multiplication when weights are 4-bit integers and activations are 16-bit floating-point numbers (W4A16). 

It compares hand-written ARM NEON vector code against textbook cache blocking, measures where the time goes, and checks the results against the physical limits of the chip.

---

## 1. Requirements and Setup

### Hardware
* **CPU**: Apple Silicon Mac (tested on Apple M4).
* **Architecture**: ARM64 / ARMv9 (supports NEON and FP16 vector instructions).

### Software
* **Operating System**: macOS (uses Apple-specific thread QoS APIs).
* **C++ Compiler**: Apple Clang / LLVM (supports C++11 and `-mcpu=native`).
* **Python** (only needed for making the chart):
  * Python 3.8+
  * `matplotlib`
  * `numpy`

### Install Python Packages (Optional)
If you want to run the chart script, install the python tools:
```bash
pip3 install matplotlib numpy
```

---

## 2. Folder and File Structure

```text
W4A16_M4/
├── README.md                           # This guide
├── plot_roofline.py                    # Python script to draw the Roofline chart
│
├── benchmarks/                         # Main experiment code and run scripts
│   ├── benchmark_massive.cpp           # Main W4A16 matrix test with tiling options
│   ├── benchmark_fp16_control.cpp      # Control test: pure FP16 without unpacking
│   └── run_experiments_massive.sh      # Bash script to run all tests automatically
│
├── assembly_proofs/                    # Proofs from the compiler
│   ├── benchmark_uk6.o                 # Compiled object file (Uk=6 unroll)
│   └── disasm_uk6.txt                  # Disassembled text showing real assembly
│
├── reports/                            # Detailed research reports
│   ├── W4A16_M4_Ablation_Study_Report.md     # Full report in Chinese
│   └── W4A16_M4_Ablation_Study_Report_EN.md  # Full report in English
│
├── bin/                                # Folder with compiled runnable programs
│   ├── benchmark_massive
│   ├── benchmark_fp16_control
│   └── ... (older binaries)
│
└── legacy_src/                         # Early prototype code during development
    ├── kernel.cpp
    ├── kernel8x16.cpp
    ├── unpack.cpp
    ├── swizzle.cpp
    ├── loop.cpp
    ├── gemm_final.cpp
    └── gemm_tiled.cpp
```

---

## 3. What Each Main File Does

* **`benchmarks/benchmark_massive.cpp`**:  
  The core test file. It holds an 8x16 vector micro-kernel. It uses preprocessor macros (`-DL2_MC`, `-DL2_NC`, `-DL1_MC`, `-DL1_NC`, `-DUK`) so you can turn cache blocking on or off, and change loop unrolling steps when compiling. It also locks the thread to a high-performance core (P-Core) and aligns memory to 16KB pages.

* **`benchmarks/benchmark_fp16_control.cpp`**:  
  The control experiment file. It uses the exact same 8x16 tile and memory pattern, but weights are stored directly as FP16. It removes all 4-bit unpacking steps (`vshlq`, `vshrq`, `vmovl`, `vcvtq`). This shows how fast the core runs when it only does math without unpacking.

* **`benchmarks/run_experiments_massive.sh`**:  
  An automated script. It builds and runs 11 different test setups one by one (no blocking, single-level blocking, two-level blocking, and different loop unrolling steps) and prints a clean comparison table.

* **`assembly_proofs/disasm_uk6.txt`**:  
  Disassembled assembly code. It gives physical proof that running too many operations at the same time causes register spilling (`str` and `ldr` stack writes).

* **`plot_roofline.py`**:  
  Draws the single-core Roofline model for the Apple M4 chip. It marks the theoretical top speed, the W4A16 speed, and the pure FP16 speed.

---

## 4. How to Build and Run

All commands below should be run from the project root folder or the `benchmarks/` folder.

### Method A: Run All Experiments with One Script (Recommended)

1. Open your terminal and go to the `benchmarks` folder:
   ```bash
   cd benchmarks
   ```
2. Make the script executable:
   ```bash
   chmod +x run_experiments_massive.sh
   ```
3. Run the script:
   ```bash
   ./run_experiments_massive.sh
   ```
This will compile and test all configurations. It outputs time in milliseconds and speed in GFLOPS for each case.

---

### Method B: Build and Run the Main W4A16 Test Manually

You can test different settings by passing macro flags to `clang++`.

#### 1. Best Setup (Flat linear scan, unroll Uk=6, no cache blocking):
```bash
clang++ -O3 -mcpu=native -std=c++11 \
  benchmarks/benchmark_massive.cpp \
  -DL2_MC=0 -DL2_NC=0 -DL1_MC=0 -DL1_NC=0 -DUK=6 \
  -o bin/benchmark_massive_uk6

./bin/benchmark_massive_uk6
```
*Expected speed: ~155 GFLOPS.*

#### 2. Test with Two-Level Cache Blocking (L2 128x256, L1 64x64, Uk=2):
```bash
clang++ -O3 -mcpu=native -std=c++11 \
  benchmarks/benchmark_massive.cpp \
  -DL2_MC=128 -DL2_NC=256 -DL1_MC=64 -DL1_NC=64 -DUK=2 \
  -o bin/benchmark_massive_tiled

./bin/benchmark_massive_tiled
```
*Expected speed: ~153.7 GFLOPS.*

---

### Method C: Run the Pure FP16 Control Test

This test removes 4-bit unpacking to measure the speed without unpacking overhead:

```bash
clang++ -O3 -mcpu=native -std=c++11 \
  benchmarks/benchmark_fp16_control.cpp \
  -o bin/benchmark_fp16_control

./bin/benchmark_fp16_control
```
*Expected speed: ~220 GFLOPS.*

---

### Method D: Draw the Roofline Chart

Run the Python script:
```bash
python3 plot_roofline.py
```
This saves an image to your desktop:
`~/Desktop/M4_Roofline_Model.png`.

---

### Method E: Check the Assembly Code

To see the real machine code generated by the compiler:
```bash
# Compile to an object file
clang++ -O3 -mcpu=native -std=c++11 -DUK=6 -c benchmarks/benchmark_massive.cpp -o assembly_proofs/benchmark_uk6.o

# Disassemble the object file to text
objdump -d assembly_proofs/benchmark_uk6.o > assembly_proofs/disasm_uk6.txt
```
Look inside `assembly_proofs/disasm_uk6.txt` to find:
* `fmla.8h`: Direct NEON half-precision vector math instructions.
* `str` and `ldr` on `[sp, ...]`: Registers running out of space (spilling) under high unrolling.

---

## 5. Main Results and Takeaways

| Test Case | Description | Measured Speed |
| :--- | :--- | :--- |
| **W4A16 Baseline** | Linear scan, no blocking, Uk=2 | **154.14 GFLOPS** |
| **W4A16 Single Blocking** | L1 or L2 single cache block | **~140 GFLOPS** (drops 9%) |
| **W4A16 Best Case** | Linear scan, no blocking, Uk=6 | **155.04 GFLOPS** (Peak) |
| **Pure FP16 Control** | No unpacking instructions, FP16 weights | **220.85 GFLOPS** (+42.5%) |
| **Theoretical Peak** | Single P-Core running at 4.4 GHz | **281.60 GFLOPS** |

### Why Did This Happen?
1. **Cache Blocking Hurt Performance**:  
   The Apple M4 has a smart prefetcher (DMP). Reading memory in a flat, straight line lets the prefetcher run at full speed. Cutting the memory into blocks breaks that line, causing a 9% speed drop.
2. **Unpacking Slows Down Math**:  
   For every 16 math instructions, the CPU runs 6 instructions just to unpack 4-bit data into 16-bit numbers. These unpacking instructions share the same compute pipes as the math instructions. This alone costs about 65 GFLOPS.
3. **Register Limit**:  
   The CPU has 32 vector registers. When unrolling past Uk=6, the core runs out of registers and has to spill data onto the stack, slowing things down.
