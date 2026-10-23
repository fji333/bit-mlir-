#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import numpy as np

# Empirical Benchmark Data on Apple M4
workloads = [
    "1024x1024\n(N=64, 90%)",
    "1024x1024\n(N=128, 95%)",
    "2048x2048\n(N=64, 95%)",
    "2048x2048\n(N=128, 98%)",
    "4096x4096\n(N=64, 98%)",
    "4096x4096\n(N=128, 99%)",
    "8192x8192\n(N=128, 99%)",
    "8192x8192\n(N=256, 99.5%)"
]

acc_lat = [0.23, 0.17, 0.40, 0.25, 0.62, 0.48, 1.78, 2.00]
neon_4p_lat = [0.19, 0.16, 0.25, 0.22, 0.40, 0.42, 1.63, 1.57]
speedups = [acc / n for acc, n in zip(acc_lat, neon_4p_lat)]

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

x = np.arange(len(workloads))
width = 0.35

# 1. Latency Comparison
rects1 = ax1.bar(x - width/2, acc_lat, width, label='Apple Accelerate (Sparse BLAS)', color='#4A90E2', alpha=0.9)
rects2 = ax1.bar(x + width/2, neon_4p_lat, width, label='Our NEON SpMM (4 P-Cores)', color='#50E3C2', alpha=0.9)

ax1.set_ylabel('Execution Latency (ms, Lower is Better)', fontsize=11, fontweight='bold')
ax1.set_title('End-to-End Latency: Apple Accelerate vs Our NEON SpMM', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(workloads, fontsize=8.5)
ax1.legend(frameon=True, facecolor='white', framealpha=0.9)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# 2. Speedup Plot
bars = ax2.bar(x, speedups, width=0.5, color='#F5A623', alpha=0.9, edgecolor='#D48806')
ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='Apple Accelerate Baseline (1.0x)')
ax2.set_ylabel('Speedup over Apple Accelerate (Higher is Better)', fontsize=11, fontweight='bold')
ax2.set_title('Relative Speedup over Apple Accelerate Sparse BLAS', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(workloads, fontsize=8.5)
ax2.set_ylim(0.8, 1.8)
ax2.legend(frameon=True, facecolor='white', framealpha=0.9)
ax2.grid(axis='y', linestyle='--', alpha=0.5)

for bar in bars:
    height = bar.get_height()
    ax2.annotate(f'{height:.2f}x',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=9.5, fontweight='bold')

plt.tight_layout()
output_dir = os.path.join(os.path.dirname(__file__), "../reports")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "spmm_m4_performance.png")
plt.savefig(output_path)
print(f"Chart successfully saved to: {output_path}")
