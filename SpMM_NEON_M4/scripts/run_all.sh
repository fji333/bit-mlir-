#!/bin/bash

# One-Click Build & Benchmark Runner for SpMM_NEON_M4 on Apple Silicon


set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${PROJECT_ROOT}/bin"
mkdir -p "${BIN_DIR}"

CLANG_BIN="/Library/Developer/CommandLineTools/usr/bin/clang++"
if [ ! -f "${CLANG_BIN}" ]; then
    CLANG_BIN="clang++"
fi

SYSROOT="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
SYSROOT_FLAG=""
if [ -d "${SYSROOT}" ]; then
    SYSROOT_FLAG="-isysroot ${SYSROOT}"
fi

CXXFLAGS="-O3 -std=c++17 -mcpu=apple-m4 ${SYSROOT_FLAG} -I${PROJECT_ROOT}/include -Wall -Wextra"
LDFLAGS="-framework Accelerate -lpthread"

echo "=== Compiling and running correctness test ==="
"${CLANG_BIN}" ${CXXFLAGS} "${PROJECT_ROOT}/tests/test_correctness.cpp" ${LDFLAGS} -o "${BIN_DIR}/test_correctness"
"${BIN_DIR}/test_correctness"

echo ""
echo "=== Compiling and running benchmark ==="
"${CLANG_BIN}" ${CXXFLAGS} "${PROJECT_ROOT}/src/benchmark.cpp" ${LDFLAGS} -o "${BIN_DIR}/benchmark"
"${BIN_DIR}/benchmark"

echo ""
echo "Done."
