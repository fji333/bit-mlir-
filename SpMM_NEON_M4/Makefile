# Makefile for High-Performance CSR SpMM on Apple Silicon (M4)

CXX ?= /Library/Developer/CommandLineTools/usr/bin/clang++
SYSROOT ?= /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk

CXXFLAGS = -O3 -std=c++17 -mcpu=apple-m4 -isysroot $(SYSROOT) -I./include -Wall -Wextra
LDFLAGS = -framework Accelerate -lpthread

BIN_DIR = bin
TARGET_TEST = $(BIN_DIR)/test_correctness
TARGET_BENCH = $(BIN_DIR)/benchmark

all: $(TARGET_TEST) $(TARGET_BENCH)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(TARGET_TEST): tests/test_correctness.cpp include/*.hpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) tests/test_correctness.cpp $(LDFLAGS) -o $(TARGET_TEST)

$(TARGET_BENCH): src/benchmark.cpp include/*.hpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) src/benchmark.cpp $(LDFLAGS) -o $(TARGET_BENCH)

test: $(TARGET_TEST)
	./$(TARGET_TEST)

bench: $(TARGET_BENCH)
	./$(TARGET_BENCH)

clean:
	rm -rf $(BIN_DIR)

.PHONY: all test bench clean
