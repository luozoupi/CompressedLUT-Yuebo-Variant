CXX ?= g++
NVCC ?= /usr/local/cuda/bin/nvcc
CFLAGS=-O3 -std=c++11
BENCH_CXXFLAGS=-O3 -std=c++17 -pthread
NVCCFLAGS=-O3 -std=c++17 -arch=sm_120
#CFLAGS=-g

all: compressedlut

compressedlut: compressedlut.o
	$(CXX) $(CFLAGS) -o compressedlut compressedlut.o

compressedlut.o: compressedlut.cpp compressedlut.h
	$(CXX) $(CFLAGS) -c compressedlut.cpp

bench_cpu: benchmarks/bench_cpu

bench_cuda: benchmarks/bench_cuda

bench_cuda_v2: benchmarks/v2/bench_cuda_v2

bench: bench_cpu bench_cuda

benchmarks/compressedlut_bench.o: compressedlut.cpp compressedlut.h
	$(CXX) $(BENCH_CXXFLAGS) -DCOMPRESSEDLUT_NO_MAIN -c compressedlut.cpp -o benchmarks/compressedlut_bench.o

benchmarks/bench_cpu: benchmarks/bench_cpu.cpp benchmarks/bench_common.h compressedlut.cpp compressedlut.h
	$(CXX) $(BENCH_CXXFLAGS) -DCOMPRESSEDLUT_NO_MAIN compressedlut.cpp benchmarks/bench_cpu.cpp -o benchmarks/bench_cpu

benchmarks/bench_cuda: benchmarks/bench_cuda.cu benchmarks/bench_common.h compressedlut.h benchmarks/compressedlut_bench.o
	$(NVCC) $(NVCCFLAGS) benchmarks/bench_cuda.cu benchmarks/compressedlut_bench.o -o benchmarks/bench_cuda

benchmarks/v2/bench_cuda_v2: benchmarks/v2/bench_cuda_v2.cu benchmarks/bench_common.h compressedlut.h benchmarks/compressedlut_bench.o
	$(NVCC) $(NVCCFLAGS) benchmarks/v2/bench_cuda_v2.cu benchmarks/compressedlut_bench.o -o benchmarks/v2/bench_cuda_v2

clean: 
	rm -f *.o compressedlut compressedlut.exe benchmarks/*.o benchmarks/bench_cpu benchmarks/bench_cuda benchmarks/v2/bench_cuda_v2

.PHONY: all bench bench_cpu bench_cuda bench_cuda_v2 clean
