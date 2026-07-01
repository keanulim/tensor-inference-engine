## Benchmarks

CPU benchmarks run on Apple M-series (single-threaded, -O3) vs PyTorch (multithreaded Accelerate).
All times averaged over 100 runs.

| Kernel           | tensor_engine      | PyTorch            | Ratio  |
|------------------|--------------------|--------------------|--------|
| GEMM 512x512     | 16.45 ms ± 0.09   | 1.35 ms ± 0.58    | 12x    |
| ReLU 1M          | 0.15 ms ± 0.005   | 0.80 ms ± 0.49    | 0.19x  |
| SiLU 1M          | 2.91 ms ± 0.03    | 1.83 ms ± 0.90    | 1.6x   |
| Softmax 10k      | 0.051 ms ± 0.005  | 0.024 ms ± 0.002  | 2.1x   |
| LayerNorm d=768  | 0.003 ms ± 0.003  | 0.007 ms ± 0.001  | 0.4x   |
| Embedding seq=512| 0.031 ms ± 0.002  | 0.040 ms ± 0.024  | 0.8x   |

**Notes:**
- GEMM gap is due to PyTorch using multithreaded Apple Accelerate (BLAS). 
  Single-threaded ikj loop order achieves 15x speedup over naive ijk (255ms → 16ms).
- ReLU, LayerNorm, and Embedding match or beat PyTorch — elementwise and memory-bound 
  kernels are competitive single-threaded.
- CUDA port in progress targeting NVIDIA GTX 1060 (sm_61).
