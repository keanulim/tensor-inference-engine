import torch
import time
import math

def benchmark(name, fn, runs=100):
    # warmup
    fn()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    mean = sum(times) / len(times)
    stddev = math.sqrt(sum((t - mean)**2 for t in times) / len(times))
    print(f"{name} | mean: {mean:.4f} ms | stddev: {stddev:.4f} ms")

# GEMM 512x512
A = torch.ones(512, 512)
B = torch.ones(512, 512)
benchmark("GEMM 512x512      ", lambda: torch.mm(A, B))

# ReLU 1M elements
x = torch.full((1 << 20,), -0.5)
benchmark("ReLU 1M           ", lambda: torch.relu(x))

# SiLU 1M elements
x = torch.full((1 << 20,), 0.5)
benchmark("SiLU 1M           ", lambda: torch.nn.functional.silu(x))

# Softmax 10k elements
x = torch.ones(10000)
benchmark("Softmax 10k       ", lambda: torch.softmax(x, dim=0))

# LayerNorm d=768
x = torch.ones(768)
ln = torch.nn.LayerNorm(768)
benchmark("LayerNorm d=768   ", lambda: ln(x))

# Embedding lookup seq=512 d=768
weight = torch.ones(50257, 768)
token_ids = torch.full((512,), 100, dtype=torch.long)
benchmark("Embedding seq=512 ", lambda: torch.nn.functional.embedding(token_ids, weight))