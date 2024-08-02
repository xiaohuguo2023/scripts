from typing import Optional
import torch
import random
import triton
import yaml

torch.manual_seed(123)
random.seed(123)

def perf(ms, m, n, k):
    return 2 * m * n * k * 1e-12 / (ms * 1e-3)

def read_input_data(input_file):
    with open(input_file, 'r') as file:
        data = yaml.safe_load(file)
    return data

# Read input data
input_file = 'input_sizes.yaml'
problem_sizes = read_input_data(input_file)

for entry in problem_sizes:
    m = entry['M']
    n = entry['N']
    k = entry['K']
    print(f"Running matmul for m={m}, n={n}, k={k}")
    
    A = torch.randn(m, k, device="cuda", dtype=torch.float16)
    B = torch.randn(n, k, device="cuda", dtype=torch.float16).T
    C = torch.zeros((m, n), device="cuda", dtype=A.dtype)

    triton_ms = triton.testing.do_bench(lambda: torch.matmul(A, B))
    print(f"PyTorch: {triton_ms:.3f} ms  {perf(triton_ms, m, n, k):.3f} tflops")
