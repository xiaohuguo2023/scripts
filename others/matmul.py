import torch
import triton
import triton.language as tl

@triton.jit
def add_matrices_atomic(
    A, B, C,
    N, M,
):
    row = tl.program_id(0)
    col = tl.program_id(1)
    if row < N and col < M:
        idx = row * M + col
        # Load elements from A and B using tl.load
        a_val = tl.load(A + idx)
        b_val = tl.load(B + idx)
        # Perform addition and use tl.atomic_add for updating C
        tl.atomic_add(C + idx, a_val + b_val)

@triton.jit
def add_matrices_no_atomic(
    A, B, C,
    N, M,
):
    row = tl.program_id(0)
    col = tl.program_id(1)
    if row < N and col < M:
        idx = row * M + col
        # Load elements from A and B
        a_val = tl.load(A + idx)
        b_val = tl.load(B + idx)
        # Compute sum and store result in C using tl.store
        tl.store(C + idx, a_val + b_val)

# Input size
N, M = 1024, 1024
A = torch.randn(N, M, device='cuda')
B = torch.randn(N, M, device='cuda')
C = torch.zeros_like(A)

# Launch the kernel
# Calculate the number of blocks needed
num_blocks = ((N + 31) // 32, (M + 31) // 32)
# Define grid of blocks (each block can process one element)
grid = (num_blocks[0] * 32, num_blocks[1] * 32)

add_matrices_no_atomic[grid](A, B, C, N, M)
add_matrices_atomic[grid](A, B, C, N, M)

