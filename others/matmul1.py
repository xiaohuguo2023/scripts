import torch
import triton
import triton.language as tl

@triton.jit
def add_matrices(
    A, B, C, 
    N: tl.constexpr, M: tl.constexpr,
):
    # Define indices for the thread
    row = tl.program_id(0)
    col = tl.program_id(1)
    
    # Compute the linear index of the element to process
    idx = row * M + col
    
    # Ensure we do not go out of bounds
    if row < N and col < M:
        # Load elements from A and B
        a_val = tl.load(A + idx)
        b_val = tl.load(B + idx)
        
        # Compute the sum
        c_val = a_val + b_val
        
        # Store the result in C
        tl.store(C + idx, c_val)

# Define the size of the matrices
N, M = 1024, 1024

# Allocate matrices
A = torch.randn(N, M, device='cuda')
B = torch.randn(N, M, device='cuda')
C = torch.empty(N, M, device='cuda')

# Flatten the matrices for indexing
A_flat = A.flatten()
B_flat = B.flatten()
C_flat = C.flatten()

# Launch the kernel
# Calculate the number of blocks needed
num_blocks = ((N + 31) // 32, (M + 31) // 32)
# Define grid of blocks (each block can process one element)
grid = (num_blocks[0] * 32, num_blocks[1] * 32)

add_matrices[grid](A_flat, B_flat, C_flat, N, M)

