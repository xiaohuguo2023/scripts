import triton
import triton.language as tl
import torch

@triton.jit
def atomic_add_kernel(
    A_ptr,  # Pointer to the memory location A
    xcd_id,  # XCD ID (0 or 1)
    value_to_add,  # Value to add (1.0 or 2.0)
):
    # Get the program ID (CU0 in this case)
    pid = tl.program_id(axis=0)

    # Only CU0 on the specified XCD will perform the atomic add
    if pid == 0:
        tl.atomic_add(A_ptr, value_to_add)

# Initialize the memory location A to 0
A = torch.tensor([0.0], device='cuda')

# Define the grid size (1 CU per XCD)
grid_size = (1,)

# Launch the kernel for XCD 0 (add 1.0)
atomic_add_kernel[grid_size](A, 0, 1.0)

# Launch the kernel for XCD 1 (add 2.0)
atomic_add_kernel[grid_size](A, 1, 2.0)

# Print the final value in A
print("Final value in A:", A.item())
