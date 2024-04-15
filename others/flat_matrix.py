import torch
import triton
import triton.language as tl

# Define the Triton kernel for flattening a 2D matrix to a 1D array
@triton.jit
def flatten_kernel(
    input_matrix,   # Pointer to the input matrix in global memory
    output_array,   # Pointer to the output array in global memory
    rows,           # Number of rows in the input matrix
    cols,           # Number of columns in the input matrix
    BLOCK_SIZE: tl.constexpr,
):
    # Compute row and column index for this thread
    row = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Ensure we do not go out of bounds
    mask = (row < rows) & (col < cols)

    # Compute the linear index in the flattened output array
    idx = row * cols + col

    # Load from the input matrix and store in the output array
    val = tl.load(input_matrix + row[:, None] * cols + col[None, :], mask=mask)
    tl.store(output_array + idx, val, mask=mask)

# Setup the PyTorch tensors
rows = 64
cols = 64
input_matrix = torch.randn(rows, cols, dtype=torch.float32, device='cuda')
output_array = torch.empty(rows * cols, dtype=torch.float32, device='cuda')

# Define grid and block sizes for the kernel
BLOCK_SIZE = 16  # Define according to your GPU's capability and the size of the matrix
grid = (rows // BLOCK_SIZE, cols // BLOCK_SIZE)

# Launch the kernel
flatten_kernel[grid,](
    input_matrix, output_array,
    rows, cols,
    BLOCK_SIZE=BLOCK_SIZE
)

# Check results
expected_output = input_matrix.view(-1)  # Flatten the matrix using PyTorch for verification
print("Does the flattened output match the expected output?", torch.allclose(output_array, expected_output))

