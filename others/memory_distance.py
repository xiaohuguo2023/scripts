# Function to calculate linear index based on row-major or column-major ordering
def calculate_linear_index(tensor, idx, order='row'):
    if order not in ['row', 'column']:
        raise ValueError("Order must be 'row' or 'column'")

    if order == 'row':
        # Row-major order (C-style)
        linear_idx = sum(idx[i] * tensor.stride(i) for i in range(tensor.dim()))
    else:
        # Column-major order (Fortran-style)
        reversed_dims = list(reversed(range(tensor.dim())))
        linear_idx = sum(idx[reversed_dims[i]] * tensor.stride(reversed_dims[i]) for i in range(tensor.dim()))

    return linear_idx

# Updated function to handle tensors of any dimension and print the results
def memory_and_element_distance_print(tensor, idx1, idx2, order='row'):
    if len(idx1) != tensor.dim() or len(idx2) != tensor.dim():
        raise ValueError("Index dimensions must match tensor dimensions")

    element_size = tensor.element_size()

    # Calculate the linear index for each element
    linear_idx1 = calculate_linear_index(tensor, idx1, order)
    linear_idx2 = calculate_linear_index(tensor, idx2, order)

    # Calculate memory distance
    memory_distance = abs(linear_idx2 - linear_idx1) * element_size
    # Calculate element distance
    element_distance = abs(linear_idx2 - linear_idx1)

    # Print the results
    print(f"Memory distance between tensor{idx1} and tensor{idx2} ({order}-major order): {memory_distance} bytes")
    print(f"Element distance between tensor{idx1} and tensor{idx2} ({order}-major order): {element_distance} elements")

    return memory_distance, element_distance

# Example usage with a 4D tensor and printing the results
tensor_4d = torch.arange(1, 49).view(2, 3, 4, 2) # Creating a 2x3x4x2 tensor
idx1_4d = (1, 0, 2, 1) # Element tensor[1, 0, 2, 1]
idx2_4d = (0, 2, 1, 0) # Element tensor[0, 2, 1, 0]
memory_and_element_distance_print(tensor_4d, idx1_4d, idx2_4d, 'row')  # Row-major order
memory_and_element_distance_print(tensor_4d, idx1_4d, idx2_4d, 'column')  # Column-major order

