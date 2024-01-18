import numpy as np
import time

def winograd_convolve_2x2_3x3(input_matrix, kernel):
    G = np.array([[1, 0], [0.5, 0.5], [0.5, -0.5], [0, 1]])
    GT = G.T
    AT = np.array([[1, 1, 1, 0], [0, 1, -1, -1]])
    A = AT.T

    # Transform the kernel
    kernel_transform = G @ kernel @ GT

    # Pad the input matrix and break it into four tiles
    input_padded = np.pad(input_matrix, [(0, 1), (0, 1)], mode='constant')
    tiles = [input_padded[i:i+2, j:j+2] for i in range(2) for j in range(2)]

    # Transform each tile and sum up the results
    output_transform = np.zeros((4, 4))
    for i, tile in enumerate(tiles):
        row, col = divmod(i, 2)
        tile_transformed = np.zeros((4, 4))
        tile_transformed[:2, :2] = tile
        output_transform += tile_transformed * kernel_transform

    # Inverse transform
    output = AT @ output_transform @ A
    return output[:2, :2]

def convolve2d(input_matrix, kernel):
    kernel = np.flipud(np.fliplr(kernel))  # Flipping the kernel
    output = np.zeros_like(input_matrix)

    padded_input = np.pad(input_matrix, [(kernel.shape[0]//2, kernel.shape[0]//2), 
                                         (kernel.shape[1]//2, kernel.shape[1]//2)], mode='constant')

    for i in range(input_matrix.shape[0]):
        for j in range(input_matrix.shape[1]):
            output[i, j] = (kernel * padded_input[i:i+kernel.shape[0], j:j+kernel.shape[1]]).sum()
    return output


# [Insert the convolve2d and winograd_convolve_2x2_3x3 functions here]

def benchmark(function, input_matrix, kernel, iterations=100):
    start_time = time.time()
    for _ in range(iterations):
        result = function(input_matrix, kernel)
    end_time = time.time()
    return (end_time - start_time) / iterations

# Define a larger input and kernel for a more noticeable performance difference
large_input = np.random.rand(100, 100)
large_kernel = np.random.rand(3, 3)

# Adjusting the input and kernel for Winograd (specific case)
winograd_input = large_input[:3, :3]  # Trimming the input for 3x3
winograd_kernel = large_kernel[:2, :2]  # Trimming the kernel for 2x2

# Benchmarking
general_conv_time = benchmark(convolve2d, large_input, large_kernel)
winograd_conv_time = benchmark(winograd_convolve_2x2_3x3, winograd_input, winograd_kernel)

print(f"Average execution time for general convolution: {general_conv_time:.6f} seconds")
print(f"Average execution time for Winograd convolution: {winograd_conv_time:.6f} seconds")
