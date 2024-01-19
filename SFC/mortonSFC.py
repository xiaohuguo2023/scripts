import numpy as np
import matplotlib.pyplot as plt

def compute_morton_index(x, y, num_bits):
    # Function to compute Morton index for given 2D coordinates (x, y)
    index = 0
    for i in range(num_bits):  # Using the specified number of bits for x and y
        index |= ((x & (1 << i)) << i) | ((y & (1 << i)) << (i + 1))
    return index

def decompose_domain(width, height, num_processes):
    num_bits = max(width.bit_length(), height.bit_length())  # Determine the number of bits needed
    morton_indices = []
    for y in range(height):
        for x in range(width):
            morton_indices.append((compute_morton_index(x, y, num_bits), x, y))

    morton_indices.sort()

    segments = [[] for _ in range(num_processes)]
    for i, cell in enumerate(morton_indices):
        segments[i % num_processes].append(cell)

    subdomains = []
    for i, segment in enumerate(segments):
        subdomains.append({
            'process_id': i,
            'cells': segment  # Store the entire segment of cells in each subdomain
        })

    return subdomains

def plot_morton_curve(width, height, subdomains):
    num_bits = max(width.bit_length(), height.bit_length())  # Determine the number of bits needed
    morton_indices = [cell for subdomain in subdomains for cell in subdomain['cells']]

    morton_indices.sort()

    x_values = [x for _, x, _ in morton_indices]
    y_values = [y for _, _, y in morton_indices]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title('Morton/Z-order Curve after Domain Decomposition')

    for i in range(len(morton_indices) - 1):
        x1, y1 = x_values[i], y_values[i]
        x2, y2 = x_values[i + 1], y_values[i + 1]
        ax.arrow(x1, y1, x2 - x1, y2 - y1, head_width=0.1, head_length=0.1, fc='black', ec='black')

    ax.scatter(x_values, y_values, c=np.arange(len(morton_indices)), cmap='viridis', s=50)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    plt.show()

# Example usage for any number of MPI tasks
width = 32
height = 32
num_processes = 7

subdomains = decompose_domain(width, height, num_processes)
plot_morton_curve(width, height, subdomains)

