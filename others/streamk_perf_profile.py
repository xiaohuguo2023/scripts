import json
import matplotlib.pyplot as plt

# Function to read and return data from a JSON file
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Load the benchmark results
file_path = 'results2.json'
benchmark_results = read_json_file(file_path)

# Prepare data for the plot
x_values = [result["m"] * result["n"] * result["k"] for result in benchmark_results]  # m*n*k for each result
y_values = [result["speedup"] for result in benchmark_results]  # Speedup for each result

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(x_values, y_values, color='blue', alpha=0.5)
plt.title('Matrix Multiplication Performance Speedup')
plt.xlabel('Matrix Size (m*n*k)')
plt.ylabel('Speedup (PyTorch / Triton)')
plt.xscale('log')  # Using a logarithmic scale for the x-axis to better handle wide range of values
plt.grid(True, which="both", ls="--", linewidth=0.5)

# Save the plot as a PDF file
plt.savefig('matrix_multiplication_speedup2.pdf')

plt.show()  # Show the plot as well if you want to view it in the notebook or script output
