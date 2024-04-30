import json
import matplotlib.pyplot as plt

# Function to read and return data from a JSON file
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Load the benchmark results from two different files
file_path = 'results_64^3.json'
file_path1 = 'results_basetriton.json'
benchmark_results = read_json_file(file_path)
benchmark_results1 = read_json_file(file_path1)

# Prepare data for the plot
x_values = [result["m"] * result["n"] * result["k"] for result in benchmark_results]  # m*n*k for each result
x_values1 = [result["m"] * result["n"] * result["k"] for result in benchmark_results1]  # m*n*k for each result from the second file
y_values = [result["speedup"] for result in benchmark_results]  # Speedup for each result
z_values = [result["speedup"] for result in benchmark_results1]  # Speedup for each result from the second file

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(x_values, y_values, color='blue', alpha=0.5, label='Results 64x64x64')
plt.scatter(x_values1, z_values, color='red', alpha=0.5, label='Results triton matmul')
plt.title('Matrix Multiplication Performance Speedup')
plt.xlabel('Matrix Size (m*n*k)')
plt.ylabel('Speedup (Triton/PyTorch)')
plt.xscale('log')  # Using a logarithmic scale for the x-axis to better handle wide range of values
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.legend()  # Add a legend to differentiate the datasets

# Save the plot as a PDF file
plt.savefig('matrix_multiplication_speedup_comparisonTS.png')

plt.show()  # Show the plot as well if you want to view it in the notebook or script output

