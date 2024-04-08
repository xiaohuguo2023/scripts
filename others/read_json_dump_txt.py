import json

# Path to the JSON file
file_path = 'results2.json'
# Path to the output file
output_file_path = 'optimized_speedup_results.txt'

# Function to read and return data from a JSON file
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Function to calculate TFLOPS
def calculate_tflops(m, n, k, ms):
    return 2 * m * n * k * 1e-12 / (ms * 1e-3)

# Function to write results to a plain text file
def write_results(data, file_path):
    with open(file_path, 'w') as file:
        for item in data:
            file.write(f"M={item['m']}, N={item['n']}, K={item['k']}, "
                       f"PyTorch_ms: {item['pytorch_ms']}, TFLOPS (PyTorch): {item['pytorch_tflops']}, "
                       f"Triton_ms: {item['triton_ms']}, TFLOPS (Triton): {item['triton_tflops']}\n")

# Load the benchmark results
benchmark_results = read_json_file(file_path)

# List to store the results for dumping
results_to_dump = []

# Process each benchmark result
for result in benchmark_results:
    if result["speedup"] > 1:
        m, n, k = result["m"], result["n"], result["k"]
        pytorch_ms = result["pytorch_ms"]
        pytorch_tflops = calculate_tflops(m, n, k, pytorch_ms)
        
        # Find the smallest triton_ms
        smallest_triton_timing = min(result["triton"], key=lambda x: x["triton_ms"])
        triton_ms = smallest_triton_timing["triton_ms"]
        triton_tflops = calculate_tflops(m, n, k, triton_ms)

        # Append the relevant data to the list for dumping
        results_to_dump.append({
            "m": m,
            "n": n,
            "k": k,
            "pytorch_ms": pytorch_ms,
            "pytorch_tflops": pytorch_tflops,
            "triton_ms": triton_ms,
            "triton_tflops": triton_tflops
        })

# Write the filtered and processed results to a file
write_results(results_to_dump, output_file_path)

