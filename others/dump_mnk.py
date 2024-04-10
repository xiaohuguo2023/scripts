import json
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Filter benchmark results based on speedup range.')
parser.add_argument('--min-speedup', type=float, required=True, help='Minimum speedup value')
parser.add_argument('--max-speedup', type=float, help='Maximum speedup value (optional)')
args = parser.parse_args()

# Set speedup filter range based on input arguments
min_speedup = args.min_speedup
max_speedup = args.max_speedup if args.max_speedup is not None else float('inf')

# Path to the JSON file
file_path = 'results.json'
# Path to the output plain text file
output_file_path = 'filtered_dimensions.txt'

# Function to read and return data from a JSON file
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Function to write data to a plain text file
def write_plain_file(data, file_path):
    with open(file_path, 'w') as file:
        for item in data:
            m, n, k = item["m"], item["n"], item["k"]
            file.write(f"{m} {n} {k}\n")

# Use the function to load the benchmark results
benchmark_results = read_json_file(file_path)

# Filtered results based on the speedup condition
filtered_dimensions = []

# Process each benchmark result
for result in benchmark_results:
    speedup = result["speedup"]
    if min_speedup <= speedup <= max_speedup:
        filtered_dimensions.append(result)

# Write the filtered dimensions to a new plain text file
write_plain_file(filtered_dimensions, output_file_path)
