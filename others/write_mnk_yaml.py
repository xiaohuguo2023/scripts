import json

# Path to the JSON file
file_path = 'results2.json'
# Path to the output file, using .txt since we are custom formatting
output_file_path = 'filtered_dimensions.yaml'

# Function to read and return data from a JSON file
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Use the function to load the benchmark results
benchmark_results = read_json_file(file_path)

# Prepare to write formatted results to a text file
with open(output_file_path, 'w') as file:
    # Process each benchmark result
    for result in benchmark_results:
        if result["speedup"] > 1:  # Check if speedup is greater than 1
            # Manually format the string to match the desired YAML-like appearance
            formatted_line = f"- {{'M': {result['m']}, 'N': {result['n']}, 'K': {result['k']}, 'rowMajorA': 'T', 'rowMajorB': 'N'}}\n"
            # Write the formatted string to the file
            file.write(formatted_line)
