import yaml
import re

def calculate_gemm_bandwidth_fp16(M, N, K, execution_time_us):
    """
    Calculate the bandwidth required for a GEMM operation using FP16 data type.

    execution_time_us (float): Execution time in microseconds.

    Returns:
    float: Bandwidth in bytes per second.
    """

    # Data transfer calculations
    # Assuming each element is a 2-byte float (FP16)
    bytes_per_element = 2

    # Reads
    reads_A = M * K * bytes_per_element
    reads_B = K * N * bytes_per_element

    # Writes
    writes_C = M * N * bytes_per_element

    # Total data transfer (in bytes)
    total_data_transfer = reads_A + reads_B + writes_C

    # Convert execution time from microseconds to seconds
    execution_time_seconds = execution_time_us / 1e6

    # Bandwidth calculation (in bytes/second)
    bandwidth = total_data_transfer / execution_time_seconds

    return bandwidth

def extract_time_from_comment(comment):
    """
    Extracts the time in microseconds from the comment in YAML.

    Parameters:
    comment (str): The comment string.

    Returns:
    float: The extracted time in microseconds.
    """
    match = re.search(r'time\(us\): (\d+\.?\d*)', comment)
    return float(match.group(1)) if match else None

# Load the YAML data from file
yaml_file_path = 'tuning_results_main_6_membounds.yaml'
operations = []

with open(yaml_file_path, 'r') as file:
    content = file.read()

# Preprocess the content to handle comments
lines = content.split('\n')
for line in lines:
    print(f"Processing line: {line}") 
    match = re.search(r'# (.*)$', line)
    if match:
        comment = match.group(1)
        print(f"Extracted comment: {comment}")  # Debug statement
        time_us = extract_time_from_comment(comment)
        if time_us is not None:
            print(f"Extracted time (us): {time_us}")  # Debug statement
            # Parse the YAML content before the comment
            yaml_content = line.split('#')[0].strip()
            if yaml_content:  # Check if there's YAML content to parse
                print(f"Parsing YAML content: {yaml_content}")  # Debug statement
                operation = yaml.safe_load(yaml_content)
                if isinstance(operation, list) and len(operation) == 1 and isinstance(operation[0], dict):
                    operation = operation[0]  # Extract the dictionary from the list
                    operation['time_us'] = time_us
                    operations.append(operation)
                    print(f"Added operation: {operation}")  # Debug statement
                else:
                    print(f"Parsed operation is not a valid single-item list: {operation}")  # Debug statement

# Final list of operations
print(f"Total operations parsed: {len(operations)}")  # Debug statement
print(f"Operations: {operations}")  # Debug statement

# Calculate and print the bandwidth for each operation
for operation in operations:
    try:
        M = operation['M']
        N = operation['N']
        K = operation['K']
        execution_time_us = operation['time_us']
        
        bandwidth = calculate_gemm_bandwidth_fp16(M, N, K, execution_time_us)
        print(f"{M},{N},{K},{execution_time_us},{bandwidth}")
        print(f"Required bandwidth for GEMM operation with FP16 (M={M}, N={N}, K={K}): {bandwidth / 1e9:.2f} GB/s")
    except KeyError as e:
        print(f"Missing key in operation: {e}, operation: {operation}")  # Debug statement
