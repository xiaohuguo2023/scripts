import yaml
import matplotlib.pyplot as plt
import numpy as np
import torch
import triton.testing  # Make sure Triton is installed and available

def load_yaml(filename):
    """Load YAML data from a file."""
    with open(filename, "r") as f:
        data = yaml.safe_load(f)
    print(f"Loaded {len(data)} items from {filename}")
    return data

def extract_data(data):
    """
    Extract x-axis labels (based on M, N, K) and TFLOPS values.
    Returns a tuple (x_labels, tflops) where:
      - x_labels: list of strings representing matrix sizes
      - tflops: list of floats for TFLOPS
    """
    x_labels = [f"M={item['M']}, N={item['N']}, K={item['K']}" for item in data]
    tflops = [float(item['TFLOPS']) for item in data]
    return x_labels, tflops

def make_unique_labels(labels):
    """
    Given a list of labels, append a suffix to duplicate labels so that they are unique.
    """
    counts = {}
    unique = []
    for label in labels:
        if label in counts:
            counts[label] += 1
            unique_label = f"{label} ({counts[label]})"
        else:
            counts[label] = 0
            unique_label = label
        unique.append(unique_label)
    return unique

def perf(ms, m, n, k):
    """Calculate TFLOPS performance given time in milliseconds."""
    return 2 * m * n * k * 1e-12 / (ms * 1e-3)

def run_pytorch_bench(matrix_data):
    """
    Run PyTorch GEMM for each matrix size in matrix_data.
    Returns a list of TFLOPS values.
    """
    torch_tflops = []
    print("Running PyTorch GEMM performance...")
    for item in matrix_data:
        m = item['M']
        n = item['N']
        k = item['K']
        print(f"Running matmul for m={m}, n={n}, k={k}")

        # Create random matrices on CUDA using float16
        A = torch.randn(m, k, device="cuda", dtype=torch.float16)
        B = torch.randn(n, k, device="cuda", dtype=torch.float16).T

        # Measure the time for torch.matmul using Triton benchmark helper
        triton_ms = triton.testing.do_bench(lambda: torch.matmul(A, B))
        torch_perf = perf(triton_ms, m, n, k)
        torch_tflops.append(torch_perf)
        print(f"PyTorch: {triton_ms:.3f} ms  {torch_perf:.3f} TFLOPS")
    return torch_tflops

def plot_grouped_bar_chart(x_labels, tflops_streamk, tflops_persistent, torch_tflops, output_file="tflops_comparison.png"):
    """
    Plot a grouped bar chart with:
      - x_labels: unique labels for the matrix sizes.
      - tflops_streamk: TFLOPS values for the streamk file.
      - tflops_persistent: TFLOPS values for the persistent file.
      - torch_tflops: TFLOPS values from PyTorch GEMM.
    The plot is saved as a PNG file.
    """
    x = np.arange(len(x_labels))  # label locations
    width = 0.25  # width for each bar

    fig, ax = plt.subplots(figsize=(14, 6))
    rects1 = ax.bar(x - width, tflops_streamk, width, label='StreamK64CUs', edgecolor='black')
    rects2 = ax.bar(x, tflops_persistent, width, label='Persistent64CUs', edgecolor='black')
    rects3 = ax.bar(x + width, torch_tflops, width, label='hipBLASlt64CUs', edgecolor='black')

    # Set labels and title
    ax.set_xlabel("Matrix Sizes (M, N, K)")
    ax.set_ylabel("TFLOPs")
    ax.set_title("TFLOPs Comparison by Matrix Sizes")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
#    ax.legend()

    fig.tight_layout()
    # Save the plot as a PNG file
    fig.savefig(output_file, dpi=300)
    print(f"Plot saved as {output_file}")
    plt.show()

def main():
    # Load the YAML files
    streamk_data = load_yaml("streamk_gemm_64cus.yaml")
    persistent_data = load_yaml("persistent_gemm_64cus.yaml")

    # Extract data from each file
    x_labels1, tflops_streamk = extract_data(streamk_data)
    x_labels2, tflops_persistent = extract_data(persistent_data)

    # Make unique labels (assumes both YAML files have the same ordering)
    x_labels = make_unique_labels(x_labels1)

    # Run PyTorch GEMM benchmark on the same matrix sizes (using the first YAML file's data)
    torch_tflops = run_pytorch_bench(streamk_data)

    # Plot grouped bar chart comparing all three methods
    plot_grouped_bar_chart(x_labels, tflops_streamk, tflops_persistent, torch_tflops)

if __name__ == "__main__":
    main()
