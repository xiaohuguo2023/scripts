import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the benchmark results from the JSON file
file_path = 'results2.json'
with open(file_path, 'r') as file:
    benchmark_results = json.load(file)

# Convert the loaded data into a DataFrame
df = pd.DataFrame(benchmark_results)

# Function to calculate and return min, mean, max speedup for a given dimension
def get_speedup_stats(grouped):
    return grouped['speedup'].agg(['min', 'mean', 'max']).reset_index()

# Group data by each dimension and calculate stats
grouped_m = get_speedup_stats(df.groupby('m'))
grouped_n = get_speedup_stats(df.groupby('n'))
grouped_k = get_speedup_stats(df.groupby('k'))

# Prepare the plot
fig, axs = plt.subplots(3, 3, figsize=(18, 12), sharey=True)
dimensions = ['m', 'n', 'k']
grouped_stats = [grouped_m, grouped_n, grouped_k]

for i, dim in enumerate(dimensions):
    for j, stat in enumerate(['min', 'mean', 'max']):
        # Scatter plot for raw speedup values vs. dimension
        axs[i, j].scatter(df[dim], df['speedup'], label='Raw Speedup', alpha=0.5, s=10)
        
        # Line plot for the specific statistic (min, mean, max) over the unique values of the dimension
        unique_dim_values = grouped_stats[i][dim]
        stat_values = grouped_stats[i][stat]
        axs[i, j].plot(unique_dim_values, stat_values, label=f'{stat.capitalize()} Speedup', color='red', marker='o', linestyle='-', markersize=5)
        
        axs[i, j].set_title(f'{dim.upper()} - {stat.capitalize()} Speedup')
        axs[i, j].set_xlabel(f'{dim.upper()} Value')
        axs[i, j].set_xscale('log')
        axs[i, j].grid(True, which='both', linestyle='--', linewidth=0.5)
        axs[i, j].legend()

        if j == 0:
            axs[i, j].set_ylabel('Speedup')

fig.suptitle('Matrix Multiplication Speedup Analysis Across Dimensions')
plt.tight_layout()
plt.savefig('matrix_multiplication_speedup_analysis.pdf')
plt.show()
