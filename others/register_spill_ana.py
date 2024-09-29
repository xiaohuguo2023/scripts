import re
import networkx as nx
import matplotlib.pyplot as plt
import mplcursors

def extract_execution_sequence(assembly_file):
    # Initialize directed graph and block metadata
    control_flow_graph = nx.DiGraph()
    block_metadata = {}

    # Regex patterns to capture labels, branch instructions, and source info
    label_pattern = re.compile(r'^(\.\w+|; %bb\.\d+):')
    jump_pattern = re.compile(r'(s_cbranch|s_branch)\s+(\.\w+|; %bb\.\d+)')
    source_info_pattern = re.compile(r'\.loc\s+(\d+)\s+(\d+)\s+(\d+)')

    current_label = None
    execution_order = []
    reached_end = False  # Flag to stop processing after s_endpgm

    # Read the assembly file
    with open(assembly_file, 'r') as file:
        for line in file:
            line = line.strip()

            # Stop processing once s_endpgm is encountered
            if 's_endpgm' in line:
                reached_end = True
                break

            # Skip processing if we've reached the end of relevant code
            if reached_end:
                continue

            # Identify labels (basic blocks like .LBB, .Ltmp, or %bb)
            label_match = label_pattern.match(line)
            if label_match:
                current_label = label_match.group(1)
                control_flow_graph.add_node(current_label)
                execution_order.append(current_label)
                continue

            # Identify source locations and add them to the current block's metadata
            source_match = source_info_pattern.search(line)
            if source_match and current_label:
                file_num, line_num, _ = source_match.groups()
                if current_label not in block_metadata:
                    block_metadata[current_label] = []
                block_metadata[current_label].append(f"Source File {file_num}, Line {line_num}")

            # Identify branch/jump instructions and connect blocks
            jump_match = jump_pattern.search(line)
            if jump_match and current_label:
                target_label = jump_match.group(2)
                control_flow_graph.add_edge(current_label, target_label)

    return control_flow_graph, execution_order, block_metadata

def plot_execution_sequence(control_flow_graph, execution_order, block_metadata):
    # Create a figure with two subplots: one for the graph and one for the legend
    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]}, figsize=(14, 8))

    # Plot the control flow graph on the left
    pos = nx.spring_layout(control_flow_graph, k=3, iterations=150, seed=42)
    nodes = nx.draw(control_flow_graph, pos, ax=ax1, with_labels=False, node_color='skyblue', node_size=1500, arrows=True)
    node_labels = {node: str(i+1) for i, node in enumerate(execution_order)}
    nx.draw_networkx_labels(control_flow_graph, pos, labels=node_labels, font_size=10, font_weight='bold', ax=ax1)

    # Create a list of blocks with their corresponding numbers
    numbered_blocks = [f"{i+1}: {block}" for i, block in enumerate(execution_order)]

    # Display the block name list in the second subplot (right-hand side)
    ax2.text(0.05, 1, '\n'.join(numbered_blocks), fontsize=12, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.8))

    # Remove axis from the legend area and draw a box around it
    ax2.axis('off')
    for spine in ax2.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

    # Draw a box around the graph
    for spine in ax1.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

    # Adjust layout and title
    plt.subplots_adjust(wspace=0.3)
    fig.suptitle("Execution Sequence of Assembly Code Blocks (Numbered Nodes)", fontsize=16)

    # Enable hover functionality with mplcursors
    cursor = mplcursors.cursor(nodes, hover=True)

    # Display tooltips on hover with the source file and line numbers (multiple sources per block)
    @cursor.connect("add")
    def on_add(sel):
        node_index = int(sel.target.index)
        block_name = execution_order[node_index]
        source_info = block_metadata.get(block_name, ['No Source Info'])
        sel.annotation.set_text(f"Block {node_labels[block_name]}\n" + "\n".join(source_info))

    plt.show()

# Load the assembly file and extract control flow graph and execution sequence
assembly_file = 'streamk_gemm.amdgcn'  # Replace with your actual assembly file path
control_flow_graph, execution_order, block_metadata = extract_execution_sequence(assembly_file)

# Plot the control flow and execution sequence
plot_execution_sequence(control_flow_graph, execution_order, block_metadata)

