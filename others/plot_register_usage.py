import re
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

# Define regular expressions at the module level so they are accessible to all functions
# Updated patterns to avoid matching numbers inside VGPR ranges
vgpr_single_pattern = re.compile(r'(?<!\[)v(\d+)(?![\d:])')          # Matches individual VGPRs not in ranges
vgpr_range_pattern = re.compile(r'v\[(\d+):(\d+)\]')                 # Matches VGPR ranges
sgpr_single_pattern = re.compile(r'(?<!\[)s(\d+)(?![\d:])')          # Matches individual SGPRs not in ranges
sgpr_range_pattern = re.compile(r's\[(\d+):(\d+)\]')                 # Matches SGPR ranges

# Patterns to detect specific instruction types
spill_pattern = re.compile(r'^scratch_store_dwordx?\d*')  # Detect scratch store spills
reload_pattern = re.compile(r'^scratch_load_dwordx?\d*')  # Detect scratch load reloads
global_load_pattern = re.compile(r'^global_load_.*')
global_store_pattern = re.compile(r'^global_store_.*')
v_mfma_pattern = re.compile(r'^v_mfma_.*')

def parse_register_usage(lines):
    block_registers = defaultdict(lambda: {
        'vgprs': set(),
        'sgprs': set(),
        'vgpr_instruction_types': defaultdict(set),  # key: vgpr number, value: set of instruction types
    })
    current_block = None

    for line in lines:
        line = line.strip()

        # Detect block labels (e.g., .LBB0_1:, .Ltmp0:, ; %bb.0:)
        label_match = re.match(r'^(\.\w+|; %bb\.\d+):', line)
        if label_match:
            current_block = label_match.group(1)
            continue

        # If we're in a block, find VGPR and SGPR usage
        if current_block:
            # Split the line on ';' to remove comments
            code = line.split(';')[0].strip()
            if not code:
                continue  # Empty or comment line

            tokens = code.split()
            if not tokens:
                continue  # No instruction

            instruction = tokens[0]

            # Determine instruction type
            if spill_pattern.match(instruction):
                instruction_type = 'scratch_store'
            elif reload_pattern.match(instruction):
                instruction_type = 'scratch_load'
            elif v_mfma_pattern.match(instruction):
                instruction_type = 'v_mfma'
            elif global_load_pattern.match(instruction):
                instruction_type = 'global_load'
            elif global_store_pattern.match(instruction):
                instruction_type = 'global_store'
            else:
                instruction_type = 'others'

            # Extract VGPRs used in the instruction
            vgprs_used = extract_vgprs_from_instruction(line)
            # Add VGPRs to the block's vgprs set and instruction types
            for vgpr in vgprs_used:
                block_registers[current_block]['vgprs'].add(vgpr)
                block_registers[current_block]['vgpr_instruction_types'][vgpr].add(instruction_type)

            # Find SGPR ranges
            sgpr_ranges = sgpr_range_pattern.findall(line)
            for sgpr_range in sgpr_ranges:
                start, end = int(sgpr_range[0]), int(sgpr_range[1])
                block_registers[current_block]['sgprs'].update(range(start, end + 1))

            # Remove SGPR ranges from the line to prevent matching numbers within ranges
            line_no_sgpr_ranges = sgpr_range_pattern.sub('', line)

            # Find individual SGPRs
            sgprs = sgpr_single_pattern.findall(line_no_sgpr_ranges)
            for sgpr in sgprs:
                block_registers[current_block]['sgprs'].add(int(sgpr))

    return block_registers

def extract_vgprs_from_instruction(line):
    # Extract VGPRs from the operands of the instruction
    # Split the line on ';' to remove comments
    code = line.split(';')[0]
    # Split code into tokens
    tokens = code.strip().split()
    if len(tokens) < 2:
        return set()
    # Get operands as a single string
    operands_str = ' '.join(tokens[1:])
    # Find VGPR ranges
    vgpr_ranges = vgpr_range_pattern.findall(operands_str)
    vgprs = set()
    for vgpr_range in vgpr_ranges:
        start, end = int(vgpr_range[0]), int(vgpr_range[1])
        vgprs.update(range(start, end + 1))
    # Remove VGPR ranges from the operands to prevent matching numbers within ranges
    operands_no_vgpr_ranges = vgpr_range_pattern.sub('', operands_str)
    # Now, find individual VGPRs
    vgprs.update(map(int, vgpr_single_pattern.findall(operands_no_vgpr_ranges)))
    return vgprs

def generate_call_tree_and_register_usage(assembly_file):
    # Read the entire assembly file into a list of lines
    with open(assembly_file, 'r') as file:
        all_lines = file.readlines()

    # Find the index of the last s_endpgm
    last_endpgm_index = None
    for idx, line in enumerate(all_lines):
        if 's_endpgm' in line:
            last_endpgm_index = idx

    if last_endpgm_index is None:
        print("No s_endpgm found in the assembly file.")
        return

    # Only process lines up to (and including) the last s_endpgm
    lines_to_process = all_lines[:last_endpgm_index + 1]

    # Initialize directed graph
    call_tree = nx.DiGraph()

    # Regex patterns to capture labels and jump/branch instructions
    label_pattern = re.compile(r'^(\.\w+|; %bb\.\d+):')
    jump_pattern = re.compile(r'(s_cbranch|s_branch)\s+(\.\w+|; %bb\.\d+)')

    # Initialize variables to store current block
    current_label = None

    # Build the call tree from the lines to process
    for line in lines_to_process:
        line = line.strip()

        # Check for labels (blocks like .Ltmp, .LBB, and %bb)
        label_match = label_pattern.match(line)
        if label_match:
            current_label = label_match.group(1)
            call_tree.add_node(current_label)
            continue

        # Check for jump/branch instructions
        jump_match = jump_pattern.search(line)
        if jump_match and current_label:
            target_label = jump_match.group(2)
            call_tree.add_edge(current_label, target_label)

    # Parse the register usage information for each block
    block_registers = parse_register_usage(lines_to_process)

    # Assign subsets (layers) for the hierarchical layout
    for node in call_tree.nodes():
        if node.startswith('.LBB'):
            call_tree.nodes[node]['subset'] = 1
        elif node.startswith('.Ltmp'):
            call_tree.nodes[node]['subset'] = 2
        elif node.startswith('; %bb'):
            call_tree.nodes[node]['subset'] = 0
        else:
            call_tree.nodes[node]['subset'] = 3

    # Generate a hierarchical tree layout from top to bottom
    pos = nx.multipartite_layout(call_tree, subset_key="subset")

    # Plot the call tree with register usage information
    plt.figure(figsize=(12, 10))

    # Draw nodes with colors for different node types
    lbb_nodes = [n for n in call_tree if n.startswith('.LBB')]
    ltmp_nodes = [n for n in call_tree if n.startswith('.Ltmp')]
    bb_nodes = [n for n in call_tree if n.startswith('; %bb')]
    other_nodes = [n for n in call_tree if n not in lbb_nodes and n not in ltmp_nodes and n not in bb_nodes]

    nx.draw_networkx_nodes(call_tree, pos, nodelist=lbb_nodes, node_color='skyblue', node_size=2000, label='.LBB Nodes')
    nx.draw_networkx_nodes(call_tree, pos, nodelist=ltmp_nodes, node_color='lightgreen', node_size=1500, label='.Ltmp Nodes')
    nx.draw_networkx_nodes(call_tree, pos, nodelist=bb_nodes, node_color='pink', node_size=1500, label='; %bb Nodes')
    nx.draw_networkx_nodes(call_tree, pos, nodelist=other_nodes, node_color='orange', node_size=1500, label='Other Nodes')

    # Draw edges
    nx.draw_networkx_edges(call_tree, pos, arrows=True)

    # Add labels with per-block register usage
    labels = {
        node: f"{node}\nVGPRs: {len(block_registers[node]['vgprs'])}, SGPRs: {len(block_registers[node]['sgprs'])}"
        for node in call_tree.nodes()
    }
    nx.draw_networkx_labels(call_tree, pos, labels, font_size=8, font_weight='bold')

    plt.legend()
    plt.title("Call Tree and Register Usage (Per Block) for AMDGCN Assembly Blocks")
    plt.show()

    # Optional: Print register usage per block
    print("Register usage per block:")
    for block, usage in block_registers.items():
        print(f"Block {block}:")
        print(f"  VGPRs: {sorted(usage['vgprs'])}")
        print(f"  SGPRs: {sorted(usage['sgprs'])}")
        print(f"  VGPR Instruction Types:")
        for vgpr, types in usage['vgpr_instruction_types'].items():
            print(f"    v{vgpr}: {', '.join(types)}")

    # Determine the calling sequence
    blocks_in_call_sequence = get_calling_sequence(call_tree)

    # Call the new function to plot VGPR usage per block with reordered blocks
    plot_vgpr_usage(block_registers, blocks_in_call_sequence)

def get_calling_sequence(call_tree):
    # Find nodes with in-degree zero (potential entry points)
    starting_nodes = [n for n, d in call_tree.in_degree() if d == 0]

    if not starting_nodes:
        print("No starting nodes found in the call tree.")
        return []

    # Use DFS traversal to get the calling sequence
    visited = set()
    calling_sequence = []

    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        calling_sequence.append(node)
        for neighbor in call_tree.successors(node):
            dfs(neighbor)

    # In case of multiple starting nodes, we traverse each
    for start_node in starting_nodes:
        dfs(start_node)

    return calling_sequence

def plot_vgpr_usage(block_registers, blocks_in_call_sequence):
    # Map blocks to indices based on the calling sequence
    blocks = blocks_in_call_sequence
    block_indices = {block: idx for idx, block in enumerate(blocks)}

    # Define colors for instruction types with priority
    instruction_colors = {
        'scratch_store': 'red',
        'scratch_load': 'green',
        'v_mfma': 'orange',
        'global_store': 'purple',
        'global_load': 'cyan',
        'others': 'blue',
    }

    # Define priority for instruction types
    instruction_priority = ['scratch_store', 'scratch_load', 'v_mfma', 'global_store', 'global_load', 'others']

    # Prepare data for plotting
    x_vals = []
    y_vals = []
    colors = []

    for block in blocks:
        idx = block_indices[block]
        vgprs = sorted(block_registers[block]['vgprs'])
        for vgpr in vgprs:
            x_vals.append(idx)
            y_vals.append(vgpr)
            instruction_types = block_registers[block]['vgpr_instruction_types'][vgpr]
            # Determine color based on highest priority instruction type
            color = 'blue'  # Default color
            for instr_type in instruction_priority:
                if instr_type in instruction_types:
                    color = instruction_colors[instr_type]
                    break
            colors.append(color)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.scatter(x_vals, y_vals, marker='s', s=20, c=colors)
    plt.xlabel('Block Index (Calling Sequence)')
    plt.ylabel('VGPR Register Index')
    plt.title('VGPR Usage per Block (Ordered by Calling Sequence)')
    plt.xticks(range(len(blocks)), blocks, rotation=90)
    plt.yticks(range(0, max(y_vals)+1, 8))  # Adjust the step as needed
    plt.grid(True)
    plt.tight_layout()

    # Add a legend
    import matplotlib.patches as mpatches
    legend_patches = [mpatches.Patch(color=color, label=instr_type.replace('_', ' ').title())
                      for instr_type, color in instruction_colors.items()]
    plt.legend(handles=legend_patches)

    plt.show()

# Call this function with the path to your assembly file
#generate_call_tree_and_register_usage('streamk_gemm.amdgcn')
generate_call_tree_and_register_usage('streamk_gemm_maskstore.amdgcn')
#generate_call_tree_and_register_usage('matmul_kernel_BM256_BN256_BK64_GM1_SK1_nW8_nS2_EU0_kP2_mfma16.amdgcn')

