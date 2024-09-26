import re
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_register_usage(assembly_file):
    # Regular expressions for VGPR, SGPR detection, and spill operations
    vgpr_single_pattern = re.compile(r'v(\d+)')  # Matches individual VGPRs
    vgpr_range_pattern = re.compile(r'v\[(\d+):(\d+)\]')  # Matches VGPR ranges
    sgpr_single_pattern = re.compile(r's(\d+)')  # Matches individual SGPRs
    spill_pattern = re.compile(r'scratch_store_dwordx(\d+)|scratch_store_dword')  # Detect scratch store spills
    
    block_registers = defaultdict(lambda: {'vgprs': set(), 'sgprs': set(), 'vgpr_spills': 0})
    current_block = None

    with open(assembly_file, 'r') as file:
        for line in file:
            line = line.strip()

            # Detect block labels (e.g., .LBB0_1:, .Ltmp0:, ; %bb.0:)
            label_match = re.match(r'^(\.\w+|; %bb\.\d+):', line)
            if label_match:
                current_block = label_match.group(1)
                continue
            
            # If we're in a block, find VGPR and SGPR usage
            if current_block:
                # Find individual VGPRs
                vgprs = vgpr_single_pattern.findall(line)
                for vgpr in vgprs:
                    block_registers[current_block]['vgprs'].add(int(vgpr))

                # Find VGPR ranges
                vgpr_ranges = vgpr_range_pattern.findall(line)
                for vgpr_range in vgpr_ranges:
                    start, end = int(vgpr_range[0]), int(vgpr_range[1])
                    block_registers[current_block]['vgprs'].update(range(start, end + 1))

                # Find individual SGPRs
                sgprs = sgpr_single_pattern.findall(line)
                for sgpr in sgprs:
                    block_registers[current_block]['sgprs'].add(int(sgpr))

                # Check for spill operations (scratch_store_dword) and count them
                spill_match = spill_pattern.search(line)
                if spill_match:
                    if 'x' in line:  # If it's a dwordxN spill
                        spill_size = int(spill_match.group(1))
                    else:  # If it's a single dword spill
                        spill_size = 1

                    # Count VGPRs spilled (check for ranges or single VGPRs)
                    vgpr_spilled = vgpr_single_pattern.findall(line)  # Find individual VGPRs spilled
                    vgpr_spilled_ranges = vgpr_range_pattern.findall(line)  # Find VGPR ranges spilled

                    # Add spills for individual VGPRs
                    block_registers[current_block]['vgpr_spills'] += len(vgpr_spilled) * spill_size

                    # Add spills for VGPR ranges
                    for vgpr_range in vgpr_spilled_ranges:
                        start, end = int(vgpr_range[0]), int(vgpr_range[1])
                        block_registers[current_block]['vgpr_spills'] += (end - start + 1) * spill_size

    return block_registers


def aggregate_register_usage(call_tree, block_registers):
    # Function to aggregate register usage for each block and its dependencies
    total_register_usage = {}

    def dfs(block, visited):
        if block in visited:
            return total_register_usage.get(block, {'vgprs': set(), 'sgprs': set(), 'vgpr_spills': 0})

        visited.add(block)
        vgprs = block_registers[block]['vgprs'].copy()
        sgprs = block_registers[block]['sgprs'].copy()
        vgpr_spills = block_registers[block]['vgpr_spills']

        # Traverse the dependencies (successors) and accumulate register usage
        for successor in call_tree.successors(block):
            usage = dfs(successor, visited)
            vgprs.update(usage['vgprs'])
            sgprs.update(usage['sgprs'])
            vgpr_spills += usage['vgpr_spills']

        total_register_usage[block] = {'vgprs': vgprs, 'sgprs': sgprs, 'vgpr_spills': vgpr_spills}
        return total_register_usage[block]

    # Perform a DFS for each node in the graph to aggregate register usage
    for block in call_tree.nodes():
        if block not in total_register_usage:
            dfs(block, set())

    return total_register_usage


def generate_call_tree_and_register_usage(assembly_file):
    # Initialize directed graph
    call_tree = nx.DiGraph()

    # Regex patterns to capture labels and jump/branch instructions
    label_pattern = re.compile(r'^(\.\w+|; %bb\.\d+):')
    jump_pattern = re.compile(r'(s_cbranch|s_branch)\s+(\.\w+|; %bb\.\d+)')

    # Initialize variables to store current block
    current_label = None
    end_program = False  # Flag to stop processing after s_endpgm

    # Read the assembly file and build the call tree
    with open(assembly_file, 'r') as file:
        for line in file:
            line = line.strip()

            # Stop processing after encountering s_endpgm
            if 's_endpgm' in line:
                end_program = True
                break

            # Check for labels (blocks like .Ltmp, .LBB, and %bb)
            label_match = label_pattern.match(line)
            if not end_program and label_match:
                current_label = label_match.group(1)
                call_tree.add_node(current_label)
                continue
            
            # Check for jump/branch instructions
            jump_match = jump_pattern.search(line)
            if not end_program and jump_match:
                target_label = jump_match.group(2)
                call_tree.add_edge(current_label, target_label)

    # Parse the register usage information for each block
    block_registers = parse_register_usage(assembly_file)

    # Aggregate register usage by traversing the call dependencies
    total_register_usage = aggregate_register_usage(call_tree, block_registers)

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

    # Draw nodes with colors for .LBB, .Ltmp, and %bb nodes
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

    # Add labels with total register usage and spills
    labels = {
        node: f"{node}\nVGPRs: {len(total_register_usage[node]['vgprs'])}, SGPRs: {len(total_register_usage[node]['sgprs'])}, Spills: {total_register_usage[node]['vgpr_spills']}"
        for node in call_tree.nodes()
    }
    nx.draw_networkx_labels(call_tree, pos, labels, font_size=8, font_weight='bold')

    plt.legend()
    plt.title("Call Tree and Register Usage (Including Spills) for AMDGCN Assembly Blocks")
    plt.show()


# Call this function with the path to your assembly file
generate_call_tree_and_register_usage('streamk_gemm.amdgcn')

