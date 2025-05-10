import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import networkx as nx
def draw_graphs_subplots(graphs, title_prefix):
    """Draw each graph from a list of NetworkX graphs on subplots with bounding boxes."""
    num_graphs = len(graphs)
    cols = math.ceil(math.sqrt(num_graphs))
    rows = math.ceil(num_graphs / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    
    # Flatten axes so we can index by a single index.
    if num_graphs == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, graph in enumerate(graphs):
        ax = axes[i]
        ax.set_title(f"{title_prefix} {i}")
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, with_labels=True, font_weight='bold', ax=ax)
        # Compute bounding box based on node positions.
        x_vals = [p[0] for p in pos.values()]
        y_vals = [p[1] for p in pos.values()]
        if x_vals and y_vals:
            x_min, x_max = min(x_vals), max(x_vals)
            y_min, y_max = min(y_vals), max(y_vals)
            margin_x = 0.1 * (x_max - x_min) if (x_max - x_min) != 0 else 0.1
            margin_y = 0.1 * (y_max - y_min) if (y_max - y_min) != 0 else 0.1
            rect = patches.Rectangle(
                (x_min - margin_x, y_min - margin_y),
                (x_max - x_min) + 2 * margin_x,
                (y_max - y_min) + 2 * margin_y,
                edgecolor='red',
                facecolor='none',
                lw=2
            )
            ax.add_patch(rect)

    # Turn off unused axes.
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()
def plot_graph_topological(G):
    # Compute the topological order
    topological_order = list(nx.topological_sort(G))
    
    # Assign x-coordinates based on topological depth
    depth_map = {}
    for node in topological_order:
        depth_map[node] = max((depth_map.get(pred, -1) for pred in G.predecessors(node)), default=-1) + 1
    
    # Group nodes by depth
    depth_levels = {}
    for node, depth in depth_map.items():
        depth_levels.setdefault(depth, []).append(node)
    
    max_depth = max(depth_map.values())
    
    # Assign positions
    pos = {}
    max_node_at_depth = max(len(nodes) for nodes in depth_levels.values())
    for depth, nodes in depth_levels.items():
        y_offset = -(len(nodes) - 1) / 2  # Center nodes vertically
        for i, node in enumerate(nodes):
            pos[node] = (depth, y_offset + i)
    
    # Create label positions by shifting each label upward
    label_pos = {node: (x, y + 0.2 * (-1)**(x)) for node, (x, y) in pos.items()}
    
    # set y limit to be 1.2x the number of nodes
    
    # Plot the graph
    plt.figure(figsize=( 5 + 1 * max_depth, 2 + max_node_at_depth))
    nx.draw(
        G, pos, with_labels=False, node_size=1000, node_color="skyblue", font_size=10, font_color="black"
    )
    plt.ylim(-max_node_at_depth*0.6, max_node_at_depth * 0.6)
    nx.draw_networkx_labels(G, label_pos, font_size=10, font_color="black")
    plt.title("Graph Layout by Topological Depth", fontsize=16)
    plt.show()
    