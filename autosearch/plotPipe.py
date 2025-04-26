import matplotlib.pyplot as plt
import networkx as nx
from operation import NanoOperation, Operation, LayeredNanoOperation
# Create a directed graph from dependencies

def getNodeName(nano_op, layer):
    return f"{nano_op.name}_{layer}"

def createGraph(nano_ops:list[NanoOperation], layer):
    G = nx.DiGraph()
    # Add nodes with durations and categories as attributes
    for nano_op in nano_ops:
        for i in range(0, layer):
            G.add_node(getNodeName(nano_op, i), nano_op = nano_op)
    for nano_op in nano_ops:
        for i in range(0, layer):
            for dep_nano_op in nano_op.depend_on_nano_ops:
                if dep_nano_op.depend_on_previous_layer:
                    if i > 0:
                        G.add_edge(getNodeName(dep_nano_op.operation, i-1), getNodeName(nano_op, i))
                else:
                    G.add_edge(getNodeName(dep_nano_op.operation, i), getNodeName(nano_op, i))
    return G

def createLogicalDependencyGraph(ops:list[Operation]):
    G = nx.DiGraph()
    # Add nodes with durations and categories as attributes
    for op in ops:
        G.add_node(op.name, operation = op)
    for op in ops:
        for dep_op in op.depend_on_operations:
            if not dep_op.depend_on_previous_layer:
                G.add_edge(dep_op.operation.name, op.name)
    return G

def createLogicalDependencyGraphOfNanoOps(nano_ops:list[NanoOperation]):
    G = nx.DiGraph()
    # Add nodes with durations and categories as attributes
    for nano_op in nano_ops:
        G.add_node(nano_op.name, nano_op = nano_op)
    for nano_op in nano_ops:
        for dep_nano_op in nano_op.logical_dependency:
            if not dep_nano_op.depend_on_previous_layer:
                G.add_edge(dep_nano_op.operation.name, nano_op.name)
    return G

def createLogicalDependencyGraphOfLayeredNanoOps(layer_nano_ops:list[LayeredNanoOperation]):
    G = nx.DiGraph()
    # Add nodes with durations and categories as attributes
    for layer_nano_op in layer_nano_ops:
        G.add_node(layer_nano_op.name, layer_nano_op = layer_nano_op)
    for layer_nano_op in layer_nano_ops:
        for dep_nano_op in layer_nano_op.logical_dependency:
            G.add_edge(dep_nano_op.operation.name, layer_nano_op.name)
    return G
def createGraphOfLayeredNanoOps(layer_nano_ops:list[LayeredNanoOperation]):
    G = nx.DiGraph()
    # Add nodes with durations and categories as attributes
    for layer_nano_op in layer_nano_ops:
        G.add_node(layer_nano_op.name, layer_nano_op = layer_nano_op)
    for layer_nano_op in layer_nano_ops:
        for dep_nano_op in layer_nano_op.depend_on_layer_nano_ops:
            G.add_edge(dep_nano_op.operation.name, layer_nano_op.name)
    return G

def calcStartEndLayered(G: nx.DiGraph):
    for node in nx.topological_sort(G):
        if G.in_degree(node) > 0: 
            start_time = max([G.nodes[predecessor]["finish_time"] for predecessor in G.predecessors(node)])
            G.nodes[node]["start_time"] = start_time
            G.nodes[node]["finish_time"] = start_time + G.nodes[node]['layer_nano_op'].duration
        else:
            G.nodes[node]["start_time"] = 0
            G.nodes[node]["finish_time"] = G.nodes[node]['layer_nano_op'].duration
    return G

def calcStartEnd(G: nx.DiGraph):
    for node in nx.topological_sort(G):
        if G.in_degree(node) > 0: 
            start_time = max([G.nodes[predecessor]["finish_time"] for predecessor in G.predecessors(node)])
            G.nodes[node]["start_time"] = start_time
            G.nodes[node]["finish_time"] = start_time + G.nodes[node]['nano_op'].duration
        else:
            G.nodes[node]["start_time"] = 0
            G.nodes[node]["finish_time"] = G.nodes[node]['nano_op'].duration
    return G

def getCycleTime(G):
    max_finish_time = max([G.nodes[node]["finish_time"] for node in G.nodes])
    return max_finish_time

def drawPipeline(G, filename=None):
    # First, calculate start and end times for each node
    G = calcStartEnd(G)
    # Compute the total cycle time based on node finish times
    cycleTime = getCycleTime(G)
    # Create figure and axis for plotting
    fig, ax = plt.subplots(figsize=(int(cycleTime), 6))

    # Extract all categories from the operations associated with nodes in G
    categories = {G.nodes[node]['nano_op'].operation.category for node in G.nodes}
    # Map each category to a vertical position index
    category_positions = {category: i for i, category in enumerate(sorted(categories))}

    # Plot a horizontal bar for each node representing its execution time
    for node in G.nodes:
        operation = G.nodes[node]['nano_op'].operation
        nano_op = G.nodes[node]['nano_op']
        category_position = category_positions[operation.category]
        start_time = G.nodes[node]['start_time']
        finish_time = G.nodes[node]['finish_time']
        duration = finish_time - start_time
        # Plot a horizontal bar from start_time to finish_time for this node
        ax.barh(category_position, duration, left=start_time, edgecolor='white', height=0.4,
                color=plt.cm.tab10(category_position % 10))

        # Annotate the bar with the node's name and the operation's duration
        tempText = f"{node}\n{int(nano_op.duration * 1000)}us"
        ax.text(start_time + duration / 2, category_position, tempText, ha='center', va='center', color='black',
                fontweight='bold')

    # Draw arrows to represent dependencies between nodes (edges in the graph)
    for dep_node, node in G.edges:
        dep_finish = G.nodes[dep_node]['finish_time']
        node_start = G.nodes[node]['start_time']
        dep_category_position = category_positions[G.nodes[dep_node]['nano_op'].operation.category]
        node_category_position = category_positions[G.nodes[node]['nano_op'].operation.category]
        # Draw an arrow from the end time of the dependency node to the start time of the current node
        ax.annotate('', xy=(node_start, node_category_position), xytext=(dep_finish, dep_category_position),
                    arrowprops=dict(arrowstyle="->", lw=2, color='black'))

    # Set labels for axes
    ax.set_xlabel('Time (units)')
    ax.set_ylabel('Category')
    # Set the y-axis ticks and labels for each category
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(sorted(categories))
    ax.set_title(f'Schedule Timeline (Cycle Time = {cycleTime} units)')
    fig.tight_layout()

    # Create a legend for categories based on the plotted bars
    from matplotlib.patches import Patch
    category_patches = []
    for category in sorted(categories):
        color_index = category_positions[category] % 10
        category_patches.append(Patch(color=plt.cm.tab10(color_index), label=category))
    ax.legend(handles=category_patches, title="Operation Categories")

    # Save the figure to a file
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close(fig)


def drawPipelineLayered(G, filename=None):
    # First, calculate start and end times for each node
    G = calcStartEndLayered(G)
    # Compute the total cycle time based on node finish times
    cycleTime = getCycleTime(G)
    # Create figure and axis for plotting
    fig, ax = plt.subplots(figsize=(int(cycleTime), 6))

    # Extract all categories from the operations associated with nodes in G
    categories = {G.nodes[node]['layer_nano_op'].nano_op.operation.category for node in G.nodes}
    # Map each category to a vertical position index
    category_positions = {category: i for i, category in enumerate(sorted(categories))}

    # Plot a horizontal bar for each node representing its execution time
    for node in G.nodes:
        operation = G.nodes[node]['layer_nano_op'].nano_op.operation
        layer_nano_op = G.nodes[node]['layer_nano_op']
        category_position = category_positions[operation.category]
        start_time = G.nodes[node]['start_time']
        finish_time = G.nodes[node]['finish_time']
        duration = finish_time - start_time
        # Plot a horizontal bar from start_time to finish_time for this node
        ax.barh(category_position, duration, left=start_time, edgecolor='white', height=0.4,
                color=plt.cm.tab10(category_position % 10))

        # Annotate the bar with the node's name and the operation's duration
        tempText = f"{node}\n{int(layer_nano_op.duration * 1000)}us"
        ax.text(start_time + duration / 2, category_position, tempText, ha='center', va='center', color='black',
                fontweight='bold')

    # Draw arrows to represent dependencies between nodes (edges in the graph)
    for dep_node, node in G.edges:
        dep_finish = G.nodes[dep_node]['finish_time']
        node_start = G.nodes[node]['start_time']
        dep_category_position = category_positions[G.nodes[dep_node]['layer_nano_op'].nano_op.operation.category]
        node_category_position = category_positions[G.nodes[node]['layer_nano_op'].nano_op.operation.category]
        # Draw an arrow from the end time of the dependency node to the start time of the current node
        ax.annotate('', xy=(node_start, node_category_position), xytext=(dep_finish, dep_category_position),
                    arrowprops=dict(arrowstyle="->", lw=2, color='black'))

    # Set labels for axes
    ax.set_xlabel('Time (units)')
    ax.set_ylabel('Category')
    # Set the y-axis ticks and labels for each category
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(sorted(categories))
    ax.set_title(f'Schedule Timeline (Cycle Time = {cycleTime} units)')
    fig.tight_layout()

    # Create a legend for categories based on the plotted bars
    from matplotlib.patches import Patch
    category_patches = []
    for category in sorted(categories):
        color_index = category_positions[category] % 10
        category_patches.append(Patch(color=plt.cm.tab10(color_index), label=category))
    ax.legend(handles=category_patches, title="Operation Categories")

    # Save the figure to a file
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close(fig)
