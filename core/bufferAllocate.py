import torch
import matplotlib.pyplot as plt
import networkx as nx
from core.IOWrapper import IOWrapper, IOBufferType
from utils.graph_plot import plot_graph_topological, draw_graphs_subplots

class BufferAllocator():
    def __init__(self, buffers_list):
        self.buffers_list = buffers_list
        self.alloc_nodes = {}
        self.allocation_graph = []
        self.total_allocated = 0

    def check_buffer_name_and_size(self):
        # Assert fullName is unique.
        fullName_list = [wrapper.fullName for wrapper in self.buffers_list]
        assert len(fullName_list) == len(set(fullName_list))

        for wrapper in self.buffers_list:
            assert wrapper.shape is not None, f"{wrapper.fullName} has no shape"
            wrapper.checkCorrectPartition()
    
    def create_dependency_graph(self):
        # Build full graph G.
        G = nx.DiGraph()
        for wrapper in self.buffers_list:
            G.add_node(wrapper.fullName, wrapper=wrapper)
        for wrapper in self.buffers_list:
            for next_wrapper in wrapper.next:
                G.add_edge(wrapper.fullName, next_wrapper.fullName)
        self.full_graph = G

    def draw_dependency_graph(self):
        # Draw full graph.
        plot_graph_topological(self.full_graph)
        plt.show()

    def get_connected_components(self):
        # Get connected components.
        components = list(nx.weakly_connected_components(self.full_graph))
        return components
    
    def draw_dependency_subgraphs(self):
        # Build list of subgraphs from G.
        G_subgraphs = [self.full_graph.subgraph(comp) for comp in self.get_connected_components()]
        draw_graphs_subplots(G_subgraphs, title_prefix="Buffer")

    def create_allocation_graph_from_component(self):
        components = self.get_connected_components()
        for component in components:
            wrappers = [self.full_graph.nodes[name]['wrapper'] for name in component]
            G_H = nx.DiGraph()
            for wrapper in wrappers:
                G_H.add_node(wrapper.fullName, wrapper=wrapper)
            # Use 'child' attribute to get connections.
            # If child does not exist, adjust to the correct attribute.
            for wrapper in wrappers:
                for child in wrapper.child:
                    G_H.add_edge(wrapper.fullName, child.fullName)
            
            # find all the nodes that have no incoming edges
            semi_root_nodes = [G_H.nodes[node]['wrapper'] for node in G_H.nodes if G_H.in_degree(node) == 0]
            # create a new node 
            alloc_node = IOWrapper("Alloc", semi_root_nodes[0].fullName, IOBufferType.FULL, dtype=semi_root_nodes[0].dtype)
            alloc_node.shape = semi_root_nodes[0].shape
            alloc_node.child = semi_root_nodes
                
            self.alloc_nodes[frozenset(component)] = alloc_node
            
            # # also add the new node to the graph
            G_H.add_node(alloc_node.fullName, wrapper=alloc_node)
            for node in semi_root_nodes:
                G_H.add_edge(alloc_node.fullName, node.fullName)
            self.allocation_graph.append(G_H)
    
    def draw_allocation_subgraphs(self):
        draw_graphs_subplots(self.allocation_graph, title_prefix="Allocation")
    
    def allocate_buffers_for_components(self):
        self.total_allocated = 0
        for comp, alloc_node in self.alloc_nodes.items():
            shape = alloc_node.shape
            tensor = torch.empty(shape, dtype=alloc_node.dtype, device='cuda')
            self.total_allocated += tensor.numel() * tensor.element_size()
            alloc_node.tensor = tensor
            for semi_root in alloc_node.child:
                semi_root.tensor = tensor
                assert semi_root.shape == shape, f"shape mismatch: {semi_root.fullName} {semi_root.shape} vs {shape}"
                # if have child then further propagate
                if semi_root.child:
                    size_for_child = [child.shape[0] for child in semi_root.child]
                    assert sum(size_for_child) == shape[0], f"shape mismatch: {semi_root.fullName} {size_for_child} vs {shape}"
                    tensor_split_list = tensor.split(size_for_child, dim=0)
                    for idx, child, tensor_split in zip(range(len(semi_root.child)), semi_root.child, tensor_split_list):
                        child.tensor = tensor_split
                        child.tensor_offset = sum(size_for_child[:idx])
                        assert child.shape == tensor_split.shape, f"shape mismatch: {child.fullName} {child.shape} vs {tensor_split.shape}"
    
    def allocate_buffer(self, plot = False):
        self.check_buffer_name_and_size()
        self.create_dependency_graph()
        self.create_allocation_graph_from_component()
        self.allocate_buffers_for_components()
        if plot:
            self.draw_dependency_graph()
            self.draw_dependency_subgraphs()
            self.draw_allocation_subgraphs()
        return self.total_allocated