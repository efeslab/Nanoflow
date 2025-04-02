import torch
import matplotlib.pyplot as plt
import networkx as nx
from core.IOWrapper import IOWrapper, IOBufferType
from operations.virtualOp.copy import Copy
from operations.virtualOp.redist import Redist
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
    
    def draw_allocation_subgraphs(self):
        draw_graphs_subplots(self.allocation_graph, title_prefix="Allocation")
    

    def handle_virtual_ops(self, component):
        wrappers = [self.full_graph.nodes[name]['wrapper'] for name in component]
        virtual_ops = []
        isInput = False
    
        # Find all virtual operations in component
        for wrapper in wrappers:
            if isinstance(wrapper.owner, (Copy, Redist)):
                if wrapper.name == "input":
                    isInput = True
                virtual_ops.append(wrapper.owner)
    
        # Only single virtual op per component
        if len(virtual_ops) > 1:
            op_names = [op.name for op in virtual_ops]
            raise RuntimeError(
                f"Component contains multiple virtual operations: {op_names}\n"
                f"Each connected component can only contain one virtual operation (Copy/Redist)"
            )
        
        if not virtual_ops:
            return False  
        
        virtual_op = virtual_ops[0] 
        G_H = nx.DiGraph()
        if isinstance(virtual_op, Copy):
            if isInput == True:
                shape = virtual_op.inputs["input"].shape
                tensor = torch.empty(shape, dtype=virtual_op.inputs["input"].dtype, device='cuda')
                self.total_allocated += tensor.numel() * tensor.element_size()
                virtual_op.inputs["input"].tensor = tensor
                virtual_op.outputs["output"].tensor = tensor
                
                G_H.add_node(virtual_op.inputs["input"].fullName, wrapper=virtual_op.inputs["input"])
                # Share tensor with prev nodes 
                for p in virtual_op.inputs["input"].prev:
                    p.tensor = tensor
                    G_H.add_edge(virtual_op.inputs["input"].fullName, p.fullName)
            else:
                G_H.add_node(virtual_op.outputs["output"].fullName, wrapper=virtual_op.outputs["output"])
                # Copy to outputs
                for n in virtual_op.outputs["output"].next:
                    n.tensor = virtual_op.outputs["output"].tensor
                    G_H.add_edge(virtual_op.outputs["output"].fullName, n.fullName)

            self.allocation_graph.append(G_H)
            return True
            
        elif isinstance(virtual_op, Redist):
            if virtual_op.mode == "partition":
                # Partition input case, allocate and share to its prev
                if isInput == True:
                    shape = virtual_op.inputs["input"].shape
                    tensor = torch.empty(shape, dtype=virtual_op.inputs["input"].dtype, device='cuda')
                    self.total_allocated += tensor.numel() * tensor.element_size()
                    virtual_op.inputs["input"].tensor = tensor
                    virtual_op.outputs["output"].tensor = tensor

                    G_H.add_node(virtual_op.inputs["input"].fullName, wrapper=virtual_op.inputs["input"])
                    for p in virtual_op.inputs["input"].prev:
                        p.tensor = tensor
                        G_H.add_edge(virtual_op.inputs["input"].fullName, p.fullName)
 
                else:
                    # Partition output case, split the tensor to its output
                    node = virtual_op.outputs["output"]
                    shape = node.shape
                    tensor = node.tensor
                    G_H.add_node(virtual_op.outputs["output"].fullName, wrapper=node)
                    size_for_child = [child.shape[0] for child in node.next]
                    assert sum(size_for_child) == shape[0], f"shape mismatch: {node.fullName} {size_for_child} vs {shape}"
                    tensor_split_list = tensor.split(size_for_child, dim=0)
                    for idx, child, tensor_split in zip(range(len(node.next)), node.next, tensor_split_list):
                        child.tensor = tensor_split
                        child.tensor_offset = sum(size_for_child[:idx])
                        assert child.shape == tensor_split.shape, f"shape mismatch: {child.fullName} {child.shape} vs {tensor_split.shape}"
                        G_H.add_edge(virtual_op.outputs["output"].fullName, child.fullName)
        
            elif virtual_op.mode == "aggregate":
                # Aggregate input case, allocate new memory and split to its inputs
                if isInput == True:
                    shape = virtual_op.outputs["output"].shape
                    tensor = torch.empty(shape, dtype=virtual_op.outputs["output"].dtype, device='cuda')
                    self.total_allocated += tensor.numel() * tensor.element_size()
                    virtual_op.inputs["input"].tensor = tensor
                    virtual_op.outputs["output"].tensor = tensor
                    
                    G_H.add_node(virtual_op.inputs["input"].fullName, wrapper=virtual_op.inputs["input"])
                    
                    node = virtual_op.inputs["input"] 
                    # Split to input nodes
                    split_sizes = [p.shape[0] for p in virtual_op.inputs["input"].prev]
                    tensors = torch.split(tensor, split_sizes, dim=0)
                    for p, t in zip(virtual_op.inputs["input"].prev, tensors):
                        p.tensor = t
                      
                        G_H.add_edge(virtual_op.inputs["input"].fullName, p.fullName)

                else:
                    G_H.add_node(virtual_op.outputs["output"].fullName, wrapper=virtual_op.outputs["output"])
                    # Aggregate output case, share to its output
                    for n in virtual_op.outputs["output"].next:
                        n.tensor = virtual_op.outputs["output"].tensor
                        G_H.add_edge(virtual_op.outputs["output"].fullName, n.fullName)
            
            self.allocation_graph.append(G_H)
            return True
            
        return False
    
    def allocate_buffers_for_components(self):
        self.total_allocated = 0
        components = self.get_connected_components()
        for comp in components:
            if self.handle_virtual_ops(comp):
                continue
            
            # Allocation logic for non-virtual components
            wrappers = [self.full_graph.nodes[name]['wrapper'] for name in comp]
            semi_root_nodes = [w for w in wrappers]
            
            if not semi_root_nodes:
                continue
                
            alloc_node = IOWrapper("Alloc", semi_root_nodes[0].fullName, IOBufferType.FULL, dtype=semi_root_nodes[0].dtype)
            alloc_node.shape = semi_root_nodes[0].shape
            alloc_node.child = semi_root_nodes
            
            tensor = torch.empty(alloc_node.shape, dtype=alloc_node.dtype, device='cuda')
            self.total_allocated += tensor.numel() * tensor.element_size()
            alloc_node.tensor = tensor
            
            for semi_root in semi_root_nodes:
                semi_root.tensor = tensor
                assert semi_root.shape == alloc_node.shape, \
                    f"Shape mismatch: {semi_root.fullName} {semi_root.shape} vs {alloc_node.shape}"
            
            G_H = nx.DiGraph()
            G_H.add_node(alloc_node.fullName, wrapper=alloc_node)
            for node in semi_root_nodes:
                G_H.add_edge(alloc_node.fullName, node.fullName)
            self.allocation_graph.append(G_H)
    
    def allocate_buffer(self, plot = False):
        self.create_dependency_graph()
        self.allocate_buffers_for_components()
        if plot:
            self.draw_dependency_graph()
            self.draw_dependency_subgraphs()
            self.draw_allocation_subgraphs()
        return self.total_allocated