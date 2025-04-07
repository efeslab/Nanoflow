import torch
import matplotlib.pyplot as plt
import networkx as nx
from core.IOWrapper import IOWrapper, IOBufferType
from operations.virtualOp.copy import Copy
from operations.virtualOp.redist import Redist, RedistMode
from utils.graph_plot import plot_graph_topological, draw_graphs_subplots

class BufferAllocator():
    def __init__(self, buffers_list):
        self.buffers_list = buffers_list
        self.alloc_nodes = {}
        self.allocation_graph = []
        self.total_allocated = 0
        self.allocate_infos = [] # list[(shape, list[(base operator's name, offset)])]

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
    

    def handle_virtual_ops(self, component, device_id):
        wrappers = [self.full_graph.nodes[name]['wrapper'] for name in component]
        virtual_ops = []
        alloc_nodes = []

        # Find all virtual operations in component
        for wrapper in wrappers:
            if wrapper.owner.isVirtual:
                if not wrapper.prev[0].owner.isVirtual:
                    alloc_nodes.append(wrapper)
                else:
                    virtual_ops.append(wrapper.owner)

        # Only single allocation node per component
        if len(alloc_nodes) > 1:
            op_names = [op.name for op in alloc_nodes]
            raise Exception(
                f"Component contains multiple virtual operations connect to real operations output: {op_names}\n"
            )
        
        if not alloc_nodes:
            return False  
        
        allocate_info = []    

        alloc_node = alloc_nodes[0] 
        if isinstance(alloc_node.owner, Copy):
            shape = alloc_node.children[device_id].shape
            tensor = torch.empty(shape, dtype=alloc_node.dtype, device=f"cuda:{device_id}")
            self.total_allocated += tensor.numel() * tensor.element_size()
            alloc_node.children[device_id].tensor = tensor
                
            allocate_info.append(shape)
            allocate_info.append([alloc_node.children[device_id].owner.name, alloc_node.children[device_id].tensor_offset])

            # Share tensor with prev real nodes 
            for p in alloc_node.prev:
                p.children[device_id].tensor = tensor
                allocate_info.append([p.children[device_id].owner.name, p.children[device_id].tensor_offset])
           
            for n in alloc_node.next:
                # if n is virtual operations, it will be processed later
                if not n.owner.isVirtual:
                    n.children[device_id].tensor = alloc_node.children[device_id].tensor
                    allocate_info.append([n.children[device_id].owner.name, n.children[device_id].tensor_offset])

            
        elif isinstance(alloc_node.owner, Redist):
            if alloc_node.owner.mode == RedistMode.PARTITION:
                shape = alloc_node.children[device_id].shape
                tensor = torch.empty(shape, dtype=alloc_node.dtype, device=f"cuda:{device_id}")
                self.total_allocated += tensor.numel() * tensor.element_size()
                alloc_node.children[device_id].tensor = tensor

                allocate_info.append(shape)
                allocate_info.append([alloc_node.children[device_id].owner.name, alloc_node.children[device_id].tensor_offset])


                # Share tensor with prev real nodes 
                for p in alloc_node.prev:
                    p.children[device_id].tensor = tensor
                    allocate_info.append([p.children[device_id].owner.name, p.children[device_id].tensor_offset])

 
                # Split the tensor to its child    
                tensor = alloc_node.children[device_id].tensor
                size_for_child = [child.children[device_id].shape[0] for child in alloc_node.next]
                assert sum(size_for_child) == shape[0], f"shape mismatch: {alloc_node.fullName} {size_for_child} vs {shape}"
                tensor_split_list = tensor.split(size_for_child, dim=0)
                for idx, child, tensor_split in zip(range(len(alloc_node.next)), alloc_node.next, tensor_split_list):
                    child.children[device_id].tensor = tensor_split
                    child.children[device_id].tensor_offset = sum(size_for_child[:idx])
                    allocate_info.append([child.children[device_id].owner.name, child.children[device_id].tensor_offset])
                    assert child.children[device_id].shape == tensor_split.shape, f"shape mismatch: {child.fullName} {child.shape} vs {tensor_split.shape}"
                                
            elif alloc_node.owner.mode ==  RedistMode.AGGREGATE:
                shape = alloc_node.children[device_id].shape
                tensor = torch.empty(shape, dtype=alloc_node.dtype, device=f"cuda:{device_id}")
                self.total_allocated += tensor.numel() * tensor.element_size()
                alloc_node.children[device_id].tensor = tensor
                    
                allocate_info.append(shape)
                allocate_info.append([alloc_node.children[device_id].owner.name, alloc_node.children[device_id].tensor_offset])

                # Split to real prev nodes
                split_sizes = [p.children[device_id].shape[0] for p in alloc_node.prev]
                assert sum(split_sizes) == shape[0], f"shape mismatch: {alloc_node.fullName} {split_sizes} vs {shape}"
                tensors = torch.split(tensor, split_sizes, dim=0)
                for idx, p, t in zip(range(len(alloc_node.prev)), alloc_node.prev, tensors):
                    p.children[device_id].tensor = t
                    p.children[device_id].tensor_offset = sum(split_sizes[:idx])
                    allocate_info.append([p.children[device_id].owner.name, p.children[device_id].tensor_offset])
                    
                      
                for n in alloc_node.next:
                    # if n is virtual operations, it will be processed later
                    if not n.owner.isVirtual:
                        n.children[device_id].tensor = alloc_node.children[device_id].tensor
                        allocate_info.append([n.children[device_id].owner.name, n.children[device_id].tensor_offset])
            

        for virtual_op in virtual_ops:
            node = virtual_op.io
            if isinstance(virtual_op, Copy):
                # If its prev node is partition, its tensor should already shared before
                if not(isinstance(node.prev[0].owner , Redist) and RedistMode.PARTITION):
                    node.children[device_id].tensor = node.prev[0].children[device_id].tensor
                    node.children[device_id].tensor_offset = node.prev[0].children[device_id].tensor_offset   
                    allocate_info.append([node.children[device_id].owner.name, node.children[device_id].tensor_offset])

            
                for n in node.next:
                    # if n is virtual operations, it will be processed later
                    if not n.owner.isVirtual:
                        n.children[device_id].tensor = node.children[device_id].tensor
                        n.children[device_id].tensor_offset = node.children[device_id].tensor_offset
                        allocate_info.append([n.children[device_id].owner.name, n.children[device_id].tensor_offset])


            elif isinstance(virtual_op, Redist):
                if virtual_op.mode == RedistMode.PARTITION:
                    node.children[device_id].tensor = node.prev[0].children[device_id].tensor
                    node.children[device_id].tensor_offset = node.prev[0].children[device_id].tensor_offset   
                    allocate_info.append([node.children[device_id].owner.name, node.children[device_id].tensor_offset])


                    tensor = node.children[device_id].tensor
                    offset = node.children[device_id].tensor_offset
                    size_for_child = [child.children[device_id].shape[0] for child in node.next]
                    assert sum(size_for_child) == shape[0], f"shape mismatch: {node.io.fullName} {size_for_child} vs {shape}"
                    tensor_split_list = tensor.split(size_for_child, dim=0)
                    for idx, child, tensor_split in zip(range(len(node.next)), node.next, tensor_split_list):
                        child.children[device_id].tensor = tensor_split
                        child.children[device_id].tensor_offset = offset + sum(size_for_child[:idx])
                        allocate_info.append([child.children[device_id].owner.name, child.children[device_id].tensor_offset])
                        assert child.children[device_id].shape == tensor_split.shape, f"shape mismatch: {child.fullName} {child.shape} vs {tensor_split.shape}"
                    
                else:
                    # Aggregation from prev nodes
                    assert all(n.children[device_id].shape[1:] == node.prev[0].children[device_id].shape[1:] for n in node.prev), \
                        f"Shape mismatch among inputs to {node.fullName}"

                    # Sort prev nodes by tensor_offset to preserve order
                    sorted_prev = sorted(node.prev, key=lambda x: x.tensor_offset)
                    for i in range(len(sorted_prev) - 1):
                        current = sorted_prev[i]
                        next_node = sorted_prev[i + 1]
                        assert current.tensor_offset + current.tensor.children[device_id].shape[0] == next_node.tensor_offset, \
                            f"Invalid dependency between: {current.fullName} and {next_node.fullName}"

                    # Collect tensors and offsets
                    prev_tensors = [n.children[device_id].tensor for n in sorted_prev]
                    node.children[device_id].tensor = torch.cat(prev_tensors, dim=0)

                    # Set offset as the first input's offset (or min offset)
                    node.children[device_id].tensor_offset = sorted_prev[0].children[device_id].tensor_offset
                    allocate_info.append([node.children[device_id].owner.name, node.children[device_id].tensor_offset])

                    for n in node.next:
                        # if n is virtual operations, it will be processed later
                        if not n.owner.isVirtual:
                            n.children[device_id].tensor = node.children[device_id].tensor
                            n.children[device_id].tensor = node.children[device_id].tensor_offset
                            allocate_info.append([n.children[device_id].owner.name, n.children[device_id].tensor_offset])

        self.allocate_infos.append(allocate_info)
        return True
    
    def allocate_buffers_for_components(self, device_id):
        self.total_allocated = 0
        self.allocate_infos = []
        components = self.get_connected_components()
        for comp in components:
            # Create a subgraph for the component:
            comp = self.full_graph.subgraph(comp)

            if nx.is_directed_acyclic_graph(comp):
                sorted_nodes = list(nx.topological_sort(comp))
            else:
                raise Exception("Component must be a DAG")

            if self.handle_virtual_ops(sorted_nodes, device_id):
                continue
            
            # Allocation logic for non-virtual components
            wrappers = [self.full_graph.nodes[name]['wrapper'] for name in comp]
            semi_root_nodes = [w for w in wrappers]
            
            if not semi_root_nodes:
                continue

            allocate_info = []    

            alloc_node = semi_root_nodes[0]
            shape = alloc_node.children[device_id].shape
            tensor = torch.empty(alloc_node.children[device_id].shape, dtype=alloc_node.dtype, device=f"cuda:{device_id}")
            self.total_allocated += tensor.numel() * tensor.element_size()
            alloc_node.tensor = tensor
           
            allocate_info.append(shape)
            
            for semi_root in semi_root_nodes:
                semi_root.children[device_id].tensor = tensor
                allocate_info.append([semi_root.children[device_id].owner.name, semi_root.children[device_id].tensor_offset])
                assert semi_root.children[device_id].shape == alloc_node.children[device_id].shape, \
                    f"Shape mismatch: {semi_root.fullName} {semi_root.shape} vs {alloc_node.shape}"
            
            self.allocate_infos.append(allocate_info)

    def allocate_buffer(self, device_id, plot = False):
        self.allocate_buffers_for_components(device_id)
        # print(self.allocate_infos)
        if plot:
            self.draw_dependency_graph()
            self.draw_dependency_subgraphs()
            # self.draw_allocation_subgraphs()
        return self.total_allocated