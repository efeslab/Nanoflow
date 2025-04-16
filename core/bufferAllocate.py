import torch
import matplotlib.pyplot as plt
import networkx as nx
import re
import sympy as sp
from core.IOWrapper import IOWrapper
from operations.virtualOp.virtual_ops import Copy, Copy_Device, Redist, Redist_Device
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
            # print(f"add node {wrapper.fullName}")
        for wrapper in self.buffers_list:
            for next_wrapper in wrapper.next:
                G.add_edge(wrapper.fullName, next_wrapper.fullName)
                # print(f"add edge {wrapper.fullName} -> {next_wrapper.fullName}")
        self.full_graph = G
    
    def set_all_batchsize_by_linear_programming(self):
        variables = {}
        equations = []
        for wrapper in self.buffers_list:
            # create a new variable for each wrapper
            variables[wrapper.fullName] = sp.symbols(wrapper.fullName)
            if wrapper.shape is not None:
                # assert wrapper.shape[0] > 0, f"{wrapper.fullName} has no shape"
                equations.append(sp.Eq(variables[wrapper.fullName], wrapper.shape[0]))
                # print(f"add equation {wrapper.fullName} = {wrapper.shape[0]}")
        
        # build the equations
        for wrapper in self.buffers_list:
        # if the wrapper is input, build the equation inside the op (Redist will only execute once)
            if wrapper.is_input_wrapper:
                if isinstance(wrapper.owner, Redist_Device) and len(wrapper.next) > 0:
                    # for Redist_Device, we need to build the equation for each input and output
                    input_symbols = [variables[input_wrapper.fullName] for input_wrapper in wrapper.owner.inputs.values()]
                    output_symbols = [variables[output_wrapper.fullName] for output_wrapper in wrapper.next]
                    # print(f"add equation {wrapper.fullName}: {input_symbols} = {output_symbols}")
                    equations.append(sp.Eq(sum(input_symbols), sum(output_symbols)))
                else:
                    # for the case of real op and Copy, the relationship is all the same buffer.
                    for output_wrapper in wrapper.owner.outputs.values():
                        # print(f"add equation {wrapper.fullName} = {output_wrapper.fullName}")
                        equations.append(sp.Eq(variables[wrapper.fullName], variables[output_wrapper.fullName]))

            # all links between the ops
            if wrapper.is_output_wrapper:
                assert len(wrapper.next) <= 1, f"{wrapper.fullName} has more than one next connections!\n"
                for next_wrapper in wrapper.next:
                    # assert wrapper.shape[1] == next_wrapper.shape[1], f"{wrapper.fullName} and {next_wrapper.fullName} has different shape"
                    # print(f"add equation {wrapper.fullName} = {next_wrapper.fullName}")
                    equations.append(sp.Eq(variables[wrapper.fullName], variables[next_wrapper.fullName]))

        # print(f"equations: {equations}")
        # Solve the linear programming problem
        solution = sp.solve(equations, variables)
        # print(f"solution: {solution}")
        assert len(solution) != 0, f"The solution space is empty, please check the batchsize setting!"
        assert len(solution) == len(variables), f"There are infinitely many solutions, please check the batchsize setting!"
        
        # Set the shape for each wrapper
        for wrapper in self.buffers_list:
            if wrapper.owner.batch_size is None:
                wrapper.owner.batch_size = solution[variables[wrapper.fullName]]
            wrapper.batch_size = solution[variables[wrapper.fullName]]
            print(f"set {wrapper.fullName} batch size to {wrapper.batch_size} with shape {wrapper.shape}")

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
    
    def allocate_buffers_for_components(self, device_id):
        self.total_allocated = 0
        components = self.get_connected_components()
        for comp in components:
            print("component: ", comp)
            # Create a subgraph for the component:
            comp = self.full_graph.subgraph(comp)
            
            if nx.is_directed_acyclic_graph(comp):
                sorted_nodes = list(nx.topological_sort(comp))
            else:
                raise Exception("Component must be a DAG")
            
            # Define a custom key function
            def sort_key(node):
                assert len(node.next) <= 1, f"Node {node.fullName} has more than one next node"
                next_node_name = node.next[0].fullName if node.next else ""
                # This regex captures a non-digit prefix and the subsequent numeric part.
                match = re.match(r'(\D+)(\d+)', next_node_name)
                if match:
                    prefix, num = match.groups()
                    return (prefix, int(num))
                # Fallback: if no match, return the original string and 0
                return (next_node_name, 0)

            root_nodes_name = [name for name, indeg in comp.in_degree() if indeg == 0]
            print(f"root_nodes_name: {root_nodes_name}")
            root_nodes = [self.full_graph.nodes[name]['wrapper'] for name in root_nodes_name]
            # sort these nodes by their next connections
            sorted_root_nodes = sorted(root_nodes, key=sort_key)
            # print(f"sorted_root_nodes: {[sorted_root.fullName for sorted_root in sorted_root_nodes]}")

            # allocate_info = []
            processing_queue = []

            cum_batchsize = 0
            # accumulate the first dimension of root nodes' shape
            for root_node in sorted_root_nodes:
                root_node.set_tensor_offset(cum_batchsize)
                cum_batchsize += root_node.shape[0]
                processing_queue.append(root_node)
            
            shape = (cum_batchsize, *sorted_root_nodes[0].shape[1:])
            dtype = sorted_root_nodes[0].dtype

            # print(f"total_size: {cum_batchsize}")
            # print(f"shape: {shape}")

            whole_buffer = torch.empty(shape, dtype=dtype, device=f"cuda:{device_id}")
            self.total_allocated += whole_buffer.numel() * whole_buffer.element_size()
            # print(f"allocated buffer: {whole_buffer.shape} with dtype: {dtype} and device: {device_id}")

            # allocate_info.append(shape)
            while processing_queue:
                node = processing_queue.pop(0)
                node.set_whole_buffer(whole_buffer)
                print("node: ", node.fullName, "with whole buffer: ", whole_buffer.shape, "tensor", node.tensor.shape,"and offset: ", node.tensor_offset)
                next_nodes = [self.full_graph.nodes[name]["wrapper"] for name in list(self.full_graph[node.fullName])]
                # print(f"next nodes: {[n for n in next_nodes]}")
                for next_node in next_nodes:
                    if isinstance(next_node.owner, Redist_Device):
                        next_node.set_whole_buffer(whole_buffer)
                        additional_offset = 0
                        flag = True
                        if next_node.is_input_wrapper:
                            for input_wrapper in next_node.owner.inputs.values():
                                if next_node.fullName == input_wrapper.fullName:
                                    next_node.set_tensor_offset(node.tensor_offset + additional_offset)
                                    if additional_offset != 0:
                                        flag = False
                                    break
                                additional_offset += input_wrapper.batch_size
                        elif next_node.is_output_wrapper:
                            for output_wrapper in next_node.owner.outputs.values():
                                if next_node.fullName == output_wrapper.fullName:
                                    next_node.set_tensor_offset(node.tensor_offset + additional_offset)
                                    break
                                # print("additional_offset: ", additional_offset, "output_wrapper: ", output_wrapper.fullName, "batch_size: ", output_wrapper.batch_size)
                                additional_offset += output_wrapper.batch_size

                        processing_queue.append(next_node) if flag else None

                    else:
                        next_node.set_whole_buffer(whole_buffer)
                        next_node.set_tensor_offset(node.tensor_offset)
                        assert node.batch_size == next_node.batch_size, f"Shape mismatch: {node.fullName} {node.batch_size} vs {next_node.batch_size}"
                        processing_queue.append(next_node)

    def allocate_buffer(self, device_id, plot = False):
        self.allocate_buffers_for_components(device_id)
        if plot:
            self.draw_dependency_graph()
            self.draw_dependency_subgraphs()
            # self.draw_allocation_subgraphs()
        return self.total_allocated