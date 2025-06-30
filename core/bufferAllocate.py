import torch
import matplotlib.pyplot as plt
import networkx as nx
import re
import sympy as sp
import os
os.environ['GRB_LICENSE_FILE'] = '/app/Nanoflow-python/gurobi.lic'
import gurobipy as gp
from gurobipy import GRB
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
                assert next_wrapper.fullName in G.nodes, f"{next_wrapper.fullName} is not in the graph"
                G.add_edge(wrapper.fullName, next_wrapper.fullName)
                # print(f"add edge {wrapper.fullName} -> {next_wrapper.fullName}")
        self.full_graph = G
    
    def set_all_batchsize_by_linear_programming(self):
        # Create a new model
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
                wrapper.owner.batch_size = int(solution[variables[wrapper.fullName]])
            wrapper.batch_size = int(solution[variables[wrapper.fullName]])
            # print(f"set {wrapper.fullName} batch size to {wrapper.batch_size} with shape {wrapper.shape}")

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
            model = gp.Model("linear_program")
            model.setParam("OutputFlag", 0)
            variables = {}
            # print("component: ", comp)
            # Create a subgraph for the component:
            comp = self.full_graph.subgraph(comp)
            if nx.is_directed_acyclic_graph(comp):
                sorted_nodes = list(nx.topological_sort(comp))
            else:
                raise Exception("Component must be a DAG")
            
            collected_copy_ops = []
            collected_redist_ops = []
            wrappers = [data['wrapper'] for _, data in comp.nodes(data=True)]
            for wrapper in wrappers:
                variables[wrapper.fullName] = model.addVar(name=wrapper.fullName, vtype=GRB.INTEGER, lb=0)
                if wrapper.owner.isVirtual:
                    if wrapper.owner.isCopy and wrapper.owner not in collected_copy_ops:
                        collected_copy_ops.append(wrapper.owner)
                    elif wrapper.owner.isRedist and wrapper.owner not in collected_redist_ops:
                        collected_redist_ops.append(wrapper.owner)
            # print("collected_copy_ops: ", [op.name for op in collected_copy_ops])
            # print("collected_redist_ops: ", [op.name for op in collected_redist_ops])
            if len(collected_copy_ops) == 0 and len(collected_redist_ops) == 0:
                # print("No copy or redist operations found in the component.")
                shape = wrappers[0].shape
                dtype = wrappers[0].dtype
                whole_buffer = torch.zeros(shape, dtype=dtype).cuda(device_id)
                self.total_allocated += whole_buffer.numel() * whole_buffer.element_size()
                for wrapper in wrappers:
                    wrapper.set_whole_buffer(whole_buffer)
                    wrapper.set_tensor_offset(0)
                    # print(f"set {wrapper.fullName}, offset: 0")
                continue

            # print("There are copy or redist operations in the component.")
            for cp_op in collected_copy_ops:
                for input_wrapper in cp_op.inputs.values():
                    prev_nodes = comp.predecessors(input_wrapper.fullName)
                    # print(f"prev_nodes: {prev_nodes}")
                    for prev_node in prev_nodes:
                        prev_wrapper = comp.nodes[prev_node]['wrapper']
                        model.addConstr(variables[input_wrapper.fullName] == variables[prev_wrapper.fullName], name=f"copy_{input_wrapper.fullName}")
                    next_nodes = comp.successors(input_wrapper.fullName)
                    for next_node in next_nodes:
                        next_wrapper = comp.nodes[next_node]['wrapper']
                        model.addConstr(variables[input_wrapper.fullName] == variables[next_wrapper.fullName], name=f"copy_{input_wrapper.fullName}")
                for output_wrapper in cp_op.outputs.values():
                    next_nodes = comp.successors(output_wrapper.fullName)
                    for next_node in next_nodes:
                        next_wrapper = comp.nodes[next_node]['wrapper']
                        model.addConstr(variables[output_wrapper.fullName] == variables[next_wrapper.fullName], name=f"copy_{output_wrapper.fullName}")
            for rd_op in collected_redist_ops:
                input_wrappers = list(rd_op.inputs.values())
                output_wrappers = list(rd_op.outputs.values())
                model.addConstr(variables[input_wrappers[0].fullName] == variables[output_wrappers[0].fullName], name=f"redist_align_{input_wrappers[0].fullName}")
                for idx, input_wrapper in enumerate(input_wrappers):
                    prev_nodes = comp.predecessors(input_wrapper.fullName)
                    for prev_node in prev_nodes:
                        prev_wrapper = comp.nodes[prev_node]['wrapper']
                        model.addConstr(variables[input_wrapper.fullName] == variables[prev_wrapper.fullName], name=f"redist_{input_wrapper.fullName}")
                    if idx < rd_op.num_inputs - 1:
                        next_input_wrapper = input_wrappers[idx + 1]
                        model.addConstr(variables[input_wrapper.fullName] + input_wrapper.batch_size == variables[next_input_wrapper.fullName], name=f"redist_{input_wrapper.fullName}")
                for idx, output_wrapper in enumerate(output_wrappers):
                    next_nodes = comp.successors(output_wrapper.fullName)
                    for next_node in next_nodes:
                        next_wrapper = comp.nodes[next_node]['wrapper']
                        model.addConstr(variables[output_wrapper.fullName] == variables[next_wrapper.fullName], name=f"redist_{output_wrapper.fullName}")
                    if idx < rd_op.num_outputs - 1:
                        next_output_wrapper = output_wrappers[idx + 1]
                        model.addConstr(variables[output_wrapper.fullName] + output_wrapper.batch_size == variables[next_output_wrapper.fullName], name=f"redist_{output_wrapper.fullName}")
            
            model.setObjective(gp.quicksum(variables[wrapper.fullName] for wrapper in wrappers), GRB.MINIMIZE)
            model.optimize()

            if model.status == GRB.OPTIMAL:
                # print(f"Optimal solution found for component {comp}:")
                # find the maximum value in the solution
                allocated = int(max([variables[node].X + data["wrapper"].batch_size for node, data in comp.nodes(data=True)]))
                # print(f"Allocated size: {allocated}")
                wrapper_for_allocation = comp.nodes[sorted_nodes[0]]['wrapper']
                # print("wrappers[0]: ", wrapper_for_allocation.fullName)
                shape = (allocated, *wrapper_for_allocation.shape[1:])
                dtype = wrapper_for_allocation.dtype
                # print(f"Allocated shape: {shape}, dtype: {dtype}")
                # allocate the buffer
                whole_buffer = torch.empty(shape, dtype=dtype).cuda(device_id)
                self.total_allocated += whole_buffer.numel() * whole_buffer.element_size()
                # set the buffer for each wrapper
                for wrapper in wrappers:
                    wrapper.set_whole_buffer(whole_buffer)
                    wrapper.set_tensor_offset(int(variables[wrapper.fullName].X))
                    # print(f"set {wrapper.fullName}, offset: {int(variables[wrapper.fullName].X)}")
            else:
                print(f"No optimal solution found for component {comp}.")
                raise Exception("No optimal solution found for component!")
        print(f"Total allocated size: {self.total_allocated}")
        print("Allocation finished.")



    def allocate_buffer(self, device_id, plot = False):
        self.allocate_buffers_for_components(device_id)
        if plot:
            self.draw_dependency_graph()
            self.draw_dependency_subgraphs()
            # self.draw_allocation_subgraphs()
        return self.total_allocated
