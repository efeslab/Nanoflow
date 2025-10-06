import torch
import matplotlib.pyplot as plt
import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from nanoflow.operations.virtualOp.virtual_ops import Redist
from nanoflow.utils.graph_plot import plot_graph_topological, draw_graphs_subplots

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
            for next_wrapper in wrapper.actual_next:
                # print("next_wrapper:", next_wrapper.fullName)
                assert next_wrapper.fullName in G.nodes, f"{next_wrapper.fullName} is not in the graph"
                G.add_edge(wrapper.fullName, next_wrapper.fullName)
                # print(f"add edge {wrapper.fullName} -> {next_wrapper.fullName}")
        self.full_graph = G
    
    def set_all_batchsize_by_linear_programming(self):
        # Create a new model
        model = gp.Model("batchsize_lp")
        model.setParam("OutputFlag", 0)
        vars_by_wrap = {}
        for w in self.buffers_list:
            # create a new variable for each wrapper
            v = model.addVar(
                name = w.fullName,
                vtype = GRB.CONTINUOUS,
                lb    = 0,
            )
            vars_by_wrap[w] = v
            if w.shape is not None:
                # assert wrapper.shape[0] > 0, f"{wrapper.fullName} has no shape"
                model.addConstr(v == w.shape[0], name=f"fixed_{w.fullName}")
                # print(f"add equation {wrapper.fullName} = {wrapper.shape[0]}")
        
        # build the equations
        for w in self.buffers_list:
        # if the wrapper is input, build the equation inside the op (Redist will only execute once)
            assert w.is_input_wrapper or w.is_output_wrapper, f"{w.fullName} is not input or output wrapper"
            if w.is_input_wrapper:
                if isinstance(w.owner, Redist) and len(w.actual_next) > 0:
                    # for Redist_Device, we need to build the equation for each input and output
                    lhs = gp.quicksum(vars_by_wrap[iw] for iw in w.owner.inputs.values())
                    rhs = gp.quicksum(vars_by_wrap[ow] for ow in w.actual_next)
                    # print(f"add equation {wrapper.fullName}: {input_symbols} = {output_symbols}")
                    model.addConstr(lhs == rhs, name=f"redist_{w.fullName}")
                else:
                    # for the case of real op and Copy, the relationship is all the same buffer.
                    in_var = vars_by_wrap[w]
                    for ow in w.owner.outputs.values():
                        # print(f"add equation {wrapper.fullName} = {ow.fullName}")
                        model.addConstr(in_var == vars_by_wrap[ow], name=f"copy_{w.fullName}")

            # all links between the ops
            if w.is_output_wrapper and len(w.actual_next) > 0:
                assert len(w.actual_next) <= 1, f"{w.fullName} has more than one next connections!\n"
                next_var = vars_by_wrap[w.actual_next[0]]
                model.addConstr(vars_by_wrap[w] == next_var, name=f"link_{w.fullName}->{w.actual_next[0].fullName}")

        # Solve the linear programming problem
        model.setObjective(0.0)     # feasibility only
        model.optimize()
        # for v in model.getVars():
        #     print(f"Variable {v.VarName}: Lower Bound = {v.LB}, Upper Bound = {v.UB}")
        # for c in model.getConstrs():
        #     expr = model.getRow(c)
        #     print(f"Constraint {c.ConstrName}: {expr} <= {c.RHS}")
        if model.status != GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi returned status {model.status} "
                           "(infeasible or unbounded).")
        # print(f"solution: {solution}")
        
        # Set the shape for each wrapper
        for w, v in vars_by_wrap.items():
            bsz = int(v.X)
            if w.owner.batch_size is None:
                w.owner.batch_size = bsz
            w.batch_size = bsz
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
    
    def allocate_buffers_for_components(self, device):
        self.total_allocated = 0
        components = self.get_connected_components()
        for comp in components:
            model = gp.Model("linear_program")
            model.setParam("OutputFlag", 0)
            variables = {}
            print("component: ", comp)
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
                variables[wrapper.fullName] = model.addVar(name=wrapper.fullName, vtype=GRB.CONTINUOUS, lb=0)
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
                print(f"Allocated shape: {shape}, dtype: {dtype}")
                whole_buffer = torch.empty(shape, dtype=dtype).to(device)
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
                whole_buffer = torch.empty(shape, dtype=dtype).to(device)
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



    def allocate_buffer(self, device, plot = False):
        self.allocate_buffers_for_components(device)
        if plot:
            self.draw_dependency_graph()
            self.draw_dependency_subgraphs()
            # self.draw_allocation_subgraphs()
        return self.total_allocated
