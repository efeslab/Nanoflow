import torch
import matplotlib.pyplot as plt
import networkx as nx
from utils.prof_marker import prof_marker
from utils.graph_plot import plot_graph_topological, draw_graphs_subplots

tensor_pool = dict()
tensor_list = []

class Executor():
    def __init__(self, operations_layers_list, layer_list):
        self.operations_layers_list = [op_layer for op_layer in operations_layers_list if op_layer.batch_size > 0]
        # self.operations_layers_list = operations_layers_list
        self.layer_list = layer_list
        self.ordered_operations = []
    
    def not_this_layer(self, op, layer):
        return (op.first_layer_only and layer != 0) or (op.last_layer_only and layer != self.layer_list[-1])
    
    
    def plan_layer_ordering(self):
        # print("plan_layer_ordering")
        G = nx.DiGraph()
        for op in self.operations_layers_list:
            G.add_node(f"{op.name}", op=op, layer = op.layer)
            op.reset_op_cuda_status()

        for op in self.operations_layers_list:
            layer = op.layer
            # print("op.name", op.name) if layer == 0 else None
            for dep, dep_on_prev_layer, dep_on_next_layer in op.prerequisites:
                # print("dep", dep.name) if layer == 0 else None
                # print("dep_on_prev_layer", dep_on_prev_layer) if layer == 0 else None
                if (self.not_this_layer(dep, layer)):
                    continue
                if dep_on_prev_layer:
                    if layer > 0:
                        prev_op_name = f"{dep.name}_{layer - 1}"
                        # check if the previous layer op exists
                        if G.has_node(prev_op_name):
                            G.add_edge(prev_op_name, f"{op.name}")
                            op.append_prev_op_layer(G.nodes[prev_op_name]['op'])
                            G.nodes[prev_op_name]['op'].set_is_depended_on(op)
                            # print("op.name", op.name, "prev_op_name", prev_op_name)
                elif dep_on_next_layer:
                    if layer < self.layer_list[-1]:
                        prev_op_name = f"{dep.name}_{layer + 1}"
                        # check if the next layer op exists
                        if G.has_node(prev_op_name):
                            G.add_edge(prev_op_name, f"{op.name}")
                            op.append_prev_op_layer(G.nodes[prev_op_name]['op'])
                            G.nodes[prev_op_name]['op'].set_is_depended_on(op)
                            # print("op.name", op.name, "prev_op_name", prev_op_name)
                else:
                    prev_op_name = f"{dep.name}_{layer}"
                    if G.has_node(prev_op_name):
                        G.add_edge(prev_op_name, f"{op.name}")
                        op.append_prev_op_layer(G.nodes[prev_op_name]['op'])
                        G.nodes[prev_op_name]['op'].set_is_depended_on(op)
                        # print("op.name", op.name, "prev_op_name", prev_op_name)

        self.ordered_operations = list(nx.topological_sort(G))
        # print(self.ordered_operations)
        self.ordered_graph = G
    
    def draw_ordered_graph(self):
        plot_graph_topological(self.ordered_graph)
    
    def execute(self, weight_map, output):
        for op_name in self.ordered_operations:
            op = self.ordered_graph.nodes[op_name]['op']
            with prof_marker(f"{op.name}"):
                op.wait_cuda_event()
                op.run()
                op.record_cuda_event()
            if "GlobalOutput" in op.name:
                torch.cuda.synchronize()
                output.copy_(op.inputs["tokens"].tensor)

    def print_debug(self, filename="out.txt", filefolder_name = None, output=None):
        global tensor_list
        file = f"{filename}"
        with open(file, "w") as f:
            f.write(f"Number of operations: {len(self.ordered_operations)}\n")
            f.write(f"Operations: {self.ordered_operations}\n")
            f.write(f"Layer list: {self.layer_list}\n")
            f.write(f"Tensor pool:\n")

        for op_name in self.ordered_operations:
            op = self.ordered_graph.nodes[op_name]['op']
            # print("op name:", op.name, "layer:", op.layer, "batch_size:", op.batch_size, "device:", op.device)

            for inputs in op.inputs.values():
                tensor_pool[f"{op.name}_{inputs.name}"] = inputs.tensor
                tensor_list.append((f"{op.name}_{inputs.name}", inputs.tensor))
            for weights in op.weights.values():
                tensor_pool[f"{op.name}_{weights.name}"] = weights.weight_map[op.layer]
                tensor_list.append((f"{op.name}_{weights.name}", weights.weight_map[op.layer]))

            with prof_marker(f"{op.name}"):
                op.wait_cuda_event()
                op.run()
                op.record_cuda_event()

            for outputs in op.outputs.values():
                tensor_pool[f"{op.name}_{outputs.name}"] = outputs.tensor
                tensor_list.append((f"{op.name}_{outputs.name}", outputs.tensor))
            
            # if "AllGatherD" in op.name:
            #     torch.cuda.synchronize()
            #     print("Execution finished, saving tensors...")
            #     with open(file, "a") as f:
            #         for name, tensor in tensor_list:
            #             f.write(f"[{name}]\n{tensor}\n{tensor.shape}\n")
            #     tensor_list = []
                # output.copy_(torch.tensor([358] * 14, dtype=torch.float16, device=op.device))
                # break
            if "GlobalOutput" in op.name:
                torch.cuda.synchronize()
                output.copy_(op.inputs["tokens"].tensor)

        torch.cuda.synchronize()
        print("Execution finished, saving tensors...")
        with open(file, "a") as f:
            for name, tensor in tensor_list:
                f.write(f"[{name}]\n{tensor}\n{tensor.shape}\n")
        # torch.save(tensor_pool, f"{filefolder_name}/output.pt")
