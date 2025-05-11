import torch
import matplotlib.pyplot as plt
import networkx as nx
from utils.prof_marker import prof_marker
from utils.graph_plot import plot_graph_topological, draw_graphs_subplots

class Executor():
    def __init__(self, operations_layers_list, layer):
        self.operations_layers_list = [op_layer for op_layer in operations_layers_list if op_layer.batch_size > 0]
        # self.operations_layers_list = operations_layers_list
        self.layer = layer
        self.ordered_operations = []
    
    def not_this_layer(self, op, layer):
        return (op.first_layer_only and layer != 0) or (op.last_layer_only and layer != self.layer - 1)
    
    
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
                    if layer < self.layer - 1:
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

    def print_debug(self, filename="out.txt", rank=0, filefolder_name = None, output=None):
        file = f"{filename}"

        with open(file, "w") as f:
            for op_name in self.ordered_operations:
                op = self.ordered_graph.nodes[op_name]['op']
                print(f"{op.name}")

                for inputs in op.inputs.values():
                    f.write(f"[{op.name}_{inputs.name}]\n")
                    f.write(str(inputs.tensor))
                    f.write("\n")
                    f.write(str(inputs.tensor.shape))
                    torch.save(inputs.tensor.cpu(), f"./{filefolder_name}/{op.name}_{inputs.name}")

                for weights in op.weights.values():
                    f.write(f"[{op.name}_{weights.name}]\n")
                    f.write(str(weights.weight_map[rank][op.layer]))
                    f.write("\n")
                    f.write(str(weights.weight_map[rank][op.layer].shape))
                    torch.save(weights.weight_map[rank][op.layer].cpu(), f"./{filefolder_name}/{op.name}_{weights.name}")

                f.flush()

                with prof_marker(f"{op.name}"):
                    op.wait_cuda_event()
                    op.run()
                    op.record_cuda_event()

                if "GlobalOutput" in op.name:
                    torch.cuda.synchronize()
                    output.copy_(op.inputs["tokens"].tensor)
                for outputs in op.outputs.values():
                    f.write(f"[{op.name}_{outputs.name}]\n")
                    f.write(str(outputs.tensor))
                    f.write("\n")
                    f.write(str(outputs.tensor.shape))
                    f.write("\n")
                    torch.save(outputs.tensor.cpu(), f"./{filefolder_name}/{op.name}_{outputs.name}")

                f.flush()
            f.close()
    