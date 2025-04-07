import torch
import matplotlib.pyplot as plt
import networkx as nx
from utils.prof_marker import prof_marker
from utils.graph_plot import plot_graph_topological, draw_graphs_subplots

class Executor():
    def __init__(self, operations_list, operation_layers_list, layer):
        self.operations_list = operations_list
        self.operations_layers_list = operation_layers_list
        self.layer = layer
        self.ordered_operations = []
    
    def not_this_layer(self, op, layer):
        return (op.first_layer_only and layer != 0) or (op.last_layer_only and layer != self.layer - 1)
    
    def plan_layer_ordering(self):
        G = nx.DiGraph()
        for op in self.operations_list:
            for l in range(self.layer):
                if (not l == 0 and op.first_layer_only) or (not l == self.layer - 1 and op.last_layer_only):
                    continue
                G.add_node(f"{op.name}_{l}", op=op, layer=l)
        for op in self.operations_list:
            for i in range(self.layer):
                if (self.not_this_layer(op, i)):
                        continue
                for dep, dep_on_prev_layer in op.prerequisites:
                    if (self.not_this_layer(dep, i)):
                        continue
                    if dep_on_prev_layer:
                        if i > 0:
                            print("add edge", f"{dep.name}_{i - 1}", f"{op.name}_{i}")
                            G.add_edge(f"{dep.name}_{i - 1}", f"{op.name}_{i}")
                    else:
                        G.add_edge(f"{dep.name}_{i}", f"{op.name}_{i}")
        self.ordered_operations = list(nx.topological_sort(G))
        self.ordered_graph = G
    
    def plan_layer_ordering_using_operator_layers(self):
        G = nx.DiGraph()
        for op_layers in self.operations_layers_list:
            for op in op_layers:
                G.add_node(f"{op.name}", op=op, layer = op.layer)


        for op_layers in self.operations_layers_list:
            for op in op_layers:
                layer = op.layer
                for dep, dep_on_prev_layer in op.prerequisites:
                    if (self.not_this_layer(dep, layer)):
                        continue
                    if dep_on_prev_layer:
                        if layer > 0:
                            G.add_edge(f"{dep.name}_{layer - 1}", f"{op.name}")
                    else:
                        G.add_edge(f"{dep.name}_{layer}", f"{op.name}")

        self.ordered_operations = list(nx.topological_sort(G))
        self.ordered_graph = G
    
    def draw_ordered_graph(self):
        plot_graph_topological(self.ordered_graph)
    
    def execute(self, weight_map, output):
        for op_name in self.ordered_operations:
            print("op_name", op_name)
            op, layer = self.ordered_graph.nodes[op_name]['op'], self.ordered_graph.nodes[op_name]['layer']          
            with prof_marker(f"{op.name}_{layer}"):
                op.run(layer)
            if op.name == "GlobalOutput":
                output.copy_(op.inputs["tokens"].tensor)
    
    def execute_using_operator_layers(self, weight_map, output):
        for op_name in self.ordered_operations:
            op = self.ordered_graph.nodes[op_name]['op']
            with prof_marker(f"{op.name}"):
                op.run()
            if op.name == "GlobalOutput_31":
                output.copy_(op.inputs["tokens"].tensor)

    def print_debug_using_operator_layers(self, filename="out.txt", filefolder_name = None, output=None):
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

                # for weights in op.weights.values():
                #     f.write(f"[{op.name}_{weights.name}]\n")
                #     f.write(str(weights.weight_map))
                #     f.write("\n")
                #     f.write(str(weights.weight_map.shape))
                #     torch.save(weights.weight_map.cpu(), f"./{filefolder_name}/{op.name}_{weights.name}")

                f.flush()

                op.run()

                if op.name == "GlobalOutput_31":
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
    
    def print_debug(self, filename="out.txt", filefolder_name = None, output=None):
        file = f"{filename}"

        with open(file, "w") as f:
            for op_name in self.ordered_operations:
                op, layer = self.ordered_graph.nodes[op_name]['op'], self.ordered_graph.nodes[op_name]['layer']
                print(f"{op.name}_{layer}")

                for inputs in op.inputs.values():

                    f.write(f"[{op.name}_{layer}_{inputs.name}]\n")
                    f.write(str(inputs.tensor))
                    f.write("\n")
                    f.write(str(inputs.tensor.shape))
                    f.write("\n")
                    torch.save(inputs.tensor.cpu(), f"./{filefolder_name}/{op.name}_{layer}_{inputs.name}")
                    

                for weights in op.weights.values():
                    f.write(f"[{op.name}_{layer}_{weights.name}]\n")
                    f.write(str(weights.weight_map[layer]))
                    f.write("\n")
                    f.write(str(weights.weight_map[layer].shape))
                    f.write("\n")
                    torch.save(weights.weight_map[layer].cpu(), f"./{filefolder_name}/{op.name}_{layer}_{weights.name}")

                f.flush()

                op.run(layer)  # Execute the operation

                if op.name == "GlobalOutput":
                    output.copy_(op.inputs["tokens"].tensor)
                for outputs in op.outputs.values():
                    f.write(f"[{op.name}_{layer}_{outputs.name}]\n")
                    f.write(str(outputs.tensor))
                    f.write("\n")
                    f.write(str(outputs.tensor.shape))
                    f.write("\n")
                    torch.save(outputs.tensor.cpu(), f"./{filefolder_name}/{op.name}_{layer}_{outputs.name}")

                f.flush()
            f.close()
                    