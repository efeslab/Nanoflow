import torch
import matplotlib.pyplot as plt
import networkx as nx
from utils.graph_plot import plot_graph_topological, draw_graphs_subplots

class Executor():
    def __init__(self, operations_list, layer):
        self.operations_list = operations_list
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
                            G.add_edge(f"{dep.name}_{i - 1}", f"{op.name}_{i}")
                    else:
                        G.add_edge(f"{dep.name}_{i}", f"{op.name}_{i}")
        self.ordered_operations = list(nx.topological_sort(G))
        self.ordered_graph = G
    
    def draw_ordered_graph(self):
        plot_graph_topological(self.ordered_graph)
    
    def execute(self, weight_map, new_token):
        for op_name in self.ordered_operations:
            op, layer = self.ordered_graph.nodes[op_name]['op'], self.ordered_graph.nodes[op_name]['layer']          
            op.run(layer)
            if op.name == "GlobalOutput":
                new_token.copy_(op.inputs["tokens"].tensor[-1])
    
    def print_debug(self, filename="out.txt", new_token=None):
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
                    # torch.save(inputs.tensor.cpu(), f"./out/{op.name}_{layer}_{inputs.name}")
                    

                for weights in op.weights.values():
                    f.write(f"[{op.name}_{layer}_{weights.name}]\n")
                    f.write(str(weights.weight_map[layer]))
                    f.write("\n")
                    f.write(str(weights.weight_map[layer].shape))
                    f.write("\n")
                    # torch.save(weights.weight_map[layer].cpu(), f"./out/{op.name}_{layer}_{weights.name}")

                f.flush()

                op.run(layer)  # Execute the operation

                if op.name == "GlobalOutput":
                    new_token.copy_(op.inputs["tokens"].tensor[-1])
                for outputs in op.outputs.values():
                    f.write(f"[{op.name}_{layer}_{outputs.name}]\n")
                    f.write(str(outputs.tensor))
                    f.write("\n")
                    f.write(str(outputs.tensor.shape))
                    f.write("\n")
                    # torch.save(outputs.tensor.cpu(), f"./out/{op.name}_{layer}_{outputs.name}")

                f.flush()
            f.close()
                    