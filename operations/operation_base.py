from unicodedata import category
from operations.impl_base import OperationImpl
import torch
from abc import ABC, abstractmethod
from core.weightWrapper import WeightWrapper    
from core.processWeight import process_weight_none

class Operations:
    def __init__(self, name, device):
        # should be initialized in the device class
        self.inputs = {}
        self.outputs = {}
        self.weights = {}
        self.externals = {}
        self.impl:OperationImpl = None

        # remain in this class
        self.name = name
        self.first_layer_only = False
        self.last_layer_only = False
        self.weight_name = None
        self.tag = "torch"
        self.children = []
        self.isNanoSplit = False
        self.nano_ops = []
        self.nano_op_batchsizes = []
        self.isVirtual = False
        self.stream = None
        self.batch_size = None

        self.device = device
        self.extra_dep = []

        self.impl_map = {}
        self.op_layer = None
        
    def init_impl_map(self):
        self.impl_map = {} 
    
    def add_impl(self, impl):
        # if impl is not a list, convert it to a list
        if not isinstance(impl, list):
            impl = [impl]
        for i in impl:
            self.impl_map[i.category_tag] = i
        
    def print_available_impl(self):
        print(self.impl_map.keys())
    
    def checkConsistencyBetweenImpl(self, outputs):
        if self.impl_map.keys() == 0:
            raise Exception("No implementation found")
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                print(f"Checking consistency between outputs {i} and {j} for operation {self.name}")
                close_elements = torch.isclose(outputs[i], outputs[j], rtol=1e-01, atol=1e-03)
                assert torch.all(close_elements), f"Outputs from different implementations are not close: {outputs[i]} and {outputs[j]}"

    def checkConnection(self):
        for name, IOwrapper in self.inputs.items():
            if IOwrapper.prev is None:
                raise Exception(f"Operation {self.name}, Input {name} is not connected")
    
    def setWeightName(self, name):
        self.weight_name = name
        return self

    def setShape(self):
        return None

    def processWeight(self, global_weight_map, weight_path, cached, device):
        return process_weight_none(global_weight_map, self.weight_name, None, self.layer_list, weight_path, cached, device)
    
    def first_only(self):
        self.first_layer_only = True
        return self
    
    def last_only(self):
        self.last_layer_only = True
        return self
    
    def print_profile(self):
        assert hasattr(self, 'cursor'), "Profiling has not been run yet. Please call profile() first."
        # print the profiling results
        for _, impl in self.impl_map.items():
            category_tag = impl.category_tag
            self.cursor.execute(f'''
                SELECT * FROM {category_tag}
            ''')
            cols = [col_desc[0] for col_desc in self.cursor.description]
            rows = self.cursor.fetchall()
            print(f"Profiling results for {category_tag}:")
            print(" | ".join(cols))            # header line
            for row in rows:
                print(" | ".join(str(val) for val in row))
        self.conn.close()
            
    def config_tag(self, tag, parameter_map = {}):
        if self.isNanoSplit:
            assert len(tag) == len(self.nano_ops), f"Operation {self.name} has {len(self.nano_ops)} nano ops, but {len(tag)} tags were provided."
            for i, nano_op in enumerate(self.nano_ops):
                nano_op.config_tag(tag[i], parameter_map)
            return self
        else:
            self.tag = tag
            self.parameter_map = parameter_map
            parts = self.tag.split(":", 1)
            category_tag = ""
            impl_tag = ""
            if len(parts) == 1:
                category_tag = parts[0]
            else:
                category_tag = parts[0]
                impl_tag = parts[1]
            # self.impl  = self.impl_map[category_tag](self.inputs, self.outputs, self.weights, device)
            self.impl  = self.impl_map[category_tag](self, self.stream, self.device)
            self.config_impl(impl_tag, parameter_map)
            # print("name: ", self.name, "category_tag: ", category_tag, "impl_tag: ", impl_tag, "impl: ", self.impl)
            return self
    
    def config_impl(self, impl_tag, parameter_map):
        self.impl.config(impl_tag, parameter_map)
        
    def get_all_tags(self):
        tag_list = []
        for key in self.impl_map.keys():
            category_tag = key
            impl_list = self.impl_map[key].list_tags()
            for impl_tag in impl_list:
                if not impl_tag == "":
                    tag_list.append(category_tag + ":" + impl_tag)
                else:
                    tag_list.append(category_tag)
        return tag_list
    
    def set_stream(self, stream):
        self.stream = stream

    def append_dependency(self, extra_dep):
        if extra_dep not in self.extra_dep:
            self.extra_dep.append(extra_dep)

    def __str__(self):
        return self.name

    def expand_layer(self, layer_list):
        if self.first_layer_only:
            self.layer_list = layer_list[:1]
        elif self.last_layer_only:
            self.layer_list = layer_list[-1:]
        else:
            self.layer_list = layer_list

        for layer_idx in self.layer_list:
            self.children.append(self.op_layer(layer_idx, self))
        
        return self.children
    
    def setBatchSize(self, batch_size):
        self.batch_size = batch_size
        for _, input_wrapper in self.inputs.items():
            input_wrapper.batch_size = batch_size
        for _, output_wrapper in self.outputs.items():
            output_wrapper.batch_size = batch_size
    
class Operation_Layer:
    def __init__(self, layer, base_op):
        self.layer = layer
        self.name = f"{base_op.name}_{layer}"
        self.inputs = base_op.inputs
        self.outputs = base_op.outputs
        self.weights = base_op.weights
        self.externals = base_op.externals
        self.parent = base_op
        self.device = base_op.device
        self.prev_op_layer = []
        self.cuda_event = torch.cuda.Event(enable_timing=True) 
        self.is_depended_on = False

    @property
    def impl(self):
        return self.parent.impl

    @property
    def stream(self):
        return self.parent.stream

    @property
    def batch_size(self):
        return self.parent.batch_size

    @property
    def prerequisites(self):
        dep = []
        dep.extend(self.parent.extra_dep)
        # print("init dep: ", self.name, "dep: ", [dep[0].name for dep in dep])
        prev = []
        depend_on_prev = []
        depend_on_next = []
        for _, input_wrapper in self.parent.inputs.items():
            prev.extend(input_wrapper.actual_prev)
            depend_on_prev.extend(input_wrapper.actual_prev_depend_on_prev_layer)
            depend_on_next.extend(input_wrapper.actual_prev_depend_on_next_layer)
        while len(prev) > 0:
            assert len(prev) == len(depend_on_prev) == len(depend_on_next), f"Operation '{self.name}' has different number of prev and depend_on_prev connections!\n"
            dep_wrapper = prev.pop()
            prev_layer = depend_on_prev.pop()
            next_layer = depend_on_next.pop()
            # if "Rope" in self.name:  
            #     print("dep_wrapper.owner.name: ", dep_wrapper.owner.name)
            #     print("dep_wrapper.name: ", dep_wrapper.name)
            if dep_wrapper.owner.isVirtual == False:
                flag = False
                for _, input_wrapper in self.parent.inputs.items():
                    if dep_wrapper.is_intersect(input_wrapper):
                        flag = True
                        break
                    # print("dep_wrapper.name: ", dep_wrapper.name)
                dep.append((dep_wrapper.owner, prev_layer, next_layer)) if flag else None
            elif dep_wrapper.owner.isCopy:
                for idx, wrapper in enumerate(dep_wrapper.prev):
                    prev.append(wrapper)
                    depend_on_prev.append(prev_layer or dep_wrapper.prev_depend_on_prev_layer[idx])
                    depend_on_next.append(next_layer or dep_wrapper.prev_depend_on_next_layer[idx])
            elif dep_wrapper.owner.isRedist:
                if dep_wrapper.is_input_wrapper:
                    for idx, wrapper in enumerate(dep_wrapper.prev):
                        prev.append(wrapper)
                        depend_on_prev.append(prev_layer or dep_wrapper.prev_depend_on_prev_layer[idx])
                        depend_on_next.append(next_layer or dep_wrapper.prev_depend_on_next_layer[idx])
                elif dep_wrapper.is_output_wrapper:
                    for idx, input_wrapper in enumerate(dep_wrapper.owner.inputs.values()):
                        # if "Rope" in self.name: 
                        #     print("input_wrapper.name: ", input_wrapper.name)
                        #     print("input_wrapper.tensor_offset: ", input_wrapper.tensor_offset)
                        #     print("input_wrapper.batch_size: ", input_wrapper.batch_size)
                        #     print("dep_wrapper.tensor_offset: ", dep_wrapper.tensor_offset)
                        #     print("dep_wrapper.batch_size: ", dep_wrapper.batch_size)
                        if input_wrapper.is_intersect(dep_wrapper):
                            # if "Rope" in self.name: 
                            #     print("added")
                            prev.append(input_wrapper)
                            depend_on_prev.append(prev_layer)
                            depend_on_next.append(next_layer)
        # print("prerequisites: ", self.name, "dep: ", [dep[0].name for dep in dep])
        return dep
    
    def reset_op_cuda_status(self):
        self.prev_op_layer = []
        self.is_depended_on = False

    def append_prev_op_layer(self, op_layer):
        self.prev_op_layer.append(op_layer)
    
    def set_is_depended_on(self, op_layer):
        if self.stream != op_layer.stream:
            self.is_depended_on = True

    def record_cuda_event(self):
        if self.is_depended_on:
            self.cuda_event.record(self.stream)
            # print("record_cuda_event: ", self.name, "cuda_event: ", self.cuda_event)
    
    def wait_cuda_event(self):
        events = []
        for op_layer in self.prev_op_layer:
            if self.stream != op_layer.stream:
                events.append(op_layer.cuda_event)
                # print("wait_cuda_event: ", self.name, "prev_op_layer: ", op_layer.name, "cuda_event: ", op_layer.cuda_event)
        for event in events:
            self.stream.wait_event(event)