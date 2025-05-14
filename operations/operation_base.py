from unicodedata import category
from operations.impl_base import OperationImpl
import torch
import sqlite3
from abc import ABC, abstractmethod
from core.weightWrapper import WeightWrapper    
from core.processWeight import process_weight_none
from core.IOWrapper import IOWrapper_Device

class Operations:
    def __init__(self, name):
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
        self.children = {}
        self.isNanoSplit = False
        self.nano_ops = []
        self.nano_op_batchsizes = []
        self.isVirtual = False
        self.stream = None

        # Connect to the database
        # self.conn = sqlite3.connect('performance.db')
        # self.cursor = self.conn.cursor()
        # # Create a table to store performance data if it doesn't exist
        # # self.cursor.execute('''
        # #     CREATE TABLE IF NOT EXISTS performance (
        # #         id INTEGER PRIMARY KEY AUTOINCREMENT,
        # #         keyword TEXT,
        # #         batch_size INTEGER,
        # #         average_time REAL
        # #     )
        # # ''')
        # self.conn.commit()
        self.impl_map = {}
        self.op_device = None
        
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
        self.updateChildrenIOShape()
    
    def updateChildrenIOShape(self):
        for op_device in self.children.values():
            op_device.setShapeForIOWrappers()

    def processWeight(self, global_weight_map, weight_path, cached, device):
        return process_weight_none(global_weight_map, self.weight_name, None, self.device_list, self.layer_list, weight_path, cached, device)
    
    def first_only(self):
        self.first_layer_only = True
        return self
    
    def last_only(self):
        self.last_layer_only = True
        return self
    
    def search_profile_data(self):
        self.cursor.execute('''
            SELECT * FROM performance
        ''')
        rows = self.cursor.fetchall()
        for row in rows:
            print(row)
            
    def config_tag(self, tag, device, parameter_map = {}):
        if self.isNanoSplit:
            for nano_op in self.nano_ops:
                nano_op.config_tag(tag, device, parameter_map)
            return self
        else:
            self.tag = tag
            self.device = device
            self.parameter_map = parameter_map
            parts = tag.split(":", 1)
            category_tag = ""
            impl_tag = ""
            if len(parts) == 1:
                category_tag = parts[0]
            else:
                category_tag = parts[0]
                impl_tag = parts[1]
            # self.impl  = self.impl_map[category_tag](self.inputs, self.outputs, self.weights, device)
            self.impl  = self.impl_map[category_tag](self, self.stream, device)
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
        for device, child in self.children.items():
            child.append_dependency((extra_dep[0].children[device], extra_dep[1], extra_dep[2]))

    def __str__(self):
        return self.name   
    
    def expand_gpu(self, num_devices):
        self.device_list = [f"cuda:{i}" for i in range(num_devices)]
        for device in self.device_list:
            op_device = self.op_device(self, device)
            self.children[device] = op_device
        
        return self.children
    
    def expand_all_gpu_and_layers(self, num_devices, num_layers):
        self.device_list = [f"cuda:{i}" for i in range(num_devices)]
        if self.first_layer_only:
            self.layer_list = [0]
        elif self.last_layer_only:
            self.layer_list = [num_layers - 1]
        else:
            self.layer_list = list(range(num_layers))

        self.op_layers_per_device = {}
        for device in self.device_list:
            op_device = self.op_device(self, device)

            self.op_layers_per_device[device] = op_device.expand_layer(self.layer_list)
            self.children[device] = op_device
        
        return self.children, self.op_layers_per_device
    
class Operation_Device:
    def __init__(self, parent, device):
        self.name = parent.name
        self.parent = parent
        self.device = device
        self.weights = parent.weights
        self.externals = self.parent.externals
        self.extra_dep = []
        self.children = []
        self.batch_size = None
        self.inputs = {}
        for key, base_wrapper in parent.inputs.items():
            dev_wrapper = IOWrapper_Device(
                owner=self,
                name=base_wrapper.name,
                device=device,
                dtype=base_wrapper.dtype,
                base_wrapper=base_wrapper
            ).is_input()
            base_wrapper.append_child(device, dev_wrapper)
            self.inputs[key] = dev_wrapper

        self.outputs = {}
        for key, base_wrapper in parent.outputs.items():
            dev_wrapper = IOWrapper_Device(
                owner=self,
                name=base_wrapper.name,
                device=device,
                dtype=base_wrapper.dtype,
                base_wrapper=base_wrapper
            ).is_output()
            base_wrapper.append_child(device, dev_wrapper)
            self.outputs[key] = dev_wrapper
    
    def append_dependency(self, dep):
        if dep not in self.extra_dep:
            self.extra_dep.append(dep)

    @property
    def impl(self):
        return self.parent.impl

    @property
    def isVirtual(self):
        return self.parent.isVirtual
    
    @property
    def first_layer_only(self):
        return self.parent.first_layer_only
    
    @property
    def last_layer_only(self):
        return self.parent.last_layer_only

    def expand_layer(self, layer_list):
        for i in layer_list:
            op_layer = self.op_layer(i, self)
            self.children.append(op_layer)
        
        return self.children

    def setBatchSize(self, batch_size):
        self.batch_size = batch_size
        for _, input_wrapper in self.inputs.items():
            input_wrapper.batch_size = batch_size
        for _, output_wrapper in self.outputs.items():
            output_wrapper.batch_size = batch_size
        

class Operation_Layer:
    def __init__(self, layer, op_device):
        self.layer = layer
        self.name = f"{op_device.name}_{layer}"
        self.inputs = op_device.inputs
        self.outputs = op_device.outputs
        self.weights = op_device.weights
        self.externals = op_device.externals
        self.parent = op_device
        self.device = op_device.device
        self.prev_op_layer = []
        self.cuda_event = None
        self.is_depended_on = False

    @property
    def impl(self):
        return self.parent.impl

    @property
    def stream(self):
        return self.parent.parent.stream

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
            prev.extend(input_wrapper.prev)
            depend_on_prev.extend(input_wrapper.prev_depend_on_prev_layer)
            depend_on_next.extend(input_wrapper.prev_depend_on_next_layer)
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
        self.cuda_event = None

    def append_prev_op_layer(self, op_layer):
        self.prev_op_layer.append(op_layer)
    
    def set_is_depended_on(self, op_layer):
        if self.stream != op_layer.stream:
            self.is_depended_on = True

    def record_cuda_event(self):
        if self.is_depended_on:
            if self.cuda_event is None:
                self.cuda_event = torch.cuda.Event(enable_timing=True)
            self.cuda_event.record(self.stream)
            # print("record_cuda_event: ", self.name, "cuda_event: ", self.cuda_event)
    
    def wait_cuda_event(self):
        events = []
        for op_layer in self.prev_op_layer:
            if op_layer.cuda_event is not None and self.stream != op_layer.stream:
                events.append(op_layer.cuda_event)
        for event in events:
            self.stream.wait_event(event)