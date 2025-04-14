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
        self.children = []
        self.isVirtual = False

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
        for op_device in self.children:
            op_device.setShapeForIOWrappers()
    
    def processWeight(self, global_weight_map, total_devices, total_layers, cached = False):
        return process_weight_none(global_weight_map, self.weight_name, None, total_devices, total_layers, cached)
    
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
            
    def config_tag(self, tag, device_id, parameter_map = {}):
        self.tag = tag
        parts = tag.split(":", 1)
        category_tag = ""
        impl_tag = ""
        if len(parts) == 1:
            category_tag = parts[0]
        else:
            category_tag = parts[0]
            impl_tag = parts[1]
        # self.impl  = self.impl_map[category_tag](self.inputs, self.outputs, self.weights, device_id)
        self.impl  = self.impl_map[category_tag](self, device_id)
        self.config_impl(impl_tag, parameter_map)
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
    

    def __str__(self):
        return self.name   
    
    def expand_gpu(self, num_devices):
        for device_id in range(num_devices):
            op_device = self.op_device(self, device_id)
            self.children.append(op_device)
        
        return self.children
    
    def expand_gpu_and_layers(self, num_devices, layer_list):
        self.op_layers_per_device = []
        for device_id in range(num_devices):
            op_device = self.op_device(self, device_id)
            if self.first_layer_only:
                layer_list = [layer_list[0]]
            elif self.last_layer_only:
                layer_list = [layer_list[-1]]
        
            self.op_layers_per_device.append(op_device.expand_layer(layer_list))
            self.children.append(op_device)
        
        return self.children, self.op_layers_per_device
    
class Operation_Device:
    def __init__(self, parent, device_id):
        self.name = parent.name
        self.parent = parent
        self.device_id = device_id
        self.weights = parent.weights
        self.externals = self.parent.externals
        self.children = []
        self.batch_size = None
        self.inputs = {}
        for key, base_wrapper in parent.inputs.items():
            dev_wrapper = IOWrapper_Device(
                owner=self,
                name=base_wrapper.name,
                device_id=device_id,
                dtype=base_wrapper.dtype,
                base_wrapper=base_wrapper
            ).is_input()
            base_wrapper.append_child(dev_wrapper)
            self.inputs[key] = dev_wrapper

        self.outputs = {}
        for key, base_wrapper in parent.outputs.items():
            dev_wrapper = IOWrapper_Device(
                owner=self,
                name=base_wrapper.name,
                device_id=device_id,
                dtype=base_wrapper.dtype,
                base_wrapper=base_wrapper
            ).is_output()
            base_wrapper.append_child(dev_wrapper)
            self.outputs[key] = dev_wrapper
    
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
        self.device_id = op_device.device_id

    @property
    def impl(self):
        return self.parent.impl

    @property
    def prerequisites(self):
        dep = []
        prev = []
        depend_on_prev = []
        for _, input_wrapper in self.parent.inputs.items():
            prev.extend(input_wrapper.prev)
            depend_on_prev.extend(input_wrapper.prev_depend_on_prev_layer)

        while len(prev) > 0:
            dep_wrapper = prev.pop()
            prev_layer = depend_on_prev.pop()
            if dep_wrapper.owner.isVirtual == False:
                dep.append((dep_wrapper.owner, prev_layer))
            elif dep_wrapper.owner.isCopy:
                assert len(dep_wrapper.prev) == 1, f"Copy operation '{dep_wrapper.name}' has more than one prev connections!\n"
                prev.append(dep_wrapper.prev[0])
                depend_on_prev.append(prev_layer or dep_wrapper.prev_depend_on_prev_layer[0])
            elif dep_wrapper.owner.isRedist:
                if dep_wrapper.is_input_wrapper:
                    assert len(dep_wrapper.prev) == 1, f"Redist operation '{dep_wrapper.name}' has more than one prev connections!\n"
                    prev.append(dep_wrapper.prev[0])
                    depend_on_prev.append(prev_layer or dep_wrapper.prev_depend_on_prev_layer[0])
                elif dep_wrapper.is_output_wrapper:
                    input_wrapper_begin_id = None
                    input_wrapper_end_id = None
                    tensor_offset = dep_wrapper.tensor_offset
                    tensor_end = tensor_offset + dep_wrapper.batch_size
                    for idx, input_wrapper in enumerate(dep_wrapper.owner.inputs.values()):
                        if input_wrapper_begin_id is None and input_wrapper.tensor_offset + input_wrapper.batch_size > tensor_offset:
                            input_wrapper_begin_id = idx
                        if input_wrapper_end_id is None and input_wrapper.tensor_offset + input_wrapper.batch_size >= tensor_end:
                            input_wrapper_end_id = idx
                            break

                    if input_wrapper_begin_id is None:
                        continue

                    for idx in range(input_wrapper_begin_id, input_wrapper_end_id + 1):
                        prev.append(dep_wrapper.owner.inputs[f"input_{idx}"])
                        depend_on_prev.append(prev_layer)
                        
        return dep
    