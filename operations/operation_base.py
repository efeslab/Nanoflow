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
        self.conn = sqlite3.connect('performance.db')
        self.cursor = self.conn.cursor()
        # Create a table to store performance data if it doesn't exist
        # self.cursor.execute('''
        #     CREATE TABLE IF NOT EXISTS performance (
        #         id INTEGER PRIMARY KEY AUTOINCREMENT,
        #         keyword TEXT,
        #         batch_size INTEGER,
        #         average_time REAL
        #     )
        # ''')
        self.conn.commit()
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
        
    @property
    def prerequisites(self):
        dep = []
        for _, input_wrapper in self.operator_device.parent.inputs.items():
            for dep_wrapper, prev_layer in zip(input_wrapper.prev, input_wrapper.prev_depend_on_prev_layer):
                # Skip the virtual operations to find the real dependency
                if dep_wrapper.owner.isVirtual == True:
                    dep.extend(dep_wrapper.real_deps[input_wrapper])
                else:
                    dep.append((dep_wrapper.owner, prev_layer))
        
        return dep
    
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
    
    def processWeight(self, global_weight_map, total_layers, cached = False):
        return process_weight_none(global_weight_map, self.weight_name, None, total_layers, cached)
    
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
            
    def config_tag(self, tag, parameter_map = {}):
        self.tag = tag
        parts = tag.split(":", 1)
        category_tag = ""
        impl_tag = ""
        if len(parts) == 1:
            category_tag = parts[0]
        else:
            category_tag = parts[0]
            impl_tag = parts[1]
        self.impl  = self.impl_map[category_tag](self.inputs, self.outputs, self.weights)
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
    
    def expand_gpu(self, gpu_list):
        for i in gpu_list:
            i_str = str(i)
            name = self.name + "_" + i_str
            op_device = Operation_Device(self, name, i)
            self.children.append(op_device)
        
        return self.children
    
class Operation_Device(Operations):
    def __init__(self, op_general, name, device):
        super().__init__(name)
        self.parent = op_general
        self.device = device
        self.weights = op_general.weights
        self.externals = self.parent.externals
        self.impl = self.parent.impl
        self.children = []
        self.inputs = {}
        for key, base_wrapper in op_general.inputs.items():
            dev_wrapper = IOWrapper_Device(
                owner=self,
                name=base_wrapper.name,
                IOtype=base_wrapper.IOtype,
                dtype=base_wrapper.dtype
            )
            base_wrapper.append_child(dev_wrapper)
            self.inputs[key] = dev_wrapper

        self.outputs = {}
        for key, base_wrapper in op_general.outputs.items():
            dev_wrapper = IOWrapper_Device(
                owner=self,
                name=base_wrapper.name,
                IOtype=base_wrapper.IOtype,
                dtype=base_wrapper.dtype
            )
            base_wrapper.append_child(dev_wrapper)
            self.outputs[key] = dev_wrapper
    
    def expand_layer(self, layer_list):
        for i in layer_list:
            op_layer = Operation_Layer(self, self.name + "_" + str(i), i, self.device)
            self.children.append(op_layer)
        
        return self.children

class Operation_Layer(Operation_Device):
    def __init__(self, op_general, name, layer, device):
        super().__init__(name)
        self.layer = layer
        self.inputs = {}
        self.outputs = {}
        self.weights = {}
        self.externals = {}
        self.impl:OperationImpl = None
        self.device = device
        self.parent = op_general

    