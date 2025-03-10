from unicodedata import category
from operations.impl_base import OperationImpl
from param import String
import torch
import sqlite3
from abc import ABC, abstractmethod
from core.weightWrapper import WeightWrapper    
from core.processWeight import process_weight_none

class Operations:
    def __init__(self, name):
        self.inputs = {}
        self.outputs = {}
        self.weights = {}
        self.externals = {}
        self.name = name
        self.first_layer_only = False
        self.last_layer_only = False
        self.weight_name = None
        self.tag = "torch"
        self.impl:OperationImpl = None

        # Connect to the database
        self.conn = sqlite3.connect('performance.db')
        self.cursor = self.conn.cursor()
        # Create a table to store performance data if it doesn't exist
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword TEXT,
                batch_size INTEGER,
                average_time REAL
            )
        ''')
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
        for _, input_wrapper in self.inputs.items():
            for dep_wrapper, prev_layer in zip(input_wrapper.prev, input_wrapper.prev_depend_on_prev_layer):
                dep.append((dep_wrapper.owner, prev_layer))
        
        return dep
        
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
            
    def config_tag(self, tag):
        self.tag = tag
        parts = tag.split(":", 1)
        category_tag = ""
        impl_tag = ""
        if len(parts) == 1:
            category_tag = parts[0]
        else:
            category_tag = parts[0]
            impl_tag = parts[1]
        self.impl  = self.impl_map[category_tag]()
        self.config_impl(impl_tag)
        return self
    
    def config_impl(self, impl_tag):
        self.impl.config(impl_tag)
        
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