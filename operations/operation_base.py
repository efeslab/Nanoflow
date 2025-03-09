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
        return self
    
    def __str__(self):
        return self.name   