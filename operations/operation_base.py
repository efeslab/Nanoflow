import os
from typing import Type
from operations.impl_base import OperationImpl
import torch
import sqlite3
from core.IOWrapper import IOWrapper
from core.weightWrapper import WeightWrapper    
from core.processWeight import process_weight_none
from utils.prof_marker import prof_marker

import gurobipy as gp
from gurobipy import GRB

class Operations():
    def __init__(self, name: str, device: str, nano_idx=None):
        # should be initialized in the device class
        self.inputs: dict[str, IOWrapper] = {}
        self.outputs: dict[str, IOWrapper] = {}
        self.weights: dict[str, WeightWrapper] = {}
        self.externals = {}
        self.impl: OperationImpl

        # remain in this class
        self.original_name = name
        if nano_idx is not None:
            self.name = f"{name}{nano_idx}"
        else:
            self.name = name
        
        self.first_layer_only = False
        self.last_layer_only = False
        self.weight_name = None
        self.tag = "torch"
        self.category = None
        self.children = []
        self.isNanoSplit = False
        self.nano_ops = []
        self.nano_op_batchsizes = []
        self.isVirtual = False
        self.stream: torch.cuda.Stream
        self.sm_count: int | None = None
        self.batch_size = None

        self.device = device
        self.extra_dep = []

        self.impl_map: dict[str, Type[OperationImpl]] = {}
        self.op_layer: Type[Operation_Layer]
        
    def init_impl_map(self):
        self.impl_map = {} 
    
    def add_impl(self, impl: Type[OperationImpl]):
        self.impl_map[impl.category_tag] = impl
        
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

    def setShape(self, *args, **kwargs):
        return None

    def processWeight(self, global_weight_map, weight_path, cached, device):
        return process_weight_none(global_weight_map, self.weight_name, None, self.layer_list, weight_path, cached, device)
    
    def first_only(self):
        self.first_layer_only = True
        return self
    
    def last_only(self):
        self.last_layer_only = True
        return self
    
    # profile related methods
    def init_profile_db(self):
        """
        Initialize the database for profiling results.
        This method should create the necessary tables and prepare the database for storing profiling data.
        """
        raise NotImplementedError("This method should be implemented in the subclass.")

    def store_profile_db(self, category_tag, impl_tag, average_elapsed_ms):
        """
        Store the profiling results in the database.
        This method should insert the profiling data into the appropriate tables.
        """
        raise NotImplementedError("This method should be implemented in the subclass.")

    def is_profiled_in_db(self, category_tag):
        self.cursor.execute(f'''
            SELECT * FROM {category_tag}
            WHERE batch_size = ? AND sm_count = ?
        ''', (self.batch_size, self.sm_count))
        row = self.cursor.fetchone()
        if row is not None:
            print(f"Name: {self.name}, Category: {category_tag}, Batch Size: {self.batch_size}, SM Count: {self.sm_count} already profiled.")
            return True
        return False

    def init_impl_configs(self):
        self.impl_configs_map = {}
        for _, impl in self.impl_map.items():
            category_tag = impl.category_tag
            self.impl_configs_map[category_tag] = [
                (None, None)
            ]
    
    def setup_profile_custom(self):
        self.init_impl_configs()

    def setup_profile(self, profile_dir, append_mode=False, is_save_db=True):
        if not os.path.exists(profile_dir):
            os.makedirs(profile_dir)
        
        self.conn = sqlite3.connect(os.path.join(profile_dir, f"{self.name}.db"))
        self.cursor = self.conn.cursor()
        self.is_save_db = is_save_db

        self.setup_profile_custom()
        # create tables for each implementation category
        if self.is_save_db:
            if not append_mode:
                for _, impl in self.impl_map.items():
                    self.cursor.execute(f'''
                        DROP TABLE IF EXISTS "{impl.category_tag}";
                    ''')
            print(f"Setting up profile database for operation {self.name} with append mode: {append_mode}")
            self.init_profile_db()
            self.conn.commit()


    def profile_update(self):
        pass

    def profile_run(self):
        pass

    def profile_all(self):
        with prof_marker(f"batchsize:{self.batch_size}"):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            g = torch.cuda.CUDAGraph()
            for _, impl in self.impl_map.items():
                self.impl = impl(self, self.stream, self.device)
                category_tag = impl.category_tag
                is_profiled = self.is_profiled_in_db(category_tag)
                if is_profiled:
                    # If already profiled, we can skip profiling
                    continue
                # loop in the impl configs
                for impl_tag, para_map in self.impl_configs_map[category_tag]:
                    self.impl.config(impl_tag, para_map)
                    self.profile_update()
                    g.reset()

                    # warm up for 10 cycles.
                    for _ in range(10):
                        self.profile_run()

                    torch.cuda.synchronize()
                    # prepare a graph for 100 cycles.
                    rounds = 100
                    with torch.cuda.graph(g, stream=self.stream):
                        for round in range(rounds):
                            self.profile_run()
                    torch.cuda.synchronize()

                    # sync for network ops.
                    self.profile_run()

                    start.record(self.stream)
                    with torch.cuda.stream(self.stream):
                        g.replay()
                    end.record(self.stream)
                    torch.cuda.synchronize()
                    elapsed_ms = start.elapsed_time(end)
                    average_elapsed_ms = elapsed_ms / rounds
                    # Store to results
                    if self.is_save_db:
                        self.store_profile_db(category_tag, impl_tag, average_elapsed_ms)
            self.conn.commit()

    def print_profile(self):
        if not hasattr(self, 'cursor'):
            print(f"Profiling has not been run yet for operation {self.name}. Please call profile() first.")
            return
        # assert hasattr(self, 'cursor'), "Profiling has not been run yet. Please call profile() first."
        # print the profiling results
        print(f"Profiling results for operation {self.name}:")
        for _, impl in self.impl_map.items():
            category_tag = impl.category_tag
            # if category_tag not in self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall():
            if not self.cursor.execute(f'''
                SELECT name FROM sqlite_master WHERE type='table' AND name="{category_tag}";
            ''').fetchone():
                print(f"No profiling results found for category: {category_tag}")
                continue
            self.cursor.execute(f'''
                SELECT * FROM {category_tag}
            ''')
            cols = [col_desc[0] for col_desc in self.cursor.description]
            rows = self.cursor.fetchall()
            print(f"Profiling results for {category_tag}:")
            print(" | ".join(cols))            # header line
            for row in rows:
                print(" | ".join(str(val) for val in row))
        # self.conn.close()
            
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


    def set_category(self, category: str) -> None:
        if self.isNanoSplit:
            for i, nano_op in enumerate(self.nano_ops):
                nano_op.set_category(category)
        else:
            self.category = category

    def set_stream(self, stream: tuple[torch.cuda.Stream, int | None] | list[tuple[torch.cuda.Stream, int | None]]) -> None:
        if self.isNanoSplit:
            assert isinstance(stream, list), "Stream must be a list of streams"
            for i, nano_op in enumerate(self.nano_ops):
                nano_op.set_stream(stream[i])
        else:
            print(f"Setting stream for {self.name} (isNanoSplit: {self.isNanoSplit}) to {stream}")
            assert not isinstance(stream, list), "Stream must be a single stream"
            self.stream = stream[0]
            self.sm_count = stream[1]


    def append_dependency(self, extra_dep):
        if extra_dep not in self.extra_dep:
            self.extra_dep.append(extra_dep)

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

    def __str__(self):
        return self.name
    
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
        self.prev_op_layer: list[Operation_Layer] = []
        self.cuda_event = torch.cuda.Event(enable_timing=True) 
        self.is_depended_on = False

        # for auto search
        self.duration_map = {}
        self.start_time: gp.Var
        self.end_time: gp.Var
    
    @property
    def impl(self):
        return self.parent.impl

    @property
    def category(self):
        return self.parent.category

    @property
    def stream(self):
        return self.parent.stream

    @property
    def batch_size(self):
        return self.parent.batch_size

    @property
    def original_name(self):
        return self.parent.original_name

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
                        if input_wrapper.is_intersect(dep_wrapper):
                            # if "Rope" in self.name: 
                            #     print("added")
                            prev.append(input_wrapper)
                            depend_on_prev.append(prev_layer)
                            depend_on_next.append(next_layer)
        # print("prerequisites: ", self.name, "dep: ", [dep[0].name for dep in dep])
        # unique dep
        dep = list(set(dep))
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

    # for auto search
    def initVariables(self, model: gp.Model, full_sm_count: int):
        self.start_time = model.addVar(vtype=GRB.CONTINUOUS, name=f"{self.name}_start")
        self.end_time = model.addVar(vtype=GRB.CONTINUOUS, name=f"{self.name}_end")
        # print(f"init_Variables: {self.name}, start_time: {self.start_time}, end_time: {self.end_time}")
        model.addConstr(self.end_time == self.start_time + self.duration_map[(self.batch_size, full_sm_count)], name=f"{self.name}_end_time")

    def initVariablesStageTwo(self, model: gp.Model, sm_counts: list[int]):
        self.start_time = model.addVar(vtype=GRB.CONTINUOUS, name=f"{self.name}_start")
        self.end_time = model.addVar(vtype=GRB.CONTINUOUS, name=f"{self.name}_end")
        self.p_vars: dict[int, gp.Var] = {}  # Variables for p choices
        self.durations: dict[int, float] = {}  # Duration in units for each p
        self.p_choice = model.addVar(vtype=GRB.CONTINUOUS, name=f"{self.name}_p_choice")


        for sm_count in sm_counts:
            self.p_vars[sm_count] = model.addVar(vtype=GRB.BINARY, name=f"{self.name}_p_{sm_count}")
            if sm_count == 76:
                self.p_vars[sm_count].Start = 1  # Force p_56 to be chosen
            else:
                self.p_vars[sm_count].Start = 0
            duration = self.duration_map[(self.batch_size, sm_count)]
            self.durations[sm_count] = duration

    def addInternalConstraintsStageTwo(self, model: gp.Model):
        model.addConstr(
            gp.quicksum(self.p_vars.values()) == 1,
            name=f"{self.name}_p_choice_sum"
        )

        model.addConstr(
            self.end_time == self.start_time + gp.quicksum(
                self.p_vars[sm_count] * self.durations[sm_count] for sm_count in self.p_vars
            ),
            name=f"{self.name}_end_time"
        )
        
        model.addConstr(
            self.p_choice == gp.quicksum(
                sm_count * self.p_vars[sm_count] for sm_count in self.p_vars
            ),
            name=f"{self.name}_p_choice_value"
        )

    def __str__(self) -> str:
        # Color codes for terminal output
        COLOR_YELLOW = "\033[33m"
        COLOR_GREEN = "\033[32m"
        COLOR_BLUE = "\033[34m"
        COLOR_RESET = "\033[0m"
        s = f"{COLOR_BLUE}{self.name}{COLOR_RESET} start: {self.start_time.X:.3f} end: {self.end_time.X:.3f} batch_size: {round(self.batch_size)}"
        return s