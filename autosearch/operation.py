from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Union, Dict, Tuple
import gurobipy as gp
from gurobipy import GRB
import math
import itertools
from matplotlib import pyplot as plt

# Color codes for terminal output
COLOR_YELLOW = "\033[33m"
COLOR_GREEN = "\033[32m"
COLOR_BLUE = "\033[34m"
COLOR_RESET = "\033[0m"

@dataclass
class Dependency:
    operation: Union[str, Operation]
    depend_on_previous_layer: bool = False

    def __str__(self) -> str:
        op_name = self.operation if isinstance(self.operation, str) else self.operation.name
        return f"{op_name}{' P' if self.depend_on_previous_layer else ''}"

@dataclass
class Operation:
    name: str
    start_batch: int
    batch_size: int
    depends_on: List[Union[str, Operation]]
    category: str
    fixed_shape: tuple[int, int]
    batch_size_choice: List[int]
    depend_on_operations: List[Dependency] = field(default_factory=list)
    duration_map: Dict[(int, float), float] = field(default_factory=dict)
    batch_chunks: List[int] = field(default_factory=list)
    num_nano_batch: int = 1
    def __hash__(self) -> int:
        return hash(self.name)
    
    @property
    def p_choices(self) -> List[float]:
        return sorted(set(p for _, p in self.duration_map.keys()))

class LayeredOperation:
    def __init__(self, operation: Operation, layer: int):
        self.operation: Operation = operation
        self.layer = layer
        self.depend_on_operations: List[LayeredOperation] = []

    @property
    def name(self) -> str:
        return f"{self.operation.name}_{self.layer}"

    def __str__(self) -> str:
        dependencies = "\n".join(f"{COLOR_GREEN}   -> {dep.name}{COLOR_RESET}" for dep in self.depend_on_operations)
        parts = [
            f"{COLOR_YELLOW}{self.name}{COLOR_RESET}",
            f"{dependencies}"
        ]
        return "\n".join(part for part in parts if part)

    def __repr__(self) -> str:
        return self.__str__()

@dataclass
class NanoOperation:
    idx: int
    layered_operation: LayeredOperation
    assign: Dict[int, gp.Var] = field(default_factory=dict)
    batch_size: gp.Var = None
    start_time: gp.Var = None
    end_time: gp.Var = None
    duration: gp.Var = None
    batch_size_choice: Dict[int, gp.Var] = field(default_factory=dict)

    def __post_init__(self):
        self.depend_on_nano_ops: List[NanoOperation] = []

    @property
    def operation(self) -> Operation:
        return self.layered_operation.operation

    def initVariables(self, model: gp.Model):
        for chunk_id in self.operation.batch_chunks:
            self.assign[chunk_id] = model.addVar(vtype=GRB.BINARY, name=f"{self.name}_selected_{chunk_id}")
        self.batch_size = model.addVar(vtype=GRB.CONTINUOUS, name=f"{self.name}_batch_size")
        self.start_time = model.addVar(vtype=GRB.CONTINUOUS, name=f"{self.name}_start_time")
        self.end_time = model.addVar(vtype=GRB.CONTINUOUS, name=f"{self.name}_end_time")
        self.duration = model.addVar(vtype=GRB.CONTINUOUS, name=f"{self.name}_duration")
        for batch_size_choice in self.operation.batch_size_choice:
            self.batch_size_choice[batch_size_choice] = model.addVar(vtype=GRB.BINARY, name=f"{self.name}_batch_size_is_{batch_size_choice}")

    def addInternalConstraints(self, model: gp.Model, chunk_size: int):
        # Batch size choice constraints
        model.addConstr(gp.quicksum(self.batch_size_choice.values()) == 1, name=f"{self.name}_batch_size_choice_sum")
        model.addConstr(self.batch_size == gp.quicksum(self.assign.values()) * chunk_size, name=f"{self.name}_batch_size")
        model.addConstr(self.batch_size == gp.quicksum(self.batch_size_choice[batch_size] * batch_size for batch_size in self.operation.batch_size_choice), name=f"{self.name}_batch_size_choice")
        # Duration based on batch size and p=1 (initially)
        model.addConstr(self.duration == gp.quicksum(self.operation.duration_map[(batch_size_choice_value, 1.0)] * self.batch_size_choice[batch_size_choice_value] for batch_size_choice_value in self.operation.batch_size_choice), name=f"{self.name}_duration")
        # End time calculation
        model.addConstr(self.end_time == self.start_time + self.duration, name=f"{self.name}_end_time")

    @property
    def name(self) -> str:
        return f"{self.layered_operation.name}_{self.idx}"

    def __hash__(self) -> int:
        return hash(self.name)

    def __str__(self) -> str:
        s = f"{COLOR_BLUE}{self.name}{COLOR_RESET} start: {self.start_time.X:.3f} end: {self.end_time.X:.3f} duration: {self.duration.X:.3f} batch_size: {round(self.batch_size.X)}"
        for batch_size_choice in self.operation.batch_size_choice:
            if self.batch_size_choice[batch_size_choice].X > 0:
                s += f" batch_size_choice: {int(batch_size_choice)}"
        s += f" chunk_id:"
        for chunk_id in self.operation.batch_chunks:
            if self.assign[chunk_id].X > 0:
                s += f" {chunk_id}"
        return s

class NanoOperationSecondStage:
    def __init__(self, nano_op: NanoOperation):
        self.layered_operation = nano_op.layered_operation
        self.idx = nano_op.idx
        self.batch_size = nano_op.batch_size.X  # Fixed from the first stage
        self.name = nano_op.name
        self.depend_on_nano_ops_logical: List[NanoOperationSecondStage] = []
        self.depend_on_nano_ops_stream: List[NanoOperationSecondStage] = []
        self.operation = self.layered_operation.operation
        self.duration_map = self.operation.duration_map  # Access duration_map via operation
        self.p_choices = self.operation.p_choices
        self.p_vars: Dict[float, gp.Var] = {}  # Variables for p choices
        self.duration_units: Dict[float, int] = {}  # Duration in units for each p
        self.isrunning: Dict[Tuple[float, int], gp.Var] = {}  # Decision variables
        self.solved = False
        self.saved_p_var_x = None

    @property 
    def depend_on_nano_ops(self):
        return self.depend_on_nano_ops_logical + self.depend_on_nano_ops_stream

    def initVariables(self, model: gp.Model, time_unit: float, time_horizon: int):
        # Create variables for p choices and start times
        for p_value in self.p_choices:
            # p_vars: Binary variables indicating the selection of p
            self.p_vars[p_value] = model.addVar(vtype=GRB.BINARY, name=f"{self.name}_p_{p_value}")
            # Calculate duration in units
            duration = self.duration_map.get((int(self.batch_size), p_value), 0.0)
            if duration == 0.0:
                raise ValueError(f"Missing duration for operation {self.name} with batch size {self.batch_size} and p {p_value}")
            duration_units = int(math.ceil(duration / time_unit))
            self.duration_units[p_value] = duration_units
            for t in range(time_horizon + 1):
                self.isrunning[p_value, t] = model.addVar(vtype=GRB.BINARY, name=f"isrunning_{self.name}_{p_value}_{t}")
            
            if self.solved:
                continue
            elif p_value == 0.5:
                self.p_vars[p_value].Start = 1.0
            else:
                self.p_vars[p_value].Start = 0.0
        if self.solved:
            self.apply_current_as_start()
        self.start_time_units = model.addVar(vtype=GRB.INTEGER, name=f"{self.name}_start_time_units")
        self.end_time_units = model.addVar(vtype=GRB.INTEGER, name=f"{self.name}_end_time_units")
        

    def addConstraints(self, model: gp.Model, time_unit: float, time_horizon: int):
        # Ensure only one p_value is selected
        model.addConstr(
            gp.quicksum(self.p_vars.values()) == 1,
            name=f'p_choice_{self.name}'
        )
        
        # Ensure end_time_units is consistent with start_time_units and duration_units
        model.addConstr(
            self.end_time_units == self.start_time_units + gp.quicksum(
                self.duration_units[p_value] * self.p_vars[p_value]
                for p_value in self.p_choices
            ),
            name=f'end_time_units_{self.name}'
        )
        
        # is running
        # ensure before start time, isrunning is 0
        M = 1000
        for p_value in self.p_choices:
            for t in range(time_horizon + 1):
                model.addConstr(
                    self.start_time_units <= t + M * (1 - self.isrunning[p_value, t]),
                    name=f'isrunning_{self.name}_{p_value}_{t}_start'
                )
        # ensure after end time, isrunning is 0
        for p_value in self.p_choices:
            for t in range(time_horizon + 1):
                model.addConstr(
                t <= self.end_time_units - 1 + M * (1 - self.isrunning[p_value, t]),
                name=f'isrunning_{self.name}_{p_value}_{t}_end'
            )
        # ensure isrunning is 0 if p is not selected
        for p_value in self.p_choices:
            for t in range(time_horizon + 1):
                model.addConstr(
                    self.isrunning[p_value, t] <= self.p_vars[p_value],
                    name=f'isrunning_{self.name}_{p_value}_{t}_p'
                )
                
        # method 1        
        # ensure 1s in isrunning is equal to duration_units
        for p_value in self.p_choices:
            model.addConstr(
                gp.quicksum(self.isrunning[p_value, t] for t in range(time_horizon + 1)) >= self.duration_units[p_value] - M * (1 - self.p_vars[p_value]),
                name=f'isrunning_{self.name}_{p_value}_sum'
            )
        
        # # # method 2
        # self.running_helper1 = {}
        # self.running_helper2 = {}
        # self.sum_helper = {}
        # for p_value in self.p_choices:
        #     for t in range(time_horizon + 1):
        #         self.running_helper1[p_value, t] = model.addVar(vtype=GRB.BINARY, name=f'running_helper1_{self.name}_{p_value}_{t}')
        #         self.running_helper2[p_value, t] = model.addVar(vtype=GRB.BINARY, name=f'running_helper2_{self.name}_{p_value}_{t}')

        #         # model.addConstr(t <= self.end_time_units - 1 + M * (1 - self.running_helper1[p_value, t]), name=f'isrunning1_{self.name}_{p_value}_{t}_end')
                
        #         # 1 only occur before end time
        #         model.addConstr(t <= self.end_time_units - 1 + M * (1 - self.running_helper2[p_value, t]), name=f'isrunning2_{self.name}_{p_value}_{t}_end')
        #         # 0 only occur after end time
        #         model.addConstr(self.end_time_units <= t + M * self.running_helper2[p_value, t] + 2 * M * (1 - self.p_vars[p_value]),name=f'running_helper2_{self.name}_{p_value}_{t}_end')
        #         # must be 0 if p is not selected
        #         model.addConstr(self.running_helper2[p_value, t] <= self.p_vars[p_value], name=f'running_helper2_{self.name}_{p_value}_{t}_p')
                
        #         # 1 only occur after start time
        #         model.addConstr(self.start_time_units >= t + 1 - M * self.running_helper1[p_value, t] - 2 * M * (1 - self.p_vars[p_value]), name=f'isrunning1_{self.name}_{p_value}_{t}_start')
        #         # 0 only occur before start time
        #         model.addConstr(self.start_time_units <= t + M * (1 - self.running_helper1[p_value, t]), name=f'running_helper1_{self.name}_{p_value}_{t}_start')
        #         # must be 0 if p is not selected
        #         model.addConstr(self.running_helper1[p_value, t] <= self.p_vars[p_value], name=f'running_helper1_{self.name}_{p_value}_{t}_p')
                
        #         model.addConstr(
        #             self.isrunning[p_value, t] >= (self.running_helper1[p_value, t] + self.running_helper2[p_value, t]) / 2 - 0.6,
        #             name=f'isrunning_{self.name}_{p_value}_{t}'
        #         )
        

    def __str__(self) -> str:
        parts = [
            f"{COLOR_BLUE}{self.name}{COLOR_RESET} batch_size: {int(self.batch_size)}"
        ]
        p_selected = [p_value for p_value, var in self.p_vars.items() if var.X > 0.5]
        if p_selected:
            parts.append(f" GPU Utilization (p): {p_selected[0]}")
            parts.append(f" Duration: {self.duration_map.get((int(self.batch_size), p_selected[0]))}")
        parts.extend(
            f"{COLOR_GREEN}   -> {dep.name}{COLOR_RESET}" for dep in self.depend_on_nano_ops_logical
        )
        parts.extend(
            f"{COLOR_YELLOW}   -> {dep.name}{COLOR_RESET}" for dep in self.depend_on_nano_ops_stream
        )
        return "\n".join(parts)
    
    def apply_current_as_start(self):
        for p_value in self.p_choices:
            self.p_vars[p_value].Start = self.saved_p_var_x[p_value]