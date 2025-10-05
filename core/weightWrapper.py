import torch

class WeightWrapper:
    def __init__(self, owner=None, name=None):
        self.owner = owner
        self.name = name
        self.full_name = None
        self.weight_map: dict[int, torch.Tensor] = {}
        self.shape = None

