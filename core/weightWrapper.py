import torch

class WeightWrapper:
    def __init__(self, owner=None):
        self.owner = owner
        self.name = None
        self.full_name = None
        self.weight_map: dict[int, torch.Tensor] = {}
        self.shape = None

