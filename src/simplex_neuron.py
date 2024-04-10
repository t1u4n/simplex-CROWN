import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import Optional

class SimplexNeuron(nn.Module):
    def __init__(self, weight: Tensor, bias: Optional[Tensor]) -> None:
        super().__init__()
        self.weight = deepcopy(weight)
        self.bias = deepcopy(bias) if bias is not None else None
    
    def forward(self, x: Tensor) -> Tensor:
        return F.relu(F.linear(x, self.weight, self.bias))
            