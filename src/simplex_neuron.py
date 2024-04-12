import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import Optional

class SimplexNeuron(nn.Module):
    def __init__(self, weight: Tensor, bias: Optional[Tensor]) -> None:
        super().__init__()
        self.weight = weight.clone()
        self.bias = bias.clone() if bias is not None else None
    
    def forward(self, x: Tensor) -> Tensor:
        return F.relu(F.linear(x, self.weight, self.bias))
            
class BoundSimplexNeuron(SimplexNeuron):
    def __init__(self, weight: Tensor, bias: Optional[Tensor]) -> None:
        super(BoundSimplexNeuron, self).__init__(weight, bias)

    @staticmethod
    def convert(simplex_neuron: SimplexNeuron) -> 'BoundSimplexNeuron':
        assert isinstance(simplex_neuron, SimplexNeuron), "Input must be an instance of `SimplexNeuron`"
        layer = BoundSimplexNeuron(simplex_neuron.weight.clone(), simplex_neuron.bias)
        return layer
    
    def boundpropogate(self, last_uA: Optional[Tensor], last_lA: Optional[Tensor], start_node: Optional[int]=None):
        assert last_lA is None, "Do not support lower bound computing now"
        A = torch.clamp(self.weight + self.bias[:, None], min=0) - torch.clamp(self.bias[:, None], min=0)
        uA = lA = None
        ubias = lbias = 0
        uA = last_uA.matmul(A)
        ubias = last_uA.matmul(torch.clamp(self.bias, min=0))
        return uA, ubias, lA, lbias
