import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class SimplexNeuron(nn.Module):
    def __init__(self, weight: Tensor, bias: Optional[Tensor]) -> None:
        super().__init__()
        self.weight = weight.clone()
        self.bias = bias.clone() if bias is not None else None
        self.simplex_coefficient = torch.relu(self.weight + self.bias.unsqueeze(1)) - \
                                   torch.relu(self.bias.unsqueeze(1))
        self.preact_lb = None
        self.preact_ub = None
    
    def forward(self, x: Tensor) -> Tensor:
        return F.relu(F.linear(x, self.weight, self.bias))
            
class BoundSimplexNeuron(SimplexNeuron):
    def __init__(self, weight: Tensor, bias: Optional[Tensor], preact_lb: Tensor, preact_ub: Tensor) -> None:
        super(BoundSimplexNeuron, self).__init__(weight, bias)
        self.preact_lb = preact_lb
        self.preact_ub = preact_ub

    @staticmethod
    def convert(simplex_neuron: SimplexNeuron) -> 'BoundSimplexNeuron':
        assert isinstance(simplex_neuron, SimplexNeuron), "Input must be an instance of `SimplexNeuron`"
        assert simplex_neuron.preact_lb is not None and simplex_neuron.preact_ub is not None, \
            "Must have preact bounds before converting"
        layer = BoundSimplexNeuron(
            simplex_neuron.weight, simplex_neuron.bias, simplex_neuron.preact_lb, simplex_neuron.preact_ub)
        return layer
    
    def boundpropogate(self, last_uA: Optional[Tensor], last_lA: Optional[Tensor], start_node: Optional[int]=None):
        assert last_lA is None, "Does not support computing for lower bounds"
        uA = None
        ubias = 0

        lb_r = self.preact_lb
        lb_r_unsq = lb_r.unsqueeze(-1)
        ub_r = self.preact_ub
        ub_r_unsq = ub_r.unsqueeze(-1)

        # Compute coefficient for upper bound
        upper_coeff = torch.where(ub_r_unsq <= 0, 0, torch.where(lb_r_unsq >= 0, self.weight, self.simplex_coefficient))
        upper_bias = torch.where(ub_r <= 0, 0, torch.where(lb_r >= 0, self.bias, self.bias.clamp(min=0)))

        # Compute coefficient for lower bound
        lower_d = (torch.abs(lb_r) >= torch.abs(ub_r)).float()
        lower_d_unsq = (torch.abs(lb_r_unsq) >= torch.abs(ub_r_unsq)).float()
        lower_coeff = torch.where(ub_r_unsq <= 0, 0, torch.where(lb_r_unsq >= 0, self.weight, lower_d_unsq * self.weight))
        lower_bias = torch.where(ub_r <= 0, 0, torch.where(lb_r >= 0, self.bias, lower_d * self.bias))


        if last_uA is not None:
            pos_uA, neg_uA = last_uA.clamp(min=0), last_uA.clamp(max=0)
            uA = pos_uA.matmul(upper_coeff) + neg_uA.matmul(lower_coeff)
            ubias = pos_uA.matmul(upper_bias.squeeze(0)) + neg_uA.matmul(lower_bias.squeeze(0))
        return uA, ubias, None, 0
