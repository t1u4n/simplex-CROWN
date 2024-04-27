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

class BoundSimplexNeuron_Alpha(SimplexNeuron):
    def __init__(self, weight: Tensor, bias: Optional[Tensor], preact_lb: Tensor, preact_ub: Tensor) -> None:
        super(BoundSimplexNeuron_Alpha, self).__init__(weight, bias)
        self.preact_lb = preact_lb
        self.preact_ub = preact_ub
        self.alpha = torch.ones_like(self.preact_lb).unsqueeze(-1).requires_grad_(True)

    @staticmethod
    def convert(simplex_neuron: SimplexNeuron) -> 'BoundSimplexNeuron_Alpha':
        assert isinstance(simplex_neuron, SimplexNeuron), "Input must be an instance of `SimplexNeuron`"
        assert simplex_neuron.preact_lb is not None and simplex_neuron.preact_ub is not None, \
            "Must have preact bounds before converting"
        layer = BoundSimplexNeuron_Alpha(
            simplex_neuron.weight, simplex_neuron.bias, simplex_neuron.preact_lb, simplex_neuron.preact_ub)
        return layer

    def boundpropogate(self, last_uA: Optional[Tensor], last_lA: Optional[Tensor], start_node: Optional[int]=None):
        assert last_lA is None, "Does not support computing for lower bounds"
        simplex_uA, simplex_ubias = self._boundpropogate_simplex(last_uA)
        crown_uA, crown_ubias = self._boundpropogate_crown(last_uA)
        uA = simplex_uA + crown_uA
        ubias = simplex_ubias + crown_ubias
        return uA, ubias, None, 0

    def _boundpropogate_crown(self, last_uA):
        # lb_r and ub_r are the bounds of input (pre-activation)
        # Here the clamping oepration ensures the results are correct for stable neurons.
        lb_r = self.preact_lb.clamp(max=0)
        ub_r = self.preact_ub.clamp(min=0)
        # avoid division by 0 when both lb_r and ub_r are 0
        ub_r = torch.max(ub_r, lb_r + 1e-8)

        # CROWN upper and lower linear bounds
        upper_d = ub_r / (ub_r - lb_r)  # slope
        upper_b = - lb_r * upper_d  # intercept
        upper_d = upper_d.unsqueeze(1)

        # Lower bound: 0 if |lb| < |ub|, 1 otherwise.
        # Equivalently we check whether the slope of the upper bound is > 0.5.
        lower_d = (upper_d > 0.5).float()

        crown_uA = None
        crown_ubias = 0

        w = self.alpha * self.weight
        b = self.alpha.squeeze(-1) * self.bias
        if last_uA is not None:
            pos_uA = last_uA.clamp(min=0)
            neg_uA = last_uA.clamp(max=0)
            # Choose upper or lower bounds based on the sign of last_A
            # New linear bound coefficent.
            crown_uA = upper_d * pos_uA + lower_d * neg_uA
            # New bias term. Adjust shapes to use matmul (better way is to use einsum).
            mult_uA = pos_uA.view(last_uA.size(0), last_uA.size(1), -1)
            crown_ubias = mult_uA.matmul(upper_b.view(upper_b.size(0), -1, 1)).squeeze(-1)

            # propagate A to the nest layer
            next_A = crown_uA.matmul(w)
            # compute the bias of this layer
            sum_bias = crown_uA.matmul(b.squeeze(0))

            crown_uA = next_A
            crown_ubias += sum_bias
        
        return crown_uA, crown_ubias

    def _boundpropogate_simplex(self, last_uA):
        # Simplex uA, ubias
        simplex_uA = None
        simplex_ubias = 0

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

        upper_coeff = (1 - self.alpha) * upper_coeff
        upper_bias = (1 - self.alpha.squeeze(-1)) * upper_bias
        lower_coeff = (1 - self.alpha) * lower_coeff
        lower_bias = (1 - self.alpha.squeeze(-1)) * lower_bias

        if last_uA is not None:
            pos_uA, neg_uA = last_uA.clamp(min=0), last_uA.clamp(max=0)
            simplex_uA = pos_uA.matmul(upper_coeff) + neg_uA.matmul(lower_coeff)
            simplex_ubias = pos_uA.matmul(upper_bias.squeeze(0)) + neg_uA.matmul(lower_bias.squeeze(0))
        
        return simplex_uA, simplex_ubias