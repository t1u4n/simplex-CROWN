import torch
import torch.nn as nn
import numpy as np
from linear import BoundLinear
from relu import BoundReLU
from simplex_neuron import SimplexNeuron, BoundSimplexNeuron, BoundSimplexNeuron_Alpha


class BoundedSequential(nn.Sequential):
    r"""This class wraps the PyTorch nn.Sequential object with bound computation."""

    def __init__(self, *args):
        super(BoundedSequential, self).__init__(*args)

    @staticmethod
    def convert(seq_model, method='alpha-simplex'):
        r"""Convert a Pytorch model to a model with bounds.
        Args:
            seq_model: An nn.Sequential module.

        Returns:
            The converted BoundedSequential module.
        """
        assert method in ['simplex', 'alpha-simplex']
        layers = []
        for l in seq_model:
            if isinstance(l, nn.Linear):
                layers.append(BoundLinear.convert(l))
            elif isinstance(l, nn.ReLU):
                layers.append(BoundReLU.convert(l))
            elif isinstance(l, SimplexNeuron):
                if method == 'alpha-simplex':
                    layers.append(BoundSimplexNeuron_Alpha.convert(l))
                elif method == 'simplex':
                    layers.append(BoundSimplexNeuron.convert(l))
        return BoundedSequential(*layers)

    def compute_bounds(self, x=None, upper=True, lower=True, optimize=False, eps=1.):
        r"""Main function for computing bounds.

        Args:
            x_U (tensor): The upper bound of x.

            x_L (tensor): The lower bound of x.

            upper (bool): Whether we want upper bound.

            lower (bool): Whether we want lower bound.

            optimize (bool): Whether we optimize alpha.

        Returns:
            ub (tensor): The upper bound of the final output.

            lb (tensor): The lower bound of the final output.
        """
        ub = lb = None
        ub, lb = self.full_boundpropogation(x=x, upper=upper, lower=lower, eps=eps)
        return ub, lb

    def full_boundpropogation(self, x=None, upper=True, lower=True, eps=1.):
        r"""A full bound propagation. We are going to sequentially compute the
        intermediate bounds for each linear layer followed by a ReLU layer. For each
        intermediate bound, we call self.boundpropogate_from_layer() to do a bound propagation
        starting from that layer.

        Args:
            x_U (tensor): The upper bound of x.

            x_L (tensor): The lower bound of x.

            upper (bool): Whether we want upper bound.

            lower (bool): Whether we want lower bound.

        Returns:
            ub (tensor): The upper bound of the final output.

            lb (tensor): The lower bound of the final output.
        """
        modules = list(self._modules.values())
        # CROWN propagation for all layers
        for i in range(len(modules)):
            # We only need the bounds before a ReLU layer
            if isinstance(modules[i], BoundReLU):
                if isinstance(modules[i - 1], BoundLinear):
                    # add a batch dimension
                    newC = torch.eye(modules[i - 1].out_features).unsqueeze(0).repeat(x.shape[0], 1, 1).to(x)
                    # Use CROWN to compute pre-activation bounds
                    # starting from layer i-1
                    ub, lb = self.boundpropogate_from_layer(x=x, C=newC, upper=True, lower=True,
                                                            start_node=i - 1, eps=eps)
                # Set pre-activation bounds for layer i (the ReLU layer)
                modules[i].upper_u = ub
                modules[i].lower_l = lb
        # Get the final layer bound
        return self.boundpropogate_from_layer(x=x,
                                              C=torch.eye(modules[i].out_features).unsqueeze(0).to(x), upper=upper,
                                              lower=lower, start_node=i, eps=eps)

    def boundpropogate_from_layer(self, x=None, C=None, upper=False, lower=True, start_node=None, eps=1.):
        r"""The bound propagation starting from a given layer. Can be used to compute intermediate bounds or the final bound.

        Args:
            x_U (tensor): The upper bound of x.

            x_L (tensor): The lower bound of x.

            C (tensor): The initial coefficient matrix. Can be used to represent the output constraints.
            But we don't have any constraints here. So it's just an identity matrix.

            upper (bool): Whether we want upper bound.

            lower (bool): Whether we want lower bound.

            start_node (int): The start node of this propagation. It should be a linear layer.
        Returns:
            ub (tensor): The upper bound of the output of start_node.
            lb (tensor): The lower bound of the output of start_node.
        """
        modules = list(self._modules.values()) if start_node is None else list(self._modules.values())[:start_node + 1]
        upper_A = C if upper else None
        lower_A = C if lower else None
        upper_sum_b = lower_sum_b = x.new([0])
        for i, module in enumerate(reversed(modules)):
            upper_A, upper_b, lower_A, lower_b = module.boundpropogate(upper_A, lower_A, start_node)
            upper_sum_b = upper_b + upper_sum_b
            lower_sum_b = lower_b + lower_sum_b

        # sign = +1: upper bound, sign = -1: lower bound
        def _get_concrete_bound(A: torch.Tensor, sum_b, x, sign=-1):
            if A is None:
                return None
            A = A.view(A.size(0), A.size(1), -1)
            # A has shape (batch, specification_size, flattened_input_size)
            x = x.reshape(x.shape[0], -1, 1)
            deviation = A.norm(np.inf, -1) * eps
            bound = A.matmul(x) + sign * deviation.unsqueeze(-1)
            bound = bound.squeeze(-1) + sum_b
            return bound

        lb = _get_concrete_bound(lower_A, lower_sum_b, x, sign=-1)
        ub = _get_concrete_bound(upper_A, upper_sum_b, x, sign=+1)
        if ub is None:
            ub = x.new([np.inf])
        if lb is None:
            lb = x.new([-np.inf])
        return ub, lb