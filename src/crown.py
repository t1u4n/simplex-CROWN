import torch
import torch.nn as nn
import numpy as np
from linear import BoundLinear
from relu import BoundReLU
from simplex_neuron import SimplexNeuron, BoundSimplexNeuron
from simplex_propagation import simplex_propagation, simplex_propagation_orig
import time
import argparse


class BoundedSequential(nn.Sequential):
    r"""This class wraps the PyTorch nn.Sequential object with bound computation."""

    def __init__(self, *args):
        super(BoundedSequential, self).__init__(*args)

    @staticmethod
    def convert(seq_model):
        r"""Convert a Pytorch model to a model with bounds.
        Args:
            seq_model: An nn.Sequential module.

        Returns:
            The converted BoundedSequential module.
        """
        layers = []
        for l in seq_model:
            if isinstance(l, nn.Linear):
                layers.append(BoundLinear.convert(l))
            elif isinstance(l, nn.ReLU):
                layers.append(BoundReLU.convert(l))
            elif isinstance(l, SimplexNeuron):
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


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--activation', default='relu', choices=['relu', 'hardtanh'],
                        type=str, help='Activation Function')
    parser.add_argument('data_file', type=str, help='input data, a tensor saved as a .pth file.')
    # Parse the command line arguments
    args = parser.parse_args()

    x_test, label = torch.load(args.data_file)

    if args.activation == 'relu':
        print('use ReLU model')
        model = nn.Sequential(
                nn.Linear(28*28, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
            )
        model.load_state_dict(torch.load('models/relu_model.pth'))
        model.eval()

    batch_size = x_test.size(0)
    x_test = x_test.reshape(batch_size, -1)
    output = model(x_test)
    y_size = output.size(1)
    print("Network prediction: {}".format(output))

    eps = 0.03
    def print_bounds(lb, ub, batch_sz, y_sz):
        for i in range(batch_sz):
            for j in range(y_sz):
                print('f_{j}(x_{i}): {l:8.4f} <= f_{j}(x_{i}+delta) <= {u:8.4f}'.format(
                    j=j, i=i, l=lb[i][j].item(), u=ub[i][j].item()))

    print(f"Verifiying Pertubation - {eps}")

    print("----------------------------------------------------")

    """Auto_LiRPA method"""
    print("Auto_LiRPA CROWN method:")
    from auto_LiRPA.perturbations import PerturbationLpNorm
    from auto_LiRPA import BoundedModule, BoundedTensor
    import warnings
    warnings.filterwarnings('ignore')

    ptb = PerturbationLpNorm(eps=eps, norm=1., x_L=torch.zeros_like(x_test).float(), x_U=torch.ones_like(x_test).float())
    bounded_x = BoundedTensor(x_test, ptb)
    auto_lirpa_bounded_model = BoundedModule(model, torch.zeros_like(x_test))
    auto_lirpa_bounded_model.eval()
    with torch.no_grad():
        lirpa_lb, lirpa_ub = auto_lirpa_bounded_model.compute_bounds(x=(bounded_x,), method='CROWN')
    print_bounds(lirpa_lb, lirpa_ub, batch_size, y_size)
    print("----------------------------------------------------")
    
    """Original model using CROWN"""
    print("CROWN on original model")
    boundedmodel = BoundedSequential.convert(model)
    orig_crown_ub, orig_crown_lb = boundedmodel.compute_bounds(x=x_test, eps=eps)
    print_bounds(orig_crown_lb, orig_crown_ub, batch_size, y_size)
    print("----------------------------------------------------")

    """Converted model using CROWN"""
    print("CROWN on converted model")
    new_model = simplex_propagation_orig(model, x_test, eps)

    x = torch.zeros(1, 2*28*28)

    boundedmodel = BoundedSequential.convert(new_model)
    converted_crown_ub, converted_crown_lb = boundedmodel.compute_bounds(x=x, upper=True, lower=True, eps=1.)
    print_bounds(converted_crown_lb, converted_crown_ub, batch_size, y_size)

    # Verify converted model does not modify result and our own implemented l1 norm crown is same as auto_LiRPA's
    # implementation.
    assert torch.allclose(lirpa_lb, orig_crown_lb), "Our own implemented l1 norm CROWN has issue (lower bound)"
    assert torch.allclose(orig_crown_lb, converted_crown_lb), "Converted model modify results on CROWN (lower bound)"
    assert torch.allclose(lirpa_ub, orig_crown_ub), "Our own implemented l1 norm CROWN has issue (upper bound)"
    assert torch.allclose(orig_crown_ub, converted_crown_ub), "Converted model modify results on CROWN (upper bound)"

    print("----------------------------------------------------")

    """Simplex method"""
    print("Simplex method:")
    new_model = simplex_propagation(model, x_test, eps)

    x = torch.zeros(1, 2*28*28)

    boundedmodel = BoundedSequential.convert(new_model)
    simplex_ub, _ = boundedmodel.compute_bounds(x=x, upper=True, lower=False, eps=1.)
    print_bounds(converted_crown_lb, simplex_ub, batch_size, y_size)