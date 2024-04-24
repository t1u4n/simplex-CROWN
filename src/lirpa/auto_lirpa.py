import torch
from copy import deepcopy
from utils import bdot
from torch.optim import Adam

class AutoLirpa():
    """
    This class implements the autolirpa method using backward fashion in primal space.
    """

    def __init__(self, layers, coeffs, lb, ub):
        """
        The object stores the lirpa coefficients lower_a and upper_a corresponding to the upper and lower bounds.
        """
        self.lower_a = [(ub[i] >= torch.abs(lb[i])).float().requires_grad_(True) for i in range(len(layers))]
        self.upper_a = [torch.ones_like(ub[i], requires_grad=True) for i in range(len(layers))]

        def _find_layer_with_coeffs(n_layers, additional_coeffs):
            for i in range(n_layers, 0, -1):
                if i in additional_coeffs:
                    return i, additional_coeffs[i]
            return None, None

        start_node, C = _find_layer_with_coeffs(len(layers), coeffs)
        if start_node is not None:
            start_node -= 1

        self.start_node = start_node
        self.C = deepcopy(C)
        self.optimizer = Adam(self.lower_a + self.upper_a, lr=1e-4)

    def get_bound_dp_lirpa_backward(self, layers, lbs, ubs):
        """
        This function is used to do a dp lirpa backward pass and get the bounds with current coefficients lower_a and upper_a.
        """
        # compute L(x, a)
        C = deepcopy(self.C)

        bias = bounds_A = None
        with torch.enable_grad():
            for i in range(self.start_node, 0, -1):
            # while start_node > 0:
                layer = layers[i]
                if i == self.start_node:
                    bias = layer.bias_backward(C)#rho is of size (batch_size*output_size)
                else:
                    bias += layer.convex_hull_bias_backward(convex_hull_A) + layer.bias_backward(crown_A)

                if i == self.start_node:
                    bounds_A = layer.backward(C)#rho is of size (batch_size*output_size)
                else:
                    bounds_A = layer.convex_hull_backward(convex_hull_A) + layer.backward(crown_A)
                
                lb = lbs[i]#this is of input_size as they are lower_bounds on that layer input
                ub = ubs[i]
                a_l = self.lower_a[i]
                a_u = self.upper_a[i]

                # beta
                beta_u = -  (a_u * (lb*ub)) / (ub - lb)
                beta_u.masked_fill_(lb > 0, 0)
                beta_u.masked_fill_(ub <= 0, 0)
                beta_l = torch.zeros_like(lb)

                # this can be implemented as a convolution
                bias += bdot(torch.where(bounds_A >= 0, beta_l.unsqueeze(1), beta_u.unsqueeze(1)), bounds_A)

                # alpha
                alpha_u_crown = a_u * ub / (ub - lb)
                alpha_u_crown.masked_fill_(lb > 0, 1)
                alpha_u_crown.masked_fill_(ub <= 0, 0)

                alpha_u_convex_hull = 1-a_u
                alpha_u_convex_hull.masked_fill_(lb > 0, 0)
                alpha_u_convex_hull.masked_fill_(ub <= 0, 0)

                alpha_l = a_l
                with torch.no_grad():
                    alpha_l.masked_fill_(lb > 0, 1)
                    alpha_l.masked_fill_(ub <= 0, 0)

                crown_A = torch.where(bounds_A >= 0, alpha_l.unsqueeze(1), alpha_u_crown.unsqueeze(1)) * bounds_A#(output(batch_size modulo)*input shape)
                convex_hull_A = torch.where(bounds_A >= 0, torch.zeros_like(ub).unsqueeze(1), alpha_u_convex_hull.unsqueeze(1)) * bounds_A
            bound = self.bound_propagate(layers[0], bias, crown_A, convex_hull_A)

        return bound
    
    def bound_propagate(self, layer, bias, crown_A, convex_hull_A):
        with torch.enable_grad():
            bias += layer.convex_hull_bias_backward(convex_hull_A) + layer.bias_backward(crown_A)
            bounds_A = layer.convex_hull_backward(convex_hull_A) + layer.backward(crown_A)
            b, _ = torch.min(bounds_A, 2)
            bound = bias + torch.clamp(b, None, 0)
        return bound

    def auto_lirpa_optimizer(self, layers, _, lbs, ubs):
        """
        # 1. Compute L(x, a)
        # 2. Compute dL/da 
        # 3. optimize a's using Adam.
        """
        num_epochs = 20
        for _ in range(num_epochs):
            # 1. Compute L(x, a)
            bound = self.get_bound_dp_lirpa_backward(layers, lbs, ubs)
            # 2. Compute dL/da 
            with torch.enable_grad():
                bound_mean = bound.mean()
            bound_mean.backward()
            # 3. Adam for subgradient ascent
            self.optimizer.step()
            self.optimizer.zero_grad()
        return bound