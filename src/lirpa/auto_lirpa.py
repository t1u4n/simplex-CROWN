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
        start_node = deepcopy(self.start_node)
        rho_split=False

        bias = None
        lmbda = None

        with torch.enable_grad():
            for i in range(start_node, 0, -1):
            # while start_node > 0:
                layer = layers[i]

                if bias is None:
                    bias = layer.bias_backward(C)#rho is of size (batch_size*output_size)
                else:
                    bias += layer.dp_bias_backward(rho_u_dp) + layer.bias_backward(rho_planet)

                if not rho_split:
                    lmbda = layer.backward(C)#rho is of size (batch_size*output_size)
                else:
                    lmbda = layer.dp_backward(rho_u_dp) + layer.backward(rho_planet)
                
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
                bias += bdot(torch.where(lmbda >= 0, beta_l.unsqueeze(1), beta_u.unsqueeze(1)), lmbda)

                # alpha
                alpha_u_planet = a_u * ub / (ub - lb)
                alpha_u_planet.masked_fill_(lb > 0, 1)
                alpha_u_planet.masked_fill_(ub <= 0, 0)

                alpha_u_dp = 1-a_u
                alpha_u_dp.masked_fill_(lb > 0, 0)
                alpha_u_dp.masked_fill_(ub <= 0, 0)

                alpha_l = a_l
                with torch.no_grad():
                    alpha_l.masked_fill_(lb > 0, 1)
                    alpha_l.masked_fill_(ub <= 0, 0)

                rho_split=True
                rho_planet = torch.where(lmbda >= 0, alpha_l.unsqueeze(1), alpha_u_planet.unsqueeze(1)) * lmbda#(output(batch_size modulo)*input shape)
                rho_u_dp = torch.where(lmbda >= 0, torch.zeros_like(ub).unsqueeze(1), alpha_u_dp.unsqueeze(1)) * lmbda
            bound = self.opt_lirpa_input_dp(layers[0], bias, rho_planet, rho_u_dp)

        return bound
    
    def opt_lirpa_input_dp(self, lay, bias, crown_A, convex_hull_A):
        with torch.enable_grad():
            bias += lay.dp_bias_backward(convex_hull_A) + lay.bias_backward(crown_A)
            lin_eq = lay.dp_backward(convex_hull_A) + lay.backward(crown_A)
            lin_eq_matrix = lin_eq.view(lin_eq.shape[0],lin_eq.shape[1],-1)
            b, _ = torch.min(lin_eq_matrix, 2)
            bound = bias + torch.clamp(b, None, 0)
        return bound

    def auto_lirpa_optimizer(self, layers, _, lbs, ubs):
        """
        # 1. Compute L(x, a)
        # 2. Compute dL/da 
        # 3. optimize a's using Adam.
        """
        num_epochs = 50
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