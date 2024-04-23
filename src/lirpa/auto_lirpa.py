import torch
import copy
from utils import bdot
from torch.optim import Adam

class AutoLirpa():
    """
    This class implements the autolirpa method using backward fashion in primal space.
    """

    def __init__(self, weights, additional_coeffs, lower_bounds, upper_bounds):
        """
        The object stores the lirpa coefficients lower_a and upper_a corresponding to the upper and lower bounds.
        """
        self.lower_a = []
        self.upper_a = []
        for i in range(len(weights)):
            self.lower_a.append(torch.ones_like(lower_bounds[i], requires_grad=True))
            self.lower_a[i].grad = None

            self.upper_a.append(torch.ones_like(upper_bounds[i], requires_grad=True))
            self.upper_a[i].grad = None

        #######################
        assert len(additional_coeffs) > 0

        final_lay_idx = len(weights)
        if final_lay_idx in additional_coeffs:
            # There is a coefficient on the output of the network
            rho = additional_coeffs[final_lay_idx]
            lay_idx = final_lay_idx
        else:
            # There is none. Just identify the shape from the additional coeffs
            lay_idx = final_lay_idx -1
            while lay_idx not in additional_coeffs:
                lay_idx -= 1
            # We now reached the time where lay_idx has an additional coefficient
            rho = additional_coeffs[lay_idx]
        lay_idx -= 1

        self.initial_lay_idx = copy.deepcopy(lay_idx)
        self.initial_rho = copy.deepcopy(rho)

        self.optimizer = Adam(self.lower_a + self.upper_a, lr=1e-4)

    def crown_initialization(self, weights, _, lower_bounds, upper_bounds):
        """
        initialized the lower coefficients as per crown
        """
        for i in range(len(weights)):
            self.lower_a[i] = (upper_bounds[i] >= torch.abs(lower_bounds[i])).type(lower_bounds[i].dtype)
            self.lower_a[i].requires_grad = True
            self.lower_a[i].grad = None

    def get_bound_dp_lirpa_backward(self, weights, lower_bounds, upper_bounds):
        """
        This function is used to do a dp lirpa backward pass and get the bounds with current coefficients lower_a and upper_a.
        """

        ##################
        ### compute L(x, a)
        ##################
        b_term = None
        rho = copy.deepcopy(self.initial_rho)
        lay_idx = copy.deepcopy(self.initial_lay_idx)
        rho_split=False

        with torch.enable_grad():
            while lay_idx > 0:
                lay = weights[lay_idx]

                if b_term is None:
                    b_term = lay.bias_backward(rho)#rho is of size (batch_size*output_size)
                else:
                    b_term += lay.dp_bias_backward(rho_u_dp) + lay.bias_backward(rho_planet)

                if not rho_split:
                    lbda = lay.backward(rho)#rho is of size (batch_size*output_size)
                else:
                    lbda = lay.dp_backward(rho_u_dp) + lay.backward(rho_planet)
                
                lbs = lower_bounds[lay_idx]#this is of input_size as they are lower_bounds on that layer input
                ubs = upper_bounds[lay_idx]

                las = self.lower_a[lay_idx]
                uas = self.upper_a[lay_idx]

                #####
                ## beta
                beta_u = -  (uas * (lbs*ubs)) / (ubs - lbs)
                beta_u.masked_fill_(lbs > 0, 0)
                beta_u.masked_fill_(ubs <= 0, 0)

                beta_l = torch.zeros_like(lbs)

                ### POSSIBLE SPEEDUP
                #### this can be implemented as a convolution
                b_term += bdot(torch.where(lbda >= 0, beta_l.unsqueeze(1), beta_u.unsqueeze(1)), lbda)
                #####


                #####
                ## alpha
                alpha_u_planet = uas * ubs / (ubs - lbs)
                alpha_u_planet.masked_fill_(lbs > 0, 1)
                alpha_u_planet.masked_fill_(ubs <= 0, 0)

                alpha_u_dp = 1-uas
                alpha_u_dp.masked_fill_(lbs > 0, 0)
                alpha_u_dp.masked_fill_(ubs <= 0, 0)

                alpha_l = las
                with torch.no_grad():
                    alpha_l.masked_fill_(lbs > 0, 1)
                    alpha_l.masked_fill_(ubs <= 0, 0)

                zeros_ten = torch.zeros_like(ubs)

                rho_split=True
                rho_planet = torch.where(lbda >= 0, alpha_l.unsqueeze(1), alpha_u_planet.unsqueeze(1)) * lbda#(output(batch_size modulo)*input shape)
                rho_u_dp = torch.where(lbda >= 0, zeros_ten.unsqueeze(1), alpha_u_dp.unsqueeze(1)) * lbda

                lay_idx -= 1

            bound = opt_lirpa_input_dp(weights[0], b_term, rho_planet, rho_u_dp)

        return bound

    def auto_lirpa_optimizer(self, weights, _, lower_bounds, upper_bounds):
        """
        # 1. Compute L(x, a)
        # 2. Compute dL/da 
        # 3. optimize a's using Adam.
        """
        n_iters = 20
        for _ in range(n_iters):

            # 1. Compute L(x, a)
            bound = self.get_bound_dp_lirpa_backward(weights, lower_bounds, upper_bounds)
            # 2. Compute dL/da 
            with torch.enable_grad():
                bound_mean = bound.mean()
            bound_mean.backward()
            # 3. Adam for subgradient ascent
            self.optimizer.step()
            self.optimizer.zero_grad()
        return bound

def opt_lirpa_input_dp(lay, b_term, rho_planet, rho_u_dp):
    with torch.enable_grad():
        b_term += lay.dp_bias_backward(rho_u_dp) + lay.bias_backward(rho_planet)
        lin_eq = lay.dp_backward(rho_u_dp)
        lin_eq += lay.backward(rho_planet)
        lin_eq_matrix = lin_eq.view(lin_eq.shape[0],lin_eq.shape[1],-1)
        b, _ = torch.min(lin_eq_matrix, 2)
        bound = b_term + torch.clamp(b, None, 0)
    return bound