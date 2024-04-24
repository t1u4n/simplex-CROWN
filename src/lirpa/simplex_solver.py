import torch
import torch.nn as nn
import math
from auto_lirpa import AutoLirpa
from utils import create_final_coeffs_slice
from linear_op import LinearOp

class SimplexSolver():
    def __init__(self, layers, max_batch=20000):
        self.layers = layers
        self.bound_solver = self.auto_lirpa_solver()
        self.max_batch = max_batch
        self.net = nn.Sequential(*layers)
        for param in self.net.parameters():
            param.requires_grad = False
    
    def auto_lirpa_solver(self):
        def solver(*args, **kwargs):
            auto_lirpa_object = AutoLirpa(*args, **kwargs)
            bounds_auto_lirpa = auto_lirpa_object.auto_lirpa_optimizer(*args, **kwargs)
            return bounds_auto_lirpa
        return solver
    
    def compute_bounds(self, x, eps):
        lb, ub, alpha, first_layer = self.get_preact_bounds_first_layer(x, eps, self.layers[0])
        self.lbs = [-torch.ones(*x.shape[:-1], 2*x.shape[-1]), lb]
        self.ubs = [torch.ones(*x.shape[:-1], 2*x.shape[-1]), ub]
        self.layer_ops = [first_layer]

        for i, layer in enumerate(self.layers[1:]):
            should_condition = i != len(self.layers)-2
            if isinstance(layer, nn.Linear):
                alpha, obj_layer = self.build_simp_layer(layer, alpha, should_condition)
                self.layer_ops.append(obj_layer)
                lb, ub = self.full_bound_propagation(self.layer_ops, self.lbs, self.ubs)
                assert (ub - lb).min() >= 0, "Incompatible bounds"
                self.lbs.append(lb)
                self.ubs.append(ub)

        return (lb, ub)
    
    @staticmethod
    def get_preact_bounds_first_layer(X, eps, layer):
        W = layer.weight
        b = layer.bias
        x = X.view(X.shape[0], -1)
        dim = x.shape[1]
        E = torch.zeros(dim, 2*dim)
        for i in range(0,dim):
            E[i, 2*i] = 1
            E[i, 2*i+1] = -1

        alpha = None
        if isinstance(layer, nn.Linear):
            # STEP-1- Conditioning this layer so that input lies in simplex
            cond_w_1 = eps * W @ E
            if x.dim() ==2:
                x = x.squeeze(0)
            cond_b_1 = b + x @ W.t()

            # STEP-2- construct LinearOp
            cond_layer = LinearOp(cond_w_1, cond_b_1)

            alpha = cond_layer.simplex_conditioning()
            
            # STEP-4- calculating pre-activation bounds of conditioned layer
            W_min, _ = torch.min(cond_layer.weights.squeeze(0), 1)
            W_min = torch.clamp(W_min, None, 0)
            W_max, _ = torch.max(cond_layer.weights.squeeze(0), 1)
            W_max = torch.clamp(W_max, 0, None)
            l_1 = W_min + cond_layer.bias
            u_1 = W_max + cond_layer.bias
            l_1 = l_1.unsqueeze(0)
            u_1 = u_1.unsqueeze(0)
            assert (u_1 - l_1).min() >= 0, "Incompatible bounds"

        return l_1, u_1, alpha, cond_layer
    
    @staticmethod
    def build_simp_layer(layer, prev_alpha, conditioning=True):
        W = layer.weight
        b = layer.bias

        alpha = None
        if isinstance(layer, nn.Linear):
            W = (W) * prev_alpha
            obj_layer = LinearOp(W, b)
            if conditioning:
                alpha = obj_layer.simplex_conditioning()

        return alpha, obj_layer
    
    def full_bound_propagation(self, layers, lbs, ubs):
        ini_lbs, ini_ubs = layers[-1].interval_forward(torch.clamp(lbs[-1], 0, None),
                                                        torch.clamp(ubs[-1], 0, None))
        out_shape = ini_lbs.shape[1:]#this is the actual output shape
        nb_out = math.prod(out_shape)#number of output neurons of that layer.
        batch_size = ini_lbs.shape[0]

        # if the resulting batch size from parallelizing over the output neurons boundings is too large, we need
        # to divide into sub-batches
        neuron_batch_size = nb_out * 2
        c_batch_size = int(math.floor(self.max_batch / batch_size))
        n_batches = int(math.ceil(neuron_batch_size / float(c_batch_size)))
        bound = None
        for sub_batch_idx in range(n_batches):
            # compute intermediate bounds on sub-batch
            start_batch_index = sub_batch_idx * c_batch_size
            end_batch_index = min((sub_batch_idx + 1) * c_batch_size, neuron_batch_size)
            subbatch_coeffs = create_final_coeffs_slice(
                start_batch_index, end_batch_index, batch_size, nb_out, ini_lbs, out_shape)
            #subbatch_coeffs is of size (batch_size*output size), each rows stores 1 or -1 for which
            #output neuron this batch corresponds to.
            additional_coeffs = {len(lbs): subbatch_coeffs}
            c_bound = self.bound_solver(layers, additional_coeffs, lbs, ubs)
            bound = c_bound if bound is None else torch.cat([bound, c_bound], 1)

        ubs = -bound[:, :nb_out]
        lbs = bound[:, nb_out:]
        lbs = lbs.view(batch_size, *out_shape)
        ubs = ubs.view(batch_size, *out_shape)

        ubs = torch.where((ubs - lbs <= 0) & (ubs - lbs >= -1e-5), lbs + 1e-5, ubs)
        assert (ubs - lbs).min() >= 0, "Incompatible bounds"

        return lbs, ubs
    
