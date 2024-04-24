import torch
import torch.nn as nn
import time, math
from auto_lirpa import AutoLirpa
from utils import create_final_coeffs_slice
import numpy as np
from linear_op import LinearOp

class SimplexSolver():
    '''
    The objects of this class are s.t: the input lies in l1.
    1. the first layer is conditioned s.t that the input lies in a probability simplex.
    2. Simplex Propagation: the layers(including first layer, excluding output linear layer) are conditioned s.t their output also lies in simplex, this is done using the simplex cut.
    3. better ib bounds are computed using these propagated simplices

    return: so the define_linear_approximation fn returns a net whose input lies in simplex and all other intermediate layers lie in a simplex too
    '''

    def __init__(self, layers, max_batch=20000, seed=0, dp=True):
        """
        :param store_bounds_progress: whether to store bounds progress over time (-1=False 0=True)
        :param store_bounds_primal: whether to store the primal solution used to compute the final bounds
        :param max_batch: maximal batch size for parallel bounding çomputations over both output neurons and domains
        """
        self.layers = layers
        self.net = nn.Sequential(*layers)

        for param in self.net.parameters():
            param.requires_grad = False

        self.optimize = self.best_naive_dp_optimizer()

        self.max_batch = max_batch

        self.init_cut_coeffs = []
        self.seed = seed
        self.dp = dp
    
    def best_naive_dp_optimizer(self):
        # best bounds out of kw and naive interval propagation

        def optimize(*args, **kwargs):
            auto_lirpa_object = AutoLirpa(*args, **kwargs)

            bounds_auto_lirpa = auto_lirpa_object.auto_lirpa_optimizer(*args, **kwargs)

            return bounds_auto_lirpa

        return optimize
    
    def define_linear_approximation(self, x, eps):
        '''
        this function computes intermediate bounds and stores them into self.lower_bounds and self.upper_bounds.
        It also stores the network weights into self.weights.
        Now this function will compute these bounds and then condition the layers such that the input constraints are simplex.

        no_conv is an option to operate only on linear layers, by transforming all
        the convolutional layers into equivalent linear layers.
        lower_bounds [input_bounds,1st_layer_output,2nd_layeroutput ....]
        '''
        conditioning = True
        self.layer_ops = []

        for i, layer in enumerate(self.layers):
            if i == len(self.layers)-1:
                conditioning=False

            if i == 0:
                l_1, u_1, init_cut_coeff, first_layer = self.get_preact_bounds_first_layer(x, eps, layer, conditioning = conditioning, seed=self.seed)
                if init_cut_coeff is not None:
                    self.init_cut_coeffs.append(init_cut_coeff)
                
                self.lbs = [-torch.ones(*x.shape[:-1], 2*x.shape[-1]), l_1]
                self.ubs = [torch.ones(*x.shape[:-1], 2*x.shape[-1]), u_1]
                
                self.layer_ops.append(first_layer)


            elif isinstance(layer, nn.Linear):
                init_cut_coeff, obj_layer = self.build_simp_layer(layer, self.init_cut_coeffs, conditioning = conditioning, seed=self.seed)
                self.layer_ops.append(obj_layer)

                lb, ub = self.solve_problem(self.layer_ops, self.lbs, self.ubs)

                assert (ub - lb).min() >= 0, "Incompatible bounds"

                if i != len(self.layers)-1:
                    self.init_cut_coeffs.append(init_cut_coeff)
                    
                self.lbs.append(lb)
                self.ubs.append(ub)
    
    @staticmethod
    def get_preact_bounds_first_layer(X, eps, layer, conditioning=True, seed=0):
        '''
        This function does 4 jobs:
        1st is to condition the first layer from l1 to simplex
        2nd 2a: it gets the init_cut_coeff. this is the b in lambda x leq b kind of cuts, for the special init in which the lamba is 1
        2b: then it conditions the layer with this coeff, so now the output also lies in simplex
        3rd it constructs the batch conv or batch conv linear layer.
        4th is to get the preact bounds of the first layer
        '''
        w_1 = layer.weight
        b_1 = layer.bias
        X_vector = X.view(X.shape[0], -1)
        dim = X_vector.shape[1]
        E = torch.zeros(dim, 2*dim)
        for row_idx in range(0,dim):
            E[row_idx, 2*row_idx] = 1
            E[row_idx, 2*row_idx+1] = -1

        init_cut_coeff = None
        if isinstance(layer, nn.Linear):
            # STEP-1- Conditioning this layer so that input lies in simplex
            cond_w_1 = eps * w_1 @ E
            if X_vector.dim() ==2:
                X_vector = X_vector.squeeze(0)
            cond_b_1 = b_1 + X_vector @ w_1.t()

            # STEP-2- construct LinearOp
            cond_layer = LinearOp(cond_w_1, cond_b_1)

            if conditioning:
                if seed!=0:
                    a = np.random.random(b_1.shape[0])
                    a /= a.sum()
                    lmbd=torch.from_numpy(a).float().to(cond_w_1.device)
                else:
                    lmbd=torch.ones(b_1.shape[0]).float().to(cond_w_1.device)
                init_cut_coeff = cond_layer.simplex_conditioning(lmbd, conditioning=True)
            
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

        return l_1, u_1, init_cut_coeff, cond_layer
    
    @staticmethod
    def build_simp_layer(layer, init_cut_coeffs, conditioning=True, seed=0):
        '''
        This function also does the conditioning using init_cut_coeff
        This function also calculates the init_cut_coeff which is the b coefficient for lambda x leq b cut
        This function return a ConvOp or LinearOp object depending on the layer type. 
        This function also computes the pre-activation bounds
        '''
        w_kp1 = layer.weight
        b_kp1 = layer.bias

        init_cut_coeff = None
        lmbd = None

        l_kp1 = None
        u_kp1 = None

        if isinstance(layer, nn.Linear):
            w_kp1 = (w_kp1) * init_cut_coeffs[-1]
            obj_layer = LinearOp(w_kp1, b_kp1)
            if conditioning:
                if seed!=0:
                    a = np.random.random(b_kp1.shape[0])
                    a /= a.sum()
                    lmbd=torch.from_numpy(a).float().to(w_kp1.device)
                else:
                    lmbd=torch.ones(b_kp1.shape[0]).float().to(w_kp1.device)

                init_cut_coeff = obj_layer.simplex_conditioning(lmbd, conditioning=True)

        return init_cut_coeff, obj_layer
    
    def solve_problem(self, layers, lower_bounds, upper_bounds):
        '''
        Compute bounds on the last layer of the problem. (it will compute 2*number of output neurons bounds.)
        With batchification, we need to optimize over all layers in any case, as otherwise the tensors of different
         sizes should be kept as a list (slow)
        '''
        ini_lbs, ini_ubs = layers[-1].interval_forward(torch.clamp(lower_bounds[-1], 0, None),
                                                        torch.clamp(upper_bounds[-1], 0, None))
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
            additional_coeffs = {len(lower_bounds): subbatch_coeffs}
            c_bound = self.optimize(layers, additional_coeffs, lower_bounds, upper_bounds)
            bound = c_bound if bound is None else torch.cat([bound, c_bound], 1)

        ubs = -bound[:, :nb_out]
        lbs = bound[:, nb_out:]
        lbs = lbs.view(batch_size, *out_shape)
        ubs = ubs.view(batch_size, *out_shape)

        ubs = torch.where((ubs - lbs <= 0) & (ubs - lbs >= -1e-5), lbs + 1e-5, ubs)
        assert (ubs - lbs).min() >= 0, "Incompatible bounds"

        return lbs, ubs
    
