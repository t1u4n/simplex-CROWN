import torch
import torch.nn as nn
import copy, time, math
from autolirpa_opt_dp import autolirpa_opt_dp
from auto_lirpa import AutoLirpa
from utils import bdot, get_relu_mask, create_final_coeffs_slice, prod
import numpy as np
from linear_op import LinearOp, BatchLinearOp

default_params = {
    "initial_eta": 100,
    "nb_inner_iter": 100,
    "nb_outer_iter": 10,
    "nb_iter": 500,
    "anderson_algorithm": "saddle",  # either saddle or prox
    "bigm": "init", # whether to use the specialized bigm relaxation solver. Alone: "only". As initializer: "init"
    "bigm_algorithm": "adam",  # either prox or adam
    "init_params": {
        'nb_outer_iter': 100,
        'initial_step_size': 1e-3,
        'final_step_size': 1e-6,
        'betas': (0.9, 0.999)
    }
}

class SimplexLP():
    '''
    The objects of this class are s.t: the input lies in l1.
    1. the first layer is conditioned s.t that the input lies in a probability simplex.
    2. Simplex Propagation: the layers(including first layer, excluding output linear layer) are conditioned s.t their output also lies in simplex, this is done using the simplex cut.
    3. better ib bounds are computed using these propagated simplices

    return: so the define_linear_approximation fn returns a net whose input lies in simplex and all other intermediate layers lie in a simplex too
    '''

    def __init__(self, 
        layers, 
        debug=False, 
        params=None, 
        view_tensorboard=False, 
        precision=torch.float, 
        store_bounds_progress=-1, 
        store_bounds_primal=False, 
        max_batch=20000, 
        seed=0,
        dp=True,
        tgt=1):
        """
        :param store_bounds_progress: whether to store bounds progress over time (-1=False 0=True)
        :param store_bounds_primal: whether to store the primal solution used to compute the final bounds
        :param max_batch: maximal batch size for parallel bounding çomputations over both output neurons and domains
        """
        self.optimizers = {
            'best_naive_dp': self.best_naive_dp_optimizer,
            # 'opt_dp': self.opt_dp_optimizer,
            # 'auto_lirpa_optimizer': self.auto_lirpa_optimizer,
        }

        self.layers = layers
        self.net = nn.Sequential(*layers)

        for param in self.net.parameters():
            param.requires_grad = False

        self.optimize, _ = self.best_naive_dp_optimizer(None)

        self.store_bounds_progress = store_bounds_progress
        self.store_bounds_primal = store_bounds_primal

        # store which relus are ambiguous. 1=passing, 0=blocking, -1=ambiguous. Shape: dom_batch_size x layer_width
        self.relu_mask = []
        self.max_batch = max_batch

        self.debug = debug
        self.view_tensorboard = view_tensorboard
        if self.view_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(comment="bigm")

        self.params = dict(default_params, **params) if params is not None else default_params
        # self.bigm_init = self.params["bigm"] and (self.params["bigm"] == "init")
        # self.bigm_only = self.params["bigm"] and (self.params["bigm"] == "only")
        # self.cut_init = ("cut" in self.params) and self.params["cut"] and (self.params["cut"] == "init")
        # self.cut_only = ("cut" in self.params) and self.params["cut"] and (self.params["cut"] == "only")

        self.precision = precision
        self.init_cut_coeffs = []

        self.seed = seed
        self.dp = dp

        self.tgt = tgt #target label

    def init_optimizer(self, method_args):
        return self.init_optimize, None
    
    def set_solution_optimizer(self, method, method_args=None):
        assert method in self.optimizers
        self.optimize, _ = self.optimizers[method](method_args)
    
    def best_naive_dp_optimizer(self, method_args):
        # best bounds out of kw and naive interval propagation

        def optimize(*args, **kwargs):
            # self.set_decomposition('pairs', 'KW')
            # bounds_kw = kw_fun(*args, **kwargs)

            # self.set_decomposition('pairs', 'naive')
            # bounds_naive = naive_fun(*args, **kwargs)
            # bounds = torch.max(bounds_kw, bounds_naive)

            # for cifar
            # opt_args = {
            #     'nb_iter': 20,
            #     'lower_initial_step_size': 0.0001,
            #     'lower_final_step_size': 1,
            #     'upper_initial_step_size': 1e2,
            #     'upper_final_step_size': 1e3,
            #     'betas': (0.9, 0.999)
            # }

            ## for food101 old__1
            # opt_args = {
            #         'nb_iter': 20,
            #         'lower_initial_step_size': 10,
            #         'lower_final_step_size': 0.01,
            #         'upper_initial_step_size': 0.001,
            #         'upper_final_step_size': 1e3,
            #         'betas': (0.9, 0.999)
            #     }

            # ## for food101 newer
            opt_args = {
                    'nb_iter': 20,
                    'lower_initial_step_size': 0.00001,
                    'lower_final_step_size': 1,
                    'upper_initial_step_size': 1,
                    'upper_final_step_size': 100,
                    'betas': (0.9, 0.999)
                }

            auto_lirpa_object = AutoLirpa(*args, **kwargs)
            auto_lirpa_object.crown_initialization(*args, **kwargs)

            bounds_auto_lirpa = auto_lirpa_object.auto_lirpa_optimizer(*args, **kwargs, dp=True, opt_args=opt_args)
            # bounds = torch.max(bounds, bounds_auto_lirpa)

            return bounds_auto_lirpa

        return optimize, [None, None]

    def opt_dp_optimizer(self, method_args):
        # best bounds out of kw and naive interval propagation
        kw_fun, kw_logger = self.optimizers['init'](None)
        naive_fun, naive_logger = self.optimizers['init'](None)

        def optimize(*args, **kwargs):
            # self.set_decomposition('pairs', 'KW')
            # bounds_kw = kw_fun(*args, **kwargs)

            # self.set_decomposition('pairs', 'naive')
            # bounds_naive = naive_fun(*args, **kwargs)
            # bounds = torch.max(bounds_kw, bounds_naive)

            bounds_auto_lirpa = autolirpa_opt_dp(*args, **kwargs)
            # bounds = torch.max(bounds, bounds_auto_lirpa)

            return bounds_auto_lirpa

        return optimize, [kw_logger, naive_logger]
    
    def define_linear_approximation(self, input_domain, emb_layer=False, no_conv=False, override_numerical_errors=False):
        '''
        this function computes intermediate bounds and stores them into self.lower_bounds and self.upper_bounds.
        It also stores the network weights into self.weights.
        Now this function will compute these bounds and then condition the layers such that the input constraints are simplex.

        no_conv is an option to operate only on linear layers, by transforming all
        the convolutional layers into equivalent linear layers.
        lower_bounds [input_bounds,1st_layer_output,2nd_layeroutput ....]
        '''

        # store which relus are ambiguous. 1=passing, 0=blocking, -1=ambiguous. Shape: dom_batch_size x layer_width
        self.relu_mask = []
        self.no_conv = no_conv
        # Setup the bounds on the inputs
        self.input_domain = input_domain
        self.opt_time_per_layer = []
        X, eps = input_domain

        next_is_linear = True
        conditioning = True
        prop_simp_bounds = False

        ################################
        ## This checks for the scenario when the first layer is a flatten layer
        ################################
        first_layer_flatten = False
        if not (isinstance(self.layers[0], nn.Conv2d) or isinstance(self.layers[0], nn.Linear)):
            first_layer_flatten = True
            self.layers_copy = copy.deepcopy(self.layers)
            self.layers = self.layers[1:]
        ################################
        ################################

        for lay_idx, layer in enumerate(self.layers):
            if lay_idx == len(self.layers)-1:
                conditioning=False

            if lay_idx == 0:
                assert next_is_linear
                next_is_linear = False
                layer_opt_start_time = time.time()
                if not emb_layer:
                    l_1, u_1, init_cut_coeff, cond_first_linear, lmbd = self.get_preact_bounds_first_layer(X, eps, layer, no_conv, conditioning = conditioning, seed=self.seed)
                else:
                    l_1, u_1, init_cut_coeff, cond_first_linear, lmbd = self.get_preact_bounds_first_emb_layer(X, eps, layer, no_conv, conditioning = conditioning, seed=self.seed)
                layer_opt_end_time = time.time()
                time_used = layer_opt_end_time - layer_opt_start_time
                print(f"Time used for layer {lay_idx}: {time_used}")
                if init_cut_coeff is not None:
                    self.init_cut_coeffs.append(init_cut_coeff)

                if no_conv:
                    # when linearizing conv layers, we need to keep track of the original shape of the bounds
                    self.original_shape_lbs = [-torch.ones_like(X), l_1]
                    self.original_shape_ubs = [torch.ones_like(X), u_1]
                    X = X.view(X.shape[0], -1)
                    X = X.view(X.shape[0], -1)
                    X = X.view(X.shape[0], -1)
                    X = X.view(X.shape[0], -1)
                
                if not emb_layer:
                    self.lower_bounds = [-torch.ones(*X.shape[:-1], 2*X.shape[-1]), l_1]
                    self.upper_bounds = [torch.ones(*X.shape[:-1], 2*X.shape[-1]), u_1]
                else:
                    self.lower_bounds = [-torch.ones(1, X[1].shape[0]), l_1]
                    self.upper_bounds = [torch.ones(1, X[1].shape[0]), u_1]
                
                weights = [cond_first_linear]
                self.relu_mask.append(get_relu_mask(l_1, u_1))


            elif isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                assert next_is_linear
                next_is_linear = False

                orig_shape_prev_ub = self.original_shape_ubs[-1] if no_conv else None
                layer_opt_start_time = time.time()
                l_kp1, u_kp1, init_cut_coeff, obj_layer, obj_layer_orig, lmbd = self.build_simp_layer(self.upper_bounds[-1], self.lower_bounds[-1], layer, self.init_cut_coeffs, lmbd, no_conv, orig_shape_prev_ub=orig_shape_prev_ub, conditioning = conditioning, seed=self.seed, prop_simp_bounds=prop_simp_bounds)
                print('Conditioning time: ', time.time()-layer_opt_start_time)

                weights.append(obj_layer)

                layer_opt_start_time = time.time()
                l_kp1_lirpa , u_kp1_lirpa = self.solve_problem(weights, self.lower_bounds, self.upper_bounds, override_numerical_errors=override_numerical_errors)


                if prop_simp_bounds:
                    l_kp1=torch.max(l_kp1, l_kp1_lirpa)
                    u_kp1=torch.min(u_kp1, u_kp1_lirpa)
                else:
                    l_kp1=l_kp1_lirpa
                    u_kp1=u_kp1_lirpa


                assert (u_kp1 - l_kp1).min() >= 0, "Incompatible bounds"
                layer_opt_end_time = time.time()
                time_used = layer_opt_end_time - layer_opt_start_time
                print(f"Time used for layer {lay_idx}: {time_used}")
                self.opt_time_per_layer.append(layer_opt_end_time - layer_opt_start_time)

                if lay_idx != len(self.layers)-1:
                    self.init_cut_coeffs.append(init_cut_coeff)

                if no_conv:
                    if isinstance(layer, nn.Conv2d):
                        self.original_shape_lbs.append(
                            l_kp1.view(obj_layer_orig.get_output_shape(self.original_shape_lbs[-1].unsqueeze(1).shape)).
                            squeeze(1)
                        )
                        self.original_shape_ubs.append(
                            u_kp1.view(obj_layer_orig.get_output_shape(self.original_shape_ubs[-1].unsqueeze(1).shape)).
                            squeeze(1)
                        )
                    else:
                        self.original_shape_lbs.append(l_kp1)
                        self.original_shape_ubs.append(u_kp1)
                self.lower_bounds.append(l_kp1)
                self.upper_bounds.append(u_kp1)
                if lay_idx < (len(self.layers)-1):
                    # the relu mask doesn't make sense on the final layer
                    self.relu_mask.append(get_relu_mask(l_kp1, u_kp1))
            elif isinstance(layer, nn.ReLU):
                assert not next_is_linear
                next_is_linear = True
            else:
                pass
        self.weights = weights
    
    @staticmethod
    def get_preact_bounds_first_layer(X, eps, layer, no_conv=False, conditioning=True, seed=0):
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
            ###############
            # STEP-1- Conditioning this layer so that input lies in simplex
            ###############
            cond_w_1 = eps * w_1 @ E
            if X_vector.dim() ==2:
                X_vector = X_vector.squeeze(0)
            cond_b_1 = b_1 + X_vector @ w_1.t()
            ###############
            ###############

            ###############
            # STEP-2- construct BatchLinearOp
            ###############
            cond_w_1_unsq = cond_w_1.unsqueeze(0)
            cond_layer = BatchLinearOp(cond_w_1_unsq, cond_b_1)
            ###############

            if conditioning:
                ###############
                # Getting the lambda
                ###############
                if seed!=0:
                    a = np.random.random(b_1.shape[0])
                    a /= a.sum()
                    lmbd=torch.from_numpy(a).float().to(cond_w_1.device)
                else:### default version
                    lmbd=torch.ones(b_1.shape[0]).float().to(cond_w_1.device)
                ###############

                init_cut_coeff = cond_layer.simplex_conditioning(lmbd, conditioning=True)
            
            ###############
            # STEP-4- calculating pre-activation bounds of conditioned layer
            ###############
            # W_min = torch.stack([min(min(row),0) for row in cond_w_1])
            # W_max = torch.stack([max(max(row),0) for row in cond_w_1])
            W_min, _ = torch.min(cond_layer.weights.squeeze(0), 1)
            W_min = torch.clamp(W_min, None, 0)
            W_max, _ = torch.max(cond_layer.weights.squeeze(0), 1)
            W_max = torch.clamp(W_max, 0, None)
            l_1 = W_min + cond_layer.bias
            u_1 = W_max + cond_layer.bias
            l_1 = l_1.unsqueeze(0)
            u_1 = u_1.unsqueeze(0)
            assert (u_1 - l_1).min() >= 0, "Incompatible bounds"
            ###############

            if isinstance(cond_layer, LinearOp) and (X.dim() > 2):
                # This is the first LinearOp, so we need to include the flattening
                cond_layer.flatten_from((X.shape[1], X.shape[2], 2*X.shape[3]))

        assert (u_1 - l_1).min() >= 0, "Incompatible bounds"

        return l_1, u_1, init_cut_coeff, cond_layer, lmbd
    
    @staticmethod
    def build_simp_layer(prev_ub, prev_lb, layer, init_cut_coeffs, lmbd_prev, no_conv=False, orig_shape_prev_ub=None, conditioning=True, seed=0, prop_simp_bounds=False):
        '''
        This function also does the conditioning using init_cut_coeff
        This function also calculates the init_cut_coeff which is the b coefficient for lambda x leq b cut
        This function return a ConvOp or LinearOp object depending on the layer type. 
        This function also computes the pre-activation bounds
        '''
        w_kp1 = layer.weight
        b_kp1 = layer.bias

        obj_layer_orig = None
        init_cut_coeff = None
        lmbd = None

        l_kp1 = None
        u_kp1 = None

        if isinstance(layer, nn.Linear):
            #################################
            ## PREVIOUS LAYER CONDITIONING ##
            #################################
            # Only needs conditioning the weights
            # 1. has alpha scaling
            # 2. has lambda scaling
            w_kp1 = (w_kp1) * init_cut_coeffs[-1]
            # if lmbd_prev is None:
            #     w_kp1 = (w_kp1) * init_cut_coeffs[-1]
            # else:
            #     w_kp1 = (w_kp1*torch.reciprocal(lmbd_prev)) * init_cut_coeffs[-1]
            #################################
            #################################

            ###############
            # STEP-2- construct LinearOp
            ###############
            obj_layer = LinearOp(w_kp1, b_kp1)
            ###############

            if conditioning:
                ###############
                # Getting the lambda
                ###############
                if seed!=0:
                    a = np.random.random(b_kp1.shape[0])
                    a /= a.sum()
                    lmbd=torch.from_numpy(a).float().to(w_kp1.device)
                else:### default version
                    lmbd=torch.ones(b_kp1.shape[0]).float().to(w_kp1.device)
                ###############

                init_cut_coeff = obj_layer.simplex_conditioning(lmbd, conditioning=True)

            # STEP-4- calculating pre-activation bounds of conditioned layer
            if prop_simp_bounds:
                weight_matrix= obj_layer.weights.view(obj_layer.weights.shape[0], -1)
                W_min, _ = torch.min(weight_matrix, 1)
                W_min = torch.clamp(W_min, None, 0)
                W_max, _ = torch.max(weight_matrix, 1)
                W_max = torch.clamp(W_max, 0, None)

                l_kp1 = W_min + obj_layer.bias
                u_kp1 = W_max + obj_layer.bias
           

        if isinstance(obj_layer, LinearOp) and (prev_ub.dim() > 2):
            # This is the first LinearOp,
            # We need to include the flattening
            obj_layer.flatten_from(prev_ub.shape[1:])

        if prop_simp_bounds:
            ini_lbs, ini_ubs = obj_layer.interval_forward(torch.clamp(prev_lb, 0, None),
                                                        torch.clamp(prev_ub, 0, None))
            batch_size = ini_lbs.shape[0]
            out_shape = ini_lbs.shape[1:]#this is the actual output shape
            l_kp1 = l_kp1.view(batch_size, *out_shape)
            u_kp1 = u_kp1.view(batch_size, *out_shape)

            assert (u_kp1 - l_kp1).min() >= 0, "Incompatible bounds"


        return l_kp1, u_kp1, init_cut_coeff, obj_layer, obj_layer_orig, lmbd
    
    def solve_problem(self, weights, lower_bounds, upper_bounds, override_numerical_errors=False):
        '''
        Compute bounds on the last layer of the problem. (it will compute 2*number of output neurons bounds.)
        With batchification, we need to optimize over all layers in any case, as otherwise the tensors of different
         sizes should be kept as a list (slow)
        '''
        ini_lbs, ini_ubs = weights[-1].interval_forward(torch.clamp(lower_bounds[-1], 0, None),
                                                        torch.clamp(upper_bounds[-1], 0, None))
        out_shape = ini_lbs.shape[1:]#this is the actual output shape
        nb_out = prod(out_shape)#number of output neurons of that layer.
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
            c_bound = self.optimize(weights, additional_coeffs, lower_bounds, upper_bounds)
            bound = c_bound if bound is None else torch.cat([bound, c_bound], 1)

        ubs = -bound[:, :nb_out]
        lbs = bound[:, nb_out:]
        lbs = lbs.view(batch_size, *out_shape)
        ubs = ubs.view(batch_size, *out_shape)

        if not override_numerical_errors:
            assert (ubs - lbs).min() >= 0, "Incompatible bounds"
        else:
            ubs = torch.where((ubs - lbs <= 0) & (ubs - lbs >= -1e-5), lbs + 1e-5, ubs)
            assert (ubs - lbs).min() >= 0, "Incompatible bounds"

        return lbs, ubs
    
