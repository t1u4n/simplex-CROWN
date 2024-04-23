import torch
import time

class LinearOp:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.out_features = weights.shape[0]
        self.in_features = weights.shape[1]
        self.flatten_from_shape = None
        self.preshape = (self.in_features,)
        self.postshape = (self.out_features,)

        #################################
        #### dp upper bound function ####
        self.dp_weights = torch.clamp(self.weights.T + self.bias, 0, None) - torch.clamp(self.bias, 0, None)
        self.dp_weights = self.dp_weights.T
        #################################

        #################################
        #### simplex cut coefficients ###
        self.alpha_coeff = 1
        self.lmbd = torch.ones(bias.shape[0], dtype=torch.float, device=weights.device)
        #################################

    def normalize_outrange(self, lbs, ubs):
        inv_range = 1.0 / (ubs - lbs)
        self.bias = inv_range * (2 * self.bias - ubs - lbs)
        self.weights = 2 * inv_range.unsqueeze(1) * self.weights

    def forward(self, inp):

        if self.flatten_from_shape is not None:
            inp = inp.view(*inp.shape[:2], -1)
        # IMPORTANT: batch matmul is bugged, bmm + expand needs to be used, instead
        # https://discuss.pytorch.org/t/unexpected-huge-memory-cost-of-matmul/41642
        forw_out = torch.bmm(
            inp,
            self.weights.t().unsqueeze(0).expand((inp.shape[0], self.weights.shape[1], self.weights.shape[0]))
        )
        forw_out += self.bias.view((1,) * (inp.dim() - 1) + self.weights.shape[:1])

        return forw_out

    def interval_forward(self, lb_in, ub_in):
        if self.flatten_from_shape is not None:
            lb_in = lb_in.view(lb_in.shape[0], -1)
            ub_in = ub_in.view(ub_in.shape[0], -1)

        pos_wt = torch.clamp(self.weights, 0, None)
        neg_wt = torch.clamp(self.weights, None, 0)
        pos_lay = LinearOp(pos_wt, self.bias)
        neg_lay = LinearOp(neg_wt, torch.zeros_like(self.bias))
        lb_out = (pos_lay.forward(lb_in.unsqueeze(1)) + neg_lay.forward(ub_in.unsqueeze(1))).squeeze(1)
        ub_out = (pos_lay.forward(ub_in.unsqueeze(1)) + neg_lay.forward(lb_in.unsqueeze(1))).squeeze(1)

        return lb_out, ub_out

    def backward(self, out):

        # IMPORTANT: batch matmul is bugged, bmm + expand needs to be used, instead
        # https://discuss.pytorch.org/t/unexpected-huge-memory-cost-of-matmul/41642
        back_inp = torch.bmm(
            out,
            self.weights.unsqueeze(0).expand((out.shape[0], self.weights.shape[0], self.weights.shape[1]))
        )

        if self.flatten_from_shape is not None:
            back_inp = back_inp.view((out.shape[0], out.shape[1]) + self.flatten_from_shape)
        return back_inp

    def get_output_shape(self, in_shape):
        """
        Return the output shape (as tuple) given the input shape. The input shape is the shape will influence the output
        shape.
        """
        return (*in_shape[:2], self.out_features)

    def get_bias(self):
        """
        Return the bias with the correct unsqueezed shape.
        """
        return self.bias.view((1, 1, *self.bias.shape))

    def __repr__(self):
        return f'<Linear: {self.in_features} -> {self.out_features}>'

    def flatten_from(self, shape):
        self.flatten_from_shape = shape


    def simplex_conditioning(self, lmbd, conditioning=False):
        '''
        Given lambda, this function first finds the corresponding alpha (init_cut_coeff) for the simplex cut.
        If conditioning is true, then it conditions the layer to lie in a simplex using the lambda and alpha


        Returns init_cut_coeff
        '''

        w_kp1 = self.weights
        b_kp1 = self.bias

        ###############
        # STEP-2a- finding the cut (this might use the above pre-act lower bounds)
        #not using for this kind of init but will use in the future
        ###############
        wb = w_kp1 + b_kp1[:,None]
        wb_clamped = torch.clamp(wb, 0, None)
        lambda_wb_clamped = (wb_clamped.T*lmbd).T
        wb_col_sum = torch.sum(lambda_wb_clamped, 0)
        # for origin point
        b_clamped = torch.clamp(b_kp1, 0, None)
        lambda_b_clamped = b_clamped*lmbd
        b_sum = torch.sum(lambda_b_clamped)
        #
        init_cut_coeff = max(torch.max(wb_col_sum),b_sum)
        # init_cut_coeff = max(max(wb_col_sum),b_sum)
        if init_cut_coeff==0:
            init_cut_coeff=torch.ones_like(init_cut_coeff)
        ###############

        if conditioning:
            ###############
            # STEP-2b- Conditioning this layer. now output also lies in simplex
            ###############
            # Needs conditioning both weights and bias
            # 1. has alpha scaling
            # 2. has lambda scaling
            # w_kp1 = (w_kp1.T*lmbd).T / init_cut_coeff
            # b_kp1 = b_kp1*lmbd / init_cut_coeff

            w_kp1 = w_kp1 / init_cut_coeff
            b_kp1 = b_kp1 / init_cut_coeff
            ###############

            ### Updatinng weights and bias
            self.weights = w_kp1
            self.bias = b_kp1

            ### Updating dp weights. this kind of update assumes that the layer has been simplex conditioned
            ### dp upper bound function ###
            self.dp_weights = torch.clamp(self.weights.T + self.bias, 0, None) - torch.clamp(self.bias, 0, None)
            self.dp_weights = self.dp_weights.T
            ###############################
        return init_cut_coeff

    ############################################
    ############# DP FUNCTIONS #################
    def update_dp_weights(self, lmbd):
        # with torch.enable_grad():

        zero_weights = torch.zeros_like(self.weights)
        lmbd_cond_weights = torch.where(lmbd > 0, self.weights*torch.reciprocal(lmbd), zero_weights)
        self.dp_weights = torch.clamp(lmbd_cond_weights.T + self.bias, 0, None) - torch.clamp(self.bias, 0, None)
        self.dp_weights = self.dp_weights.T
        self.dp_weights = self.dp_weights*lmbd

        if torch.any(self.dp_weights.isnan()).item():
            print(lmbd)
            print(self.dp_weights)
            input('Found NaN')

    def dp_forward(self, inp):
        ######################
        # This function is for dp weights and bias
        # it returns W'x + relu(b)
        ######################

        if self.flatten_from_shape is not None:
            inp = inp.view(*inp.shape[:2], -1)

        # IMPORTANT: batch matmul is bugged, bmm + expand needs to be used, instead
        # https://discuss.pytorch.org/t/unexpected-huge-memory-cost-of-matmul/41642
        forw_out = torch.bmm(
            inp,
            self.dp_weights.t().unsqueeze(0).expand((inp.shape[0], self.weights.shape[1], self.weights.shape[0]))
        )
        forw_out += torch.clamp(self.bias, 0, None).view((1,) * (inp.dim() - 1) + self.weights.shape[:1])

        return forw_out

    def dp_backward(self, out, inp_bound=None):
        ######################
        # This function is for dp weights
        ######################

        # IMPORTANT: batch matmul is bugged, bmm + expand needs to be used, instead
        # https://discuss.pytorch.org/t/unexpected-huge-memory-cost-of-matmul/41642
        back_inp = torch.bmm(
            out,
            self.dp_weights.unsqueeze(0).expand((out.shape[0], self.weights.shape[0], self.weights.shape[1]))
        )

        if self.flatten_from_shape is not None:
            back_inp = back_inp.view((out.shape[0], out.shape[1]) + self.flatten_from_shape)
        return back_inp
    ############################################
    ############################################

    ############################################
    ########## LIRPA FUNCTIONS ##################
    ## this is a function created for lirpa
    def bias_backward(self, out):
        back_inp = out @ self.bias

        # if self.flatten_from_shape is not None:
        #     back_inp = back_inp.view((out.shape[0], out.shape[1]) + self.flatten_from_shape)
        return back_inp

    ## this is a function created for lirpa
    def dp_bias_backward(self, out):
        back_inp = out @ torch.clamp(self.bias, 0, None)

        # if self.flatten_from_shape is not None:
        #     back_inp = back_inp.view((out.shape[0], out.shape[1]) + self.flatten_from_shape)
        return back_inp
    ############################################
    ############################################

class BatchLinearOp(LinearOp):
    """
    Exactly same interface and functionality as LinearOp, but batch of weights and biases.
    Ignores flatten from shape as this will never be used as a final layer
    """
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.batch_size = weights.shape[0]
        self.out_features = weights.shape[1]
        self.in_features = weights.shape[2]
        self.flatten_from_shape = None
        self.preshape = (self.in_features,)
        self.postshape = (self.out_features,)
        ### dp upper bound function ###
        self.dp_weights = torch.clamp(self.weights.squeeze(0).T + self.bias, 0, None) - torch.clamp(self.bias, 0, None)
        self.dp_weights = self.dp_weights.T.unsqueeze(0)
        ###############################

    def forward(self, inp):
        if self.flatten_from_shape is not None:
            inp = inp.view(*inp.shape[:2], -1)
        domain_batch = inp.shape[0]
        layer_batch = inp.shape[1]
        # TODO: there must be a more memory-efficient way to perform this
        forw_out = (self.weights.unsqueeze(1) @ inp.unsqueeze(-1)).view(
            (domain_batch, layer_batch, self.weights.shape[1], 1)).squeeze(-1)
        forw_out += self.bias.view((1,) * (inp.dim() - 1) + self.weights.shape[1:2])
        return forw_out

    def backward(self, out):
        # back_inp = out @ self.weights
        # TODO: there must be a more memory-efficient way to perform this
        back_inp = (out.unsqueeze(1) @ self.weights.unsqueeze(1)).squeeze(1)
        if self.flatten_from_shape is not None:
            back_inp = back_inp.view((out.shape[0], out.shape[1]) + self.flatten_from_shape)
        return back_inp

    def interval_forward(self, lb_in, ub_in):
        if self.flatten_from_shape is not None:
            lb_in = lb_in.view(lb_in.shape[0], -1)
            ub_in = ub_in.view(ub_in.shape[0], -1)

        pos_wt = torch.clamp(self.weights, 0, None)
        neg_wt = torch.clamp(self.weights, None, 0)
        pos_lay = BatchLinearOp(pos_wt, self.bias)
        neg_lay = BatchLinearOp(neg_wt, torch.zeros_like(self.bias))
        lb_out = (pos_lay.forward(lb_in.unsqueeze(1)) + neg_lay.forward(ub_in.unsqueeze(1))).squeeze(1)
        ub_out = (pos_lay.forward(ub_in.unsqueeze(1)) + neg_lay.forward(lb_in.unsqueeze(1))).squeeze(1)

        return lb_out, ub_out

    # def get_bias(self):
    #     """
    #     Return the bias with the correct unsqueezed shape.
    #     """
    #     return self.bias.unsqueeze(1)

    def simplex_conditioning(self, lmbd, conditioning=False):
        '''
        Given lambda, this function first finds the corresponding alpha (init_cut_coeff) for the simplex cut.
        If conditioning is true, then it conditions the layer to lie in a simplex using the lambda and alpha


        Returns init_cut_coeff
        '''

        start_time = time.time()
        cond_w_1 = self.weights.squeeze(0)
        cond_b_1 = self.bias
        ###############
        # STEP-2a- finding the cut (this might use the above pre-act lower bounds)
        #not using for this kind of init but will use in the future
        ###############
        wb = cond_w_1 + cond_b_1[:,None]
        wb_clamped = torch.clamp(wb, 0, None)
        # lambda_wb_clamped = (wb_clamped.T*lmbd).T
        lambda_wb_clamped = (wb_clamped)
        wb_col_sum = torch.sum(lambda_wb_clamped, 0)
        # for origin point
        b_clamped = torch.clamp(cond_b_1, 0, None)
        # lambda_b_clamped = b_clamped*lmbd
        lambda_b_clamped = b_clamped
        b_sum = torch.sum(lambda_b_clamped)
        #
        init_cut_coeff = max(torch.max(wb_col_sum),b_sum)
        if init_cut_coeff==0:
            init_cut_coeff=torch.ones_like(init_cut_coeff)
        ###############

        if conditioning:
            ###############
            # STEP-2b- Conditioning this layer. now output also lies in simplex
            ###############
            # Needs conditioning both weights and bias
            # 1. has alpha scaling
            # 2. has lambda scaling
            # cond_w_1 = (cond_w_1.T*lmbd).T / init_cut_coeff
            # cond_b_1 =  cond_b_1*lmbd / init_cut_coeff

            cond_w_1 = cond_w_1 / init_cut_coeff
            cond_b_1 =  cond_b_1 / init_cut_coeff
            ###############

            ### Updatinng weights and bias
            self.weights = cond_w_1.unsqueeze(0)
            self.bias = cond_b_1

            ### Updating dp weights
            ### dp upper bound function ###
            self.dp_weights = torch.clamp(cond_w_1.T + cond_b_1, 0, None) - torch.clamp(cond_b_1, 0, None)
            self.dp_weights = self.dp_weights.T.unsqueeze(0)
            ###############################

        return init_cut_coeff

    def dp_forward(self, inp):
        ######################
        # This function is for dp weights and bias
        # it returns W'x + relu(b)
        ######################
        print(f"DP forward called, input: {inp}")
        if self.flatten_from_shape is not None:
            inp = inp.view(*inp.shape[:2], -1)
        domain_batch = inp.shape[0]
        layer_batch = inp.shape[1]
        # TODO: there must be a more memory-efficient way to perform this
        forw_out = (self.dp_weights.unsqueeze(1) @ inp.unsqueeze(-1)).view(
            (domain_batch, layer_batch, self.weights.shape[1], 1)).squeeze(-1)
        forw_out += self.bias.view((1,) * (inp.dim() - 1) + self.weights.shape[1:2])
        return forw_out

    def dp_backward(self, out, inp_bound=None):
        ######################
        # This function is for dp weights
        ######################
        # back_inp = out @ self.weights
        # TODO: there must be a more memory-efficient way to perform this
        back_inp = (out.unsqueeze(1) @ self.dp_weights.unsqueeze(1)).squeeze(1)
        if self.flatten_from_shape is not None:
            back_inp = back_inp.view((out.shape[0], out.shape[1]) + self.flatten_from_shape)
        return back_inp

    def __repr__(self):
        return f'{self.batch_size} x <Linear: {self.in_features} -> {self.out_features}>'