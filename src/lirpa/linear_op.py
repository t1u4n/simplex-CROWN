import torch
import torch.nn.functional as F

class LinearOp:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.out_features = weights.shape[0]
        self.in_features = weights.shape[1]
        self.preshape = (self.in_features,)
        self.postshape = (self.out_features,)

        # convex hull coefficient
        self.dp_weights = torch.clamp(self.weights.T + self.bias, 0, None) - torch.clamp(self.bias, 0, None)
        self.dp_weights = self.dp_weights.T

        self.alpha_coeff = 1
        self.lmbd = torch.ones(bias.shape[0], dtype=torch.float, device=weights.device)

    def forward(self, inp):
        return F.linear(inp, self.weights, self.bias)

    def interval_forward(self, lb_in, ub_in):
        pos_wt = torch.clamp(self.weights, 0, None)
        neg_wt = torch.clamp(self.weights, None, 0)
        pos_lay = LinearOp(pos_wt, self.bias)
        neg_lay = LinearOp(neg_wt, torch.zeros_like(self.bias))
        lb_out = (pos_lay.forward(lb_in.unsqueeze(1)) + neg_lay.forward(ub_in.unsqueeze(1))).squeeze(1)
        ub_out = (pos_lay.forward(ub_in.unsqueeze(1)) + neg_lay.forward(lb_in.unsqueeze(1))).squeeze(1)

        return lb_out, ub_out

    def backward(self, out):
        return torch.matmul(out, self.weights)

    def __repr__(self):
        return f'<Linear: {self.in_features} -> {self.out_features}>'


    def simplex_conditioning(self, lmbd, conditioning=False):
        w_kp1 = self.weights
        b_kp1 = self.bias

        wb = w_kp1 + b_kp1[:,None]
        wb_clamped = torch.clamp(wb, 0, None)
        lambda_wb_clamped = (wb_clamped.T*lmbd).T
        wb_col_sum = torch.sum(lambda_wb_clamped, 0)
        b_clamped = torch.clamp(b_kp1, 0, None)
        lambda_b_clamped = b_clamped*lmbd
        b_sum = torch.sum(lambda_b_clamped)
        init_cut_coeff = max(torch.max(wb_col_sum),b_sum)
        if init_cut_coeff==0:
            init_cut_coeff=torch.ones_like(init_cut_coeff)

        if conditioning:
            w_kp1 = w_kp1 / init_cut_coeff
            b_kp1 = b_kp1 / init_cut_coeff

            self.weights = w_kp1
            self.bias = b_kp1

            self.dp_weights = torch.clamp(self.weights.T + self.bias, 0, None) - torch.clamp(self.bias, 0, None)
            self.dp_weights = self.dp_weights.T
        return init_cut_coeff

    def dp_forward(self, inp):
        return F.linear(inp, self.dp_weights, self.bias)

    def dp_backward(self, out):
        return torch.matmul(out, self.dp_weights)

    def bias_backward(self, out):
        return torch.matmul(out, self.bias)

    def dp_bias_backward(self, out):
        return torch.matmul(out, self.bias.clamp(min=0))