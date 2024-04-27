import torch
import torch.nn.functional as F

class LinearOp:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        # convex hull coefficient
        self.convex_hull_coeff = torch.relu(self.weights + self.bias.unsqueeze(1)) \
                                - torch.relu(self.bias.unsqueeze(1))

    def forward(self, inp):
        return F.linear(inp, self.weights, self.bias)

    def interval_forward(self, lb, ub):
        pos_wt = torch.clamp(self.weights, 0, None)
        neg_wt = torch.clamp(self.weights, None, 0)
        pos_lay = LinearOp(pos_wt, self.bias)
        neg_lay = LinearOp(neg_wt, torch.zeros_like(self.bias))
        lb_out = (pos_lay.forward(lb.unsqueeze(1)) + neg_lay.forward(ub.unsqueeze(1))).squeeze(1)
        ub_out = (pos_lay.forward(ub.unsqueeze(1)) + neg_lay.forward(lb.unsqueeze(1))).squeeze(1)

        return lb_out, ub_out

    def backward(self, out):
        return torch.matmul(out, self.weights)

    def simplex_conditioning(self):
        # Compute scaling factor alpha
        wb = self.weights + self.bias[:,None]
        wb_clamped = torch.clamp(wb, 0, None)
        lambda_wb_clamped = (wb_clamped.T).T
        wb_col_sum = torch.sum(lambda_wb_clamped, 0)
        b_clamped = torch.clamp(self.bias, 0, None)
        lambda_b_clamped = b_clamped
        b_sum = torch.sum(lambda_b_clamped)
        alpha = max(torch.max(wb_col_sum), b_sum, 1.)

        self.weights /= alpha
        self.bias /= alpha
        self.convex_hull_coeff = torch.relu(self.weights + self.bias.unsqueeze(1)) - torch.relu(self.bias.unsqueeze(1))
        return alpha

    def convex_hull_forward(self, inp):
        return F.linear(inp, self.convex_hull_coeff, self.bias)

    def convex_hull_backward(self, out):
        return torch.matmul(out, self.convex_hull_coeff)

    def bias_backward(self, out):
        return torch.matmul(out, self.bias)

    def convex_hull_bias_backward(self, out):
        return torch.matmul(out, self.bias.clamp(min=0))