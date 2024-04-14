import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_alpha(W, bias, lmbda: float=1.) -> float:
    wb = W + bias[:, None]
    wb_clamped = torch.clamp(wb, min=0) * lmbda
    wb_max = torch.max(torch.sum(wb_clamped, dim=0))

    b_clamped = torch.clamp(bias, min=0) * lmbda
    b_max = torch.sum(b_clamped)

    alpha = torch.max(wb_max, b_max)
    return alpha.item() if alpha.item() != 0 else 1.

def compute_uk(x, w, bias):
    batch_size, input_dim = x.shape
    output_dim = bias.shape[0]
    
    output = torch.zeros(batch_size, output_dim).to(x.device)

    Lk_0 = bias
    sigma_Lk_0 = F.relu(Lk_0)

    for k in range(output_dim):
        for i in range(input_dim):
            Lk_ei = w[k, i] + bias[k]
            sigma_Lk_ei = F.relu(Lk_ei)

            difference = sigma_Lk_ei - sigma_Lk_0[k]

            output[:, k] += x[:, i] * difference

        output[:, k] += sigma_Lk_0[k]

    return output

for _ in range(1000):
    w1 = torch.randn(64, 28*28)
    b1 = torch.randn(64)
    w2 = torch.randn(10, 64)
    b2 = torch.randn(10)
    x = torch.rand(1, 28*28)
    x /= torch.sum(x)

    alpha1 = compute_alpha(w1, b1)
    w1 /= alpha1
    b1 /= alpha1
    w2 *= alpha1

    A1 = torch.clamp(w1 + b1[:, None], min=0) - torch.clamp(b1[:, None], min=0)
    ub1 = x.matmul(A1.t()) + torch.clamp(b1, min=0)
    A2 = torch.clamp(w2 + b2[:, None], min=0) - torch.clamp(b2[:, None], min=0)
    ub2 = ub1.matmul(A2.t()) + torch.clamp(b2, min=0)

    ub3 = x.matmul(A2.matmul(A1).t()) + \
        A2.matmul(torch.clamp(b1, min=0)) + torch.clamp(b2, min=0)
    assert torch.allclose(ub2, ub3)

    A = A2.matmul(A1)
    sum_bias = A2.matmul(torch.clamp(b1, min=0)) + torch.clamp(b2, min=0)
    ub = x.matmul(A.t()) + sum_bias
    assert torch.allclose(ub2, ub)

    out1 = torch.clamp(x.matmul(w1.t()) + b1, min=0)
    out2 = torch.clamp(out1.matmul(w2.t()) + b2, min=0)

    def _check(ub, out):
        mask = ub >= out
        if not torch.all(mask):
            violating_indices = torch.where(mask == False)
            print("Violating indices:", violating_indices)
            print("ub1 elements at violating indices:", ub1[violating_indices])
            print("out1 elements at violating indices:", out1[violating_indices])
            assert False, f"ub: {ub1}, out: {out1}"

    _check(ub1, out1)
    _check(ub2, out2)
print("Upper bound", ub2)
print("Out", out2)