import torch
import torch.nn as nn
from torch import Tensor
from copy import deepcopy
from simplex_neuron import SimplexNeuron

def convert_first_layer(layer: nn.Linear, x0: Tensor, eps: float) -> nn.Linear:
    # shape of W: [out_features, in_features]
    # shape of bias: [out_features]
    W, bias = layer.weight, layer.bias
    m = x0.numel()

    # shape of M: [in_features, 2*in_features]
    M = torch.zeros(m, 2*m, device=W.device)
    M[torch.arange(m), torch.arange(0, 2*m, 2)] = 1
    M[torch.arange(m), torch.arange(1, 2*m, 2)] = -1

    # shape of W': [out_features, 2*in_features]
    W_prime = eps * torch.matmul(W, M)
    # shape of bias': [1, out_features]
    bias_prime = torch.matmul(x0, W.t()) + bias if bias is not None else torch.matmul(x0, W.t())
    bias_prime.squeeze_(0)

    lin_layer = nn.Linear(W_prime.shape[1], W_prime.shape[0])
    with torch.no_grad():
        lin_layer.weight.copy_(W_prime)
        lin_layer.bias.copy_(bias_prime)

    return lin_layer

def compute_alpha(W: Tensor, bias: Tensor, lmbda: float=1.) -> float:
    wb = W + bias[:, None]
    wb_clamped = torch.clamp(wb, min=0) * lmbda
    wb_max = torch.max(torch.sum(wb_clamped, dim=0))

    b_clamped = torch.clamp(bias, min=0) * lmbda
    b_max = torch.sum(b_clamped)

    alpha = torch.max(wb_max, b_max)
    return alpha.item() if alpha.item() != 0 else 1.

def condition_layer(layer: nn.Linear, lmbda: float=1., scale: bool=True) -> float:
    alpha = compute_alpha(layer.weight, layer.bias, lmbda)
    with torch.no_grad():
        if scale:
            layer.weight.copy_(lmbda * layer.weight / alpha)
            if layer.bias is not None:
                layer.bias.copy_(layer.bias / alpha)
        else:
            layer.weight.copy_(lmbda * layer.weight)
    return alpha

def simplex_propagation(model: nn.Sequential, x0: Tensor, eps: float) -> nn.Sequential:
    # Each linear except the final layer should be followed by a ReLU activation layer
    layers = [deepcopy(l) for l in model]
    new_layers = []

    layers[0] = convert_first_layer(layers[0], x0, eps)
    alphas = []
    for i, layer in enumerate(layers):
        if isinstance(layer, nn.Linear):
            assert i == len(layers) - 1 or isinstance(layers[i + 1], nn.ReLU)
            alpha = condition_layer(
                layer=layer, lmbda=alphas[-1] if alphas else 1., scale=(i != len(layers) - 1))
            alphas.append(alpha)

            if i < len(layers) - 1:
                new_layers.append(SimplexNeuron(layer.weight, layer.bias))
            else:
                new_layers.append(deepcopy(layer))
    
    return nn.Sequential(*new_layers)





def test_output_of_all_layers_lie_on_simplex(num: int, eps: float) -> None:
    model = nn.Sequential(
                nn.Linear(28*28, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
            )
    model.load_state_dict(torch.load('models/relu_model.pth'))
    model.eval()

    def _generate_simplex_input(dim: int) -> Tensor:
        # Generate a random point on the unit hyper-simplex (simplex).
        vec = torch.rand(dim)
        vec /= torch.sum(vec)
        return vec
    
    def _is_in_simplex(vec: Tensor) -> bool:
        return (torch.sum(vec) <= 1.) and (torch.all(vec >= 0))

    for i in range(num):
        # Generate random input of size
        new_model = simplex_propagation(model, torch.rand(1, 28*28), eps)
        x = _generate_simplex_input(2*28*28).unsqueeze(0)
        
        outputs = []
        for j, layer in enumerate(new_model):
            x = layer(x)
            outputs.append(x)
            if isinstance(layer, SimplexNeuron):
                assert _is_in_simplex(x), f"Test {i}: Layer {j} output is not in simplex, sum is {torch.sum(x)}, all positive {torch.all(x>=0)}"

    print(f"All {num} tests passed! Output of each layer lies on the simplex.")

def test_does_not_modify_final_output(num: int, eps: float) -> None:
    model = nn.Sequential(
                nn.Linear(28*28, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
            )
    model.load_state_dict(torch.load('models/relu_model.pth'))
    model.eval()

    def _generate_simplex_input(dim: int) -> Tensor:
        # Generate a random point on the unit hyper-simplex (simplex).
        vec = torch.rand(dim)
        vec /= torch.sum(vec)
        return vec

    for i in range(num):
        # Generate random input of size
        x0 = torch.rand(1, 28*28)
        layers = [deepcopy(l) for l in model]
        layers[0] = convert_first_layer(layers[0], x0, eps)

        new_model = simplex_propagation(model, x0, eps)
        modified_model = nn.Sequential(*layers)

        simplex_inp = _generate_simplex_input(2*28*28).unsqueeze(0)
        
        orig_out = modified_model(simplex_inp)
        new_out = new_model(simplex_inp)

        assert torch.allclose(orig_out, new_out), f"Test {i}, original output: {orig_out}, new output: {new_out}"
        
    print(f"All {num} tests passed! Final outputs haven't been modified.")

if __name__=="__main__":
    test_output_of_all_layers_lie_on_simplex(500, 1e-3)
    test_does_not_modify_final_output(500, 1e-3)