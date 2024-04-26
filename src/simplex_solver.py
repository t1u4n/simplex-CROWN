import torch
from torch import Tensor
import torch.nn as nn
from copy import deepcopy
from simplex_propagation import simplex_propagation_orig
from crown import BoundedSequential
from simplex_neuron import SimplexNeuron

class SimplexSolver:
    def compute_bounds(self, model: nn.Sequential, x0: Tensor, eps: float):
        conditioned_model = simplex_propagation_orig(model, x0, eps)
        bounded_model = BoundedSequential.convert(conditioned_model)
        batch_size = x0.shape[0]
        dim = x0.shape[1]
        crown_ub, lb = bounded_model.compute_bounds(torch.zeros(batch_size, 2*dim), upper=True, lower=True)
        simplexified_model = self.simplexify(bounded_model)
        bounded_simplex_model = BoundedSequential.convert(simplexified_model)
        ub, _ = bounded_simplex_model.compute_bounds(torch.zeros(batch_size, 2*dim), upper=True, lower=False)
        return (lb, torch.min(ub, crown_ub))

    def simplexify(self, bounded_model: BoundedSequential):
        simplexified_layers = []
        layers = list(bounded_model.children())
        for i, layer in enumerate(layers):
            if isinstance(layer, nn.Linear):
                assert i == len(layers) - 1 or isinstance(layers[i + 1], nn.ReLU)
                if i < len(layers) - 1:
                    neuron = SimplexNeuron(layer.weight, layer.bias)
                    neuron.preact_ub = layers[i + 1].upper_u
                    neuron.preact_lb = layers[i + 1].lower_l
                    simplexified_layers.append(neuron)
                else:
                    simplexified_layers.append(deepcopy(layer))
        return nn.Sequential(*simplexified_layers)
    
# Sanity check
if __name__=="__main__":
    model = nn.Sequential(
                nn.Linear(28*28, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
            )
    model.load_state_dict(torch.load('models/relu_model.pth'))
    model.eval()

    x_test, label = torch.load('data/data1.pth')
    batch_size = x_test.size(0)
    x_test = x_test.reshape(batch_size, -1)
    output = model(x_test)
    y_size = output.size(1)

    eps = 0.3

    solver = SimplexSolver()
    lb, ub = solver.compute_bounds(model, x_test, eps)
    
    def print_bounds(lb, ub, batch_sz, y_sz):
        for i in range(batch_sz):
            for j in range(y_sz):
                assert lb[i][j] <= output[i][j] and ub[i][j] >= output[i][j]
                print('f_{j}(x_{i}): {l:8.4f} <= f_{j}(x_{i}+delta) <= {u:8.4f}'.format(
                    j=j, i=i, l=lb[i][j].item(), u=ub[i][j].item()))
    
    print_bounds(lb, ub, batch_size, y_size)

    print("------------------------------------------------")

    from auto_LiRPA import PerturbationLpNorm, BoundedModule, BoundedTensor
    ptb = PerturbationLpNorm(eps=eps, norm=1., x_L=torch.zeros_like(x_test).float(), x_U=torch.ones_like(x_test).float())
    bounded_x = BoundedTensor(x_test, ptb)
    auto_lirpa_bounded_model = BoundedModule(model, torch.zeros_like(x_test))
    auto_lirpa_bounded_model.eval()
    with torch.no_grad():
        lirpa_lb, lirpa_ub = auto_lirpa_bounded_model.compute_bounds(x=(bounded_x,), method='CROWN')
    print_bounds(lirpa_lb, lirpa_ub, batch_size, y_size)

    # Compute diff
    simplex_diff = [u - l for u, l in zip(ub, lb)]
    lirpa_diff = [u - l for u, l in zip(lirpa_ub, lirpa_lb)]
    print("\n\n---------------- Differences between bounds methods ----------------\n")
    total_diff = 0
    for i in range(batch_size):
        for j in range(y_size):
            diff = max(0, lirpa_diff[i][j].item() - simplex_diff[i][j].item())
            diff /= abs(lirpa_diff[i][j].item())
            total_diff += diff
            print("f_{}: {:+.4f}%".format(j, diff * 100))

    print('\nAverage relative difference: {:+.4f}%'.format(total_diff/(batch_size*y_size) * 100))