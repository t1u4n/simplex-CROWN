import torch
from torch import Tensor
import torch.nn as nn
from copy import deepcopy
from simplexcrown.simplex_propagation import simplex_propagation_orig
from simplexcrown.crown import BoundedSequential
from simplexcrown.simplex_neuron import SimplexNeuron, BoundSimplexNeuron_Alpha
from torch.optim import Adam

class SimplexSolver:
    def __init__(self, method: str='alpha-simplex', loss_fn: str='naive') -> None:
        assert method in ['simplex', 'alpha-simplex']
        assert loss_fn in ['naive', 'margin']
        self.method = method

        def _naive_loss_fn(label, upper_bounds, lower_bounds):
            return upper_bounds.mean()
    
        def _margin_loss_fn(label, upper_bounds, lower_bounds):
            correct_lb = lower_bounds[0][label]
            incorrect_labels_upper_bounds = upper_bounds.clone()
            incorrect_labels_upper_bounds[0][label] = float("inf")
            max_incorrect_upper_bounds = incorrect_labels_upper_bounds.max(dim=1)[0]
            return (correct_lb - max_incorrect_upper_bounds)
        
        if loss_fn == 'naive':
            self.loss_fn = _naive_loss_fn
        elif loss_fn == 'margin':
            self.loss_fn = _margin_loss_fn

    def compute_bounds(
            self,
            model: nn.Sequential,
            x0: Tensor,
            eps: float,
            label: int,
            optimize_epochs: int=50, 
            lr :float=1e-3
        ):
        conditioned_model = simplex_propagation_orig(model, x0, eps)
        bounded_model = BoundedSequential.convert(conditioned_model)
        batch_size = x0.shape[0]
        dim = x0.shape[1]
        crown_ub, lb = bounded_model.compute_bounds(torch.zeros(batch_size, 2*dim), upper=True, lower=True)
        simplexified_model = self.simplexify(bounded_model)
        bounded_simplex_model = BoundedSequential.convert(simplexified_model, self.method)

        if self.method == 'alpha-simplex':
            alphas = []
            for l in bounded_simplex_model.children():
                if isinstance(l, BoundSimplexNeuron_Alpha):
                    alphas.append(l.alpha)
            optimizer = Adam(alphas, lr=lr)
            for _ in range(optimize_epochs):
                ub, _ = bounded_simplex_model.compute_bounds(torch.zeros(batch_size, 2*dim), upper=True, lower=False)
                optimizer.zero_grad()
                loss = self.loss_fn(label, ub, lb)
                loss.backward(retain_graph=True)  # Retain the computational graph
                optimizer.step()
                with torch.no_grad():
                    for l in bounded_simplex_model.children():
                        if isinstance(l, BoundSimplexNeuron_Alpha):
                            l.alpha.clamp_(min=0, max=1)
        elif self.method == 'simplex':
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