from simplexcrown.simplex_solver import SimplexSolver
from auto_LiRPA import PerturbationLpNorm, BoundedTensor, BoundedModule
import argparse
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time
from simple_mlp import SimpleMLP
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def print_bounds(output, lb, ub, batch_sz, y_sz):
    for i in range(batch_sz):
        for j in range(y_sz):
            assert lb[i][j] <= output[i][j] and ub[i][j] >= output[i][j]
            print('f_{j}(x_{i}): {l:8.4f} <= f_{j}(x_{i}+delta) <= {u:8.4f}'.format(
                j=j, i=i, l=lb[i][j].item(), u=ub[i][j].item()))

def main(args):
    model = torch.load(args.model)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=1, shuffle=True)

    simplex_solver = SimplexSolver(method=args.method, loss_fn=args.loss)
    ptb = PerturbationLpNorm(eps=args.epsilon, norm=1.)
    auto_lirpa_bounded_model = BoundedModule(model, torch.empty(1, 28*28))
    auto_lirpa_bounded_model.eval()

    total_diff = 0
    num_samples = 0

    simplex_times = []
    auto_lirpa_times = []
    for x_test, label in test_loader:
        x_test = x_test.view(-1, 28*28)
        output = model(x_test)
        
        simplex_start = time.time()
        simplex_lb, simplex_ub = simplex_solver.compute_bounds(
                model=model, x0=x_test, eps=args.epsilon, label=label, optimize_epochs=args.epochs, lr=args.lr)
        simplex_end = time.time()
        simplex_times.append(simplex_end - simplex_start)

        with torch.no_grad():
            bounded_x = BoundedTensor(x_test, ptb)
            auto_lirpa_bounded_model.bounded_input = torch.zeros_like(x_test)
            auto_lirpa_start = time.time()
            lirpa_lb, lirpa_ub = auto_lirpa_bounded_model.compute_bounds(x=(bounded_x,), method='CROWN')
            auto_lirpa_end = time.time()
            auto_lirpa_times.append(auto_lirpa_end - auto_lirpa_start)

        simplex_diff = simplex_ub - simplex_lb
        lirpa_diff = lirpa_ub - lirpa_lb
        diff = (lirpa_diff - simplex_diff) / torch.abs(lirpa_diff)
        total_diff += torch.sum(diff)
        num_samples += 1

        if args.show_bounds:
            print("\n\n--------------------------Simplex--------------------------\n")
            print_bounds(output, simplex_lb, simplex_ub, 1, 10)
            print("\n\n--------------------------auto_LiRPA--------------------------\n")
            print_bounds(output, lirpa_lb, lirpa_ub, 1, 10)

        if num_samples == args.n_data:
            break

    average_diff = total_diff / num_samples
    print('\nAverage relative difference: {:.4f}%'.format(average_diff * 100))
    # Print out timing information for each algorithm
    print('\nAverage running time\nSimplex: {}\nauto_LiRPA: {}'
        .format(np.average(simplex_times), np.average(auto_lirpa_times)))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Adversarial testing of neural networks.")

    parser.add_argument('--epsilon', type=float, default=0.03, help='Epsilon for Lp perturbation')
    parser.add_argument('--n_data', type=int, default=100, help='Number of data samples to process')
    parser.add_argument('--method', type=str, default='alpha-simplex', help='Optimization method to use')
    parser.add_argument('--loss', type=str, default='naive', help='Type of loss function')
    parser.add_argument('--epochs', type=int, default=50, help='Number of optimization epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for optimization')
    parser.add_argument('--model', type=str, help='Learning rate for optimization')
    parser.add_argument('--show_bounds', type=bool, default=False, help='Learning rate for optimization')

    args = parser.parse_args()

    main(args)