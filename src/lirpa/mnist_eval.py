from simplex_solver import SimplexSolver
from auto_LiRPA import PerturbationLpNorm, BoundedModule, BoundedTensor
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time
import numpy as np

import warnings
warnings.filterwarnings('ignore')

model = nn.Sequential(
    nn.Linear(28*28, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)
model.load_state_dict(torch.load('models/relu_model.pth'))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=1, shuffle=True)
x_test, label = next(iter(test_loader))
x_test = x_test.view(-1, 28*28)

eps = 0.01


solver = SimplexSolver(list(model.children()))
ptb = PerturbationLpNorm(eps=eps, norm=1.)
bounded_x = BoundedTensor(x_test, ptb)
auto_lirpa_bounded_model = BoundedModule(model, torch.zeros_like(x_test))
auto_lirpa_bounded_model.eval()


total_diff = 0
num_samples = 0

simplex_times = []
auto_lirpa_times = []
for x_test, label in test_loader:
    x_test = x_test.view(-1, 28*28)
    
    with torch.no_grad():
        simplex_start = time.time()
        simplex_ub, simplex_lb = solver.compute_bounds(x_test, eps)
        simplex_end = time.time()
        simplex_times.append(simplex_end - simplex_start)

        bounded_x = BoundedTensor(x_test, ptb)
        auto_lirpa_bounded_model.bounded_input = torch.zeros_like(x_test)
        auto_lirpa_start = time.time()
        lirpa_lb, lirpa_ub = auto_lirpa_bounded_model.compute_bounds(x=(bounded_x,), method='CROWN')
        auto_lirpa_end = time.time()
        auto_lirpa_times.append(auto_lirpa_end - auto_lirpa_start)
    
    simplex_diff = simplex_ub - lirpa_lb
    lirpa_diff = lirpa_ub - lirpa_lb
    diff = (lirpa_diff - simplex_diff) / torch.abs(lirpa_diff)
    total_diff += torch.sum(diff)
    num_samples += 1

    if num_samples == 1000:
        break

average_diff = total_diff / num_samples
print('\nAverage relative difference: {:.4f}%'.format(average_diff * 100))
# Print out timing information for each algorithm
print('\nAverage running time\nSimplex: {}\nauto_LiRPA: {}'
      .format(np.average(simplex_times), np.average(auto_lirpa_times)))