from simplex_solver import SimplexSolver
from auto_LiRPA import PerturbationLpNorm, BoundedModule, BoundedTensor
import torch
import torch.nn as nn

model = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
model.load_state_dict(torch.load('models/relu_model.pth'))
model.eval()

layers = list(model.children())

x_test, label = torch.load('data/data1.pth')
batch_size = x_test.size(0)
x_test = x_test.reshape(batch_size, -1)
output = model(x_test)
y_size = output.size(1)

eps = 0.03

solver = SimplexSolver(layers)
with torch.no_grad():
    simplex_lb, simplex_ub = solver.compute_bounds(x_test, eps)

ptb = PerturbationLpNorm(eps=eps, norm=1.)
bounded_x = BoundedTensor(x_test, ptb)
auto_lirpa_bounded_model = BoundedModule(model, torch.zeros_like(x_test))
auto_lirpa_bounded_model.eval()
with torch.no_grad():
    lirpa_lb, lirpa_ub = auto_lirpa_bounded_model.compute_bounds(x=(bounded_x,), method='CROWN')

def print_bounds(lb, ub, batch_sz, y_sz):
    for i in range(batch_sz):
        for j in range(y_sz):
            print('f_{j}(x_{i}): {l:8.4f} <= f_{j}(x_{i}+delta) <= {u:8.4f}'.format(
                j=j, i=i, l=lb[i][j].item(), u=ub[i][j].item()))

print("\n\n---------------- Simplex Bounds ----------------\n")
print_bounds(simplex_lb, simplex_ub, batch_size, y_size)

print("\n\n----------------- Crown Bounds -----------------\n")
print_bounds(lirpa_lb, lirpa_ub, batch_size, y_size)

# Compute diff
simplex_diff = [ub - lb for ub, lb in zip(simplex_ub, lirpa_lb)]
lirpa_diff = [ub - lb for ub, lb in zip(lirpa_ub, lirpa_lb)]
print("\n\n---------------- Differences between bounds methods ----------------\n")
total_diff = 0
for i in range(batch_size):
    for j in range(y_size):
        diff = (lirpa_diff[i][j].item() - simplex_diff[i][j].item()) / abs(lirpa_diff[i][j].item())
        total_diff += diff
        print("f_{}: {:+.4f}%".format(j, diff * 100))

print('\nAverage relative difference: {}%'.format(total_diff/(batch_size*y_size) * 100))