from simplex_solver import SimplexLP
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

solver = SimplexLP(layers)
with torch.no_grad():
    solver.define_linear_approximation((x_test, eps))

def print_bounds(lb, ub, batch_sz, y_sz):
    for i in range(batch_sz):
        for j in range(y_sz):
            print('f_{j}(x_{i}): {l:8.4f} <= f_{j}(x_{i}+delta) <= {u:8.4f}'.format(
                j=j, i=i, l=lb[i][j].item(), u=ub[i][j].item()))
            
print_bounds(solver.lower_bounds[-1], solver.upper_bounds[-1], batch_size, y_size)