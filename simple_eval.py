import torch
import torch.nn as nn
from simplexcrown.simplex_solver import SimplexSolver

if __name__=="__main__":
    def print_bounds(lb, ub, batch_sz, y_sz):
        for i in range(batch_sz):
            for j in range(y_sz):
                assert lb[i][j] <= output[i][j] and ub[i][j] >= output[i][j]
                print('f_{j}(x_{i}): {l:8.4f} <= f_{j}(x_{i}+delta) <= {u:8.4f}'.format(
                    j=j, i=i, l=lb[i][j].item(), u=ub[i][j].item()))

    model = nn.Sequential(
                nn.Linear(28*28, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
            )
    model.load_state_dict(torch.load('models/relu_model.pth'))
    model.eval()

    x_test, label = torch.load('data/data2.pth')
    batch_size = x_test.size(0)
    x_test = x_test.reshape(batch_size, -1)
    output = model(x_test)
    y_size = output.size(1)

    eps = 10

    solver = SimplexSolver(method='alpha-simplex', loss_fn='naive')
    lb, ub = solver.compute_bounds(model, x_test, eps, label, 50, 1e-1)
    
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
            diff = lirpa_diff[i][j].item() - simplex_diff[i][j].item()
            diff /= abs(lirpa_diff[i][j].item())
            total_diff += max(0, diff)
            print("f_{}: {:+.4f}%".format(j, diff * 100))

    print('\nAverage relative difference: {:+.4f}%'.format(total_diff/(batch_size) * 100))