import torch.nn as nn

class SimpleMLP(nn.Sequential):
    def __init__(self, input_size, num_layers, neurons_per_layer, output_size):
        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_size, neurons_per_layer))
            layers.append(nn.ReLU())
            input_size = neurons_per_layer
        layers.append(nn.Linear(neurons_per_layer, output_size))
        super(SimpleMLP, self).__init__(*layers)