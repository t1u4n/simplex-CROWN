import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as MNIST
from torch.utils.data import DataLoader
import argparse
from simple_mlp import SimpleMLP


def main(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = MNIST.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = MNIST.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    model = SimpleMLP(28*28, args.num_layers, args.neurons_per_layer, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            images = images.view(-1, 28*28)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.view(-1, 28*28)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total:.2f}%')

    torch.save(model, args.save_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a simple MLP on MNIST')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers in the MLP')
    parser.add_argument('--neurons_per_layer', type=int, default=100, help='Number of neurons per layer')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_model', type=str, default='mnist_mlp_model.pth', help='Filename for saved model')
    args = parser.parse_args()

    main(args)
