from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from src.data.mnist import load_mnist
from src.network.mlp import MultiPerceptron


def main(num_epochs=5):
    train_loader, valid_loader, test_loader = load_mnist()
    device = 'cpu'
    net = MultiPerceptron().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    print('training start...')
    for epoch in range(num_epochs):
        loss, correct = 0, 0
        net.train()
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, labels = inputs.view(-1, 28*28*1).to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss += loss.item()
            _, prediction = torch.max(outputs.data, 1)
            correct += (prediction == labels).sum().item()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train.Loss: {round(float(loss / len(train_loader.dataset)), 5)} "
              f"Train.Acc: {round(float(correct / len(train_loader.dataset)), 5)} ", end='')

        loss, correct = 0, 0
        net.eval()
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.view(-1, 28 * 28 * 1).to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss += loss.item()
                _, prediction = torch.max(outputs.data, 1)
                correct += (prediction == labels).sum().item()
        print(f"Valid.Loss: {round(float(loss / len(valid_loader.dataset)), 5)} "
              f"Valid.Acc: {round(float(correct / len(valid_loader.dataset)), 5)}")

    total, correct = 0, 0
    net.eval()
    with torch.no_grad():
        for (inputs, labels) in test_loader:
            inputs, labels = inputs.view(-1, 28 * 28 * 1).to(device), labels.to(device)
            outputs = net(inputs)
            _, prediction = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (prediction == labels).sum().item()
    print(f"Test.Acc: {100 * float(correct / total)}")


if __name__ == '__main__':
    main()
