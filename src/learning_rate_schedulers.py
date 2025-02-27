# 📌 Learning Rate Schedulers in Deep Learning

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ✅ Load MNIST Dataset

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# 🔹 Define a Simple Neural Network 🔹
class NeuralNet(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=128, output_size=10):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten images
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x