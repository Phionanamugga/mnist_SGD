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
    
# ✅ Model Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# ✅ Different Learning Rate Schedulers
schedulers = {
    "StepLR": optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5),
    "ExponentialLR": optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9),
    "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10),
    "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
}


# 🔹 Training Function 🔹
def train_model(scheduler, num_epochs=10):
    model.train()
    lr_history = []

    for epoch in range(num_epochs):
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Step the learning rate scheduler
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(loss)  # Special case for ReduceLROnPlateau
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}")

    return lr_history