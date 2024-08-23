import torch
from kat import KAT_1DGroup
from rational.torch import Rational
from torch import nn
import time
import torch.optim as optim

    
import torch
from torch import nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from rational.torch import Rational
import time
import numpy as np
import random

def set_random_seed(seed_value=42):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)
    random.seed(seed_value)

class NeuralNet(nn.Module):
    def __init__(self, activation_func):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.activation = activation_func
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.activation(x.unsqueeze(1)).squeeze(1)
        x = self.fc2(x)
        return x

def train_and_benchmark(activation_func, label, epochs=10, seed=42):
    set_random_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNet(activation_func).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model.train()
    start_time = time.time()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'{label} - Epoch {epoch+1}: Loss {total_loss / len(data_loader)}')
    duration = time.time() - start_time
    print(f'{label} Training completed in {duration:.2f} seconds.')

if __name__ == "__main__":
    rational_activation = Rational(approx_func="gelu")
    kat_activation = KAT_1DGroup(num_groups=8, init_mode="gelu") # Placeholder for KAT_1DGroup if not accessible

    train_and_benchmark(rational_activation, 'Rational GELU')
    train_and_benchmark(kat_activation, 'KAT 1DGroup (as ReLU placeholder)')
