import torch
from torch import nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from rational.torch import Rational
import time
import numpy as np
import random
from kat_rational import KAT_1DGroup, KAT_1DGroupv2
from rational.torch import Rational

def set_random_seed(seed_value=42):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)
    random.seed(seed_value)

class CIFARNet(nn.Module):
    def __init__(self, activation_func):
        super(CIFARNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.act1 = activation_func()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.act2 = activation_func()
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.act3 = activation_func()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(self.act1(self.conv1(x)))
        x = self.pool(self.act2(self.conv2(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = self.act3(self.fc1(x))
        x = self.fc2(x)
        return x

def train_and_benchmark(activation_func, label, epochs=10, seed=42):
    set_random_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CIFARNet(activation_func).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
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
    
    # Testing phase
    model.eval()
    total_correct = 0
    total_images = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    accuracy = total_correct / total_images * 100
    duration = time.time() - start_time
    print(f'{label} Testing Accuracy: {accuracy:.2f}%, Total time: {duration:.2f} seconds.')

if __name__ == "__main__":
    # gelu = nn.GELU
    # train_and_benchmark(gelu, 'GELU')
    
    # rational_activation = Rational
    # train_and_benchmark(rational_activation, 'Rational GELU')

    # Placeholder for your custom KAT_1DGroup implementation
    # Assuming you have a similar API and it's suitable for convolutional layers
    kat_activation = KAT_1DGroupv2  # Replace with your actual KAT_1DGroup class if available
    train_and_benchmark(kat_activation, 'KAT 1DGroup (as ReLU placeholder)')
