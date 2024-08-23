import torch
from torch import nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from kat import KAT_1DGroup
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

def erfc_softplus_squared_torch(x):
    softplus = torch.nn.Softplus()
    softplus_x = softplus(x)
    erfc_x = torch.erfc(softplus_x)
    return erfc_x ** 2

def train_and_benchmark(activation_func, func, label, epochs=1000, seed=42):
    set_random_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = activation_func.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    x = torch.linspace(-8, 8, 1000).unsqueeze(0).unsqueeze(0).to(device)
    y = func(x).to(device)
    model.train()
    start_time = time.time()
    for epoch in range(epochs):
        total_loss = 0
        outputs = model(x)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}')
    duration = time.time() - start_time
    print(f'{label} Training completed in {duration:.2f} seconds.')
    
    # Plot the output
    import matplotlib.pyplot as plt
    pred = model(x).squeeze(0).squeeze(0).detach().cpu().numpy()
    x = x.squeeze(0).squeeze(0).cpu().numpy()
    y = y.squeeze(0).squeeze(0).cpu().numpy()
    
    plt.plot(x, y, label='True')
    plt.plot(x, pred, label='Predicted')
    plt.xlabel("Input")
    
    plt.ylabel("Output")
    plt.title("Response of " + label)
    plt.grid(True)
    plt.legend()
    plt.savefig(label + ".png")
    

if __name__ == "__main__":
    func = erfc_softplus_squared_torch
    kat_activation = KAT_1DGroup(num_groups=1) # Placeholder for KAT_1DGroup if not accessible
    train_and_benchmark(kat_activation, func, 'KAT 1DGroup (as ReLU placeholder)')
    print(kat_activation.weight_numerator, kat_activation.weight_denominator)
