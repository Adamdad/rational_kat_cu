import torch
import my_lib

x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
y = torch.tensor([1, 1, 1, 1, 1], dtype=torch.float32)
z = my_lib.add_cpu(x, y)
w = x + y
print(torch.allclose(z, w))  # True