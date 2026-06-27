import torch

device = "cuda"

a = torch.tensor([[1.,2.], [3.,4.]]).to(device)
b = torch.tensor([[5.,6.], [7.,8.]]).to(device)


c = a @ b

print(c)
print(c.device)