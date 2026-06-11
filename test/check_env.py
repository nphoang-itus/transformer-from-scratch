import torch

print("PyTorch version:", torch.__version__)

x = torch.rand(2, 3)
print("Random tensor:")
print(x)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

x = x.to(device)
print("Tensor device:", x.device)