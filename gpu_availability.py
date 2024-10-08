import torch

print("CUDA availability: ", torch.cuda.is_available())
print("CUDA device count: ", torch.cuda.device_count())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")