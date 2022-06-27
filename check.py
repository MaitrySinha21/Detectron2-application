import torch
gpu_avail = torch.cuda.is_available()
print(f"Is the GPU available? {gpu_avail}")