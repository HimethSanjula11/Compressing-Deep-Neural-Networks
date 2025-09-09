import torch

# Check if CUDA GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Training on GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("⚠️ Training on CPU")

# Example usage: move a model to the selected device
# model = model.to(device)