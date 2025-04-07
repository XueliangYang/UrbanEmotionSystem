import torch

   # Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

   # Optionally, check if CUDA is available and print additional info
if device.type == 'cuda':
       print(f"CUDA is available. Number of GPUs: {torch.cuda.device_count()}")
       print(f"Current GPU: {torch.cuda.get_device_name(0)}")
else:
       print("CUDA is not available. Running on CPU.")