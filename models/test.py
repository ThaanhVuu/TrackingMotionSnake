import torch


print("✅ CUDA available:", torch.cuda.is_available())
print("🧠 GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")
print("🔥 CUDA version:", torch.version.cuda)

