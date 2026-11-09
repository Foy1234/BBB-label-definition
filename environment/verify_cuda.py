# verify_cuda.py — Reports the installed PyTorch version and CUDA runtime availability, inspects the selected accelerator, and executes a small matrix multiplication to check the active computation device.
import torch
print('PyTorch:', torch.__version__)
print('Compiled CUDA:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
    print('Compute capability:', torch.cuda.get_device_capability(0))
    a = torch.rand(32, 32, device='cuda')
    print('GPU multiplication result:', float((a @ a).sum().item()))
