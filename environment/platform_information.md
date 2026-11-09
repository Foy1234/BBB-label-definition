# Reference Compute Environment

- Operating system: Ubuntu 22.04, x86_64 Linux
- Python: 3.12.3
- GPU: NVIDIA V100 32 GB (1 device)
- CPU: Intel Xeon Skylake, 10 virtual CPUs
- RAM: 72 GB
- System disk: 30 GB
- CUDA: 12.8
- PyTorch: 2.8.0+cu128
- PyTorch Geometric: 2.8.0.post1
- RDKit: 2026.3.5
- XGBoost: 3.2.0
- NumPy: 2.4.6; pandas: 3.0.5; SciPy: 1.17.1; scikit-learn: 1.9.0; Matplotlib: 3.10.5

These versions are from the original AutoDL package listing. CUDA availability and architecture compatibility depend on the installed driver and actual PyTorch wheel; `verify_cuda.py` performs a device-level check. The complete historical pip freeze contains image-specific local package links and is not itself used as a portable installation file.
