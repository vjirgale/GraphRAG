import torch

def check_cuda():
    """
    Checks for CUDA availability and prints detailed CUDA device information.
    """
    print("--- CUDA Availability Check ---")
    if torch.cuda.is_available():
        print("CUDA is available! Your PyTorch is built with CUDA version:", torch.version.cuda)
        num_devices = torch.cuda.device_count()
        print(f"Found {num_devices} CUDA device(s).")
        for i in range(num_devices):
            print(f"\n--- Device {i} ---")
            print(f"Device Name: {torch.cuda.get_device_name(i)}")
            print(f"Is current device: {torch.cuda.current_device() == i}")
            print(f"Memory Total: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
            print(f"Memory Free: {torch.cuda.memory_reserved(i) / (1024**3):.2f} GB (Reserved)")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(i) / (1024**3):.2f} GB (Allocated)")
    else:
        print("CUDA is not available. PyTorch will run on CPU.")
        print("Please check your CUDA installation and PyTorch-CUDA compatibility.")

if __name__ == "__main__":
    check_cuda()