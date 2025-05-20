import os
import torch

# --- Start of new diagnostic block ---
print("--- PyTorch CUDA Diagnostics ---", flush=True)
print(f"PYTHON EXECUTABLE: {os.sys.executable}", flush=True) # To be super sure which python is running this
print(f"TORCH VERSION: {torch.__version__}", flush=True)
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}", flush=True)

if torch.cuda.is_available():
    print(f"torch.version.cuda: {torch.version.cuda}", flush=True)
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}", flush=True)
    try:
        for i in range(torch.cuda.device_count()):
            print(f"torch.cuda.get_device_name({i}): {torch.cuda.get_device_name(i)}", flush=True)
            print(f"torch.cuda.get_device_capability({i}): {torch.cuda.get_device_capability(i)}", flush=True)
        print(f"torch.cuda.current_device(): {torch.cuda.current_device()}", flush=True)
    except Exception as e:
        print(f"Error during detailed CUDA device info: {e}", flush=True)
else:
    print("CUDA is NOT available to PyTorch in this environment.", flush=True)
    # Try to get more info if PyTorch was compiled with CUDA but it's not usable
    if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'is_built'):
         print(f"torch.backends.cuda.is_built(): {torch.backends.cuda.is_built()}", flush=True)
    else:
        print("Cannot determine if PyTorch was built with CUDA support (old PyTorch version or unusual build).", flush=True)

print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}", flush=True)
print("--- End of PyTorch CUDA Diagnostics ---", flush=True)
# --- End of new diagnostic block ---

# Your existing device selection logic
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# The rest of your script...