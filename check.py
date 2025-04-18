import numpy as np

# Load and check the original `.npy` file
original_npy = "videos/push_up_260.npy"
output_npy = "videos/output_video.npy"

def check_npy_shape(npy_path):
    data = np.load(npy_path)
    print(f"📂 File: {npy_path}")
    print(f"🔍 Shape: {data.shape}")
    print(f"🔢 Data Type: {data.dtype}")
    print(f"📊 Min: {data.min()}, Max: {data.max()}")
    print("-" * 50)

check_npy_shape(original_npy)
check_npy_shape(output_npy)
