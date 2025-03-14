import torch
import numpy as np
import argparse
from inception_i3d.pytorch_i3d import InceptionI3d

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, required=True, help='Path to the input video .npy file')
args = parser.parse_args()

# Load Pretrained I3D Model
i3d = InceptionI3d(400, in_channels=3)
i3d.load_state_dict(torch.load('inception_i3d/models/rgb_imagenet.pt'))
i3d.to('cpu')
i3d.eval()

# Load Video
vid = np.load(args.video)
vid = torch.tensor(vid, dtype=torch.float, device='cpu')

# Ensure correct format: (batch, channels, frames, height, width)
if vid.shape[-1] == 3:  # If video is in (frames, height, width, channels) format
    vid = vid.permute(3, 0, 1, 2)  # Convert to (channels, frames, height, width)

vid = vid.unsqueeze(0)  # Add batch dimension -> (1, 3, frames, height, width)

# Run Inference
with torch.no_grad():
    logits = i3d(vid).mean(2)  # Temporal pooling
    pred_label = torch.argmax(logits, dim=1).item()

print(f"Predicted Label: {pred_label}")
