import os
import argparse
import subprocess

import torch

# Argument Parsing
parser = argparse.ArgumentParser(description="Batch process videos using VBAD attack pipeline")
parser.add_argument("--video_list", type=str, required=True, help="Path to the video list file")
parser.add_argument("--sigma", type=float, default=1e-3, help="Sigma value for VBAD attack (default: 1e-3)")
parser.add_argument("--untargeted", action="store_true", help="Enable untargeted attack mode")
args = parser.parse_args()

# Directories
RAW_VIDEOS_DIR = "raw_videos"
VIDEOS_DIR = "videos"
OUTPUT_VIDEOS_DIR = "output_videos"
FINAL_VIDEOS_DIR = "final_pertubed_videos"

# Ensure necessary directories exist
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(OUTPUT_VIDEOS_DIR, exist_ok=True)
os.makedirs(FINAL_VIDEOS_DIR, exist_ok=True)

# Read video_list.txt
with open(args.video_list, "r") as f:
    video_entries = [line.strip().split() for line in f.readlines()]

# Step 1: Convert videos to .npy
for video_name, label in video_entries:
    video_path = os.path.join(RAW_VIDEOS_DIR, video_name)
    npy_path = os.path.join(VIDEOS_DIR, f"{os.path.splitext(video_name)[0]}.npy")

    if not os.path.exists(npy_path):  # Avoid redundant conversion
        print(f"🎥 Converting {video_name} → {npy_path}")
        subprocess.run(["python", "convert_mp4_to_npy.py", "--video", video_path])
        torch.cuda.empty_cache()  # Clear CUDA memory after each attack
    else:
        print(f"✅ {npy_path} already exists, skipping conversion.")

# Step 2: Run VBAD attack
for video_name, label in video_entries:
    npy_path = os.path.join(VIDEOS_DIR, f"{os.path.splitext(video_name)[0]}.npy")
    output_npy_path = os.path.join(OUTPUT_VIDEOS_DIR, f"output_{label}.npy")

    print(f"📌 Processing {video_name} → {npy_path}")
    print(f"📌 Expected output: {output_npy_path}")

    if not os.path.exists(output_npy_path):  # Avoid redundant attacks
        print(f"⚡ Running attack on {npy_path} → {output_npy_path}")
        attack_command = [
            "python", "main.py",
            "--gpus", "0",
            "--video", npy_path,
            "--label", label,
            "--adv-save-path", output_npy_path,
            "--sigma", str(args.sigma),
        ]
        if args.untargeted:
            attack_command.append("--untargeted")  # Add --untargeted if specified
        subprocess.run(attack_command, check=True)
        torch.cuda.empty_cache()
    else:
        print(f"✅ {output_npy_path} already exists, skipping attack.")




# Step 3: Convert adversarial .npy videos to .mp4
for video_name, label in video_entries:
    output_npy_path = os.path.join(OUTPUT_VIDEOS_DIR, f"output_{label}.npy")
    final_video_path = os.path.join(FINAL_VIDEOS_DIR, f"adv_{label}.mp4")

    if os.path.exists(output_npy_path) and not os.path.exists(final_video_path):  # Avoid redundant conversion
        print(f"🎬 Converting {output_npy_path} → {final_video_path}")
        subprocess.run(["python", "convert_npy_to_mp4.py", "--npy", output_npy_path, "--output", final_video_path, "--fps", "30"])
        torch.cuda.empty_cache()  # Clear CUDA memory after each attack
    else:
        print(f"✅ {final_video_path} already exists, skipping conversion.")

print("🎉 Batch processing completed!")
