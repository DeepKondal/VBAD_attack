import os
import argparse
import subprocess
import torch
import multiprocessing

# Argument Parsing
parser = argparse.ArgumentParser(description="Batch process videos using VBAD attack pipeline")
parser.add_argument("--video_list", type=str, required=True, help="Path to the video list file")
parser.add_argument("--sigma", type=float, default=1e-3, help="Sigma value for VBAD attack (default: 1e-3)")
args = parser.parse_args()

# Directories
RAW_VIDEOS_DIR = "raw_videos"
VIDEOS_DIR = "videos"
OUTPUT_VIDEOS_TARGETED = "output_videos_targeted"
OUTPUT_VIDEOS_UNTARGETED = "output_videos_untargeted"
FINAL_VIDEOS_TARGETED = "final_perturbed_videos/targeted"
FINAL_VIDEOS_UNTARGETED = "final_perturbed_videos/untargeted"

# Ensure necessary directories exist
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(OUTPUT_VIDEOS_TARGETED, exist_ok=True)
os.makedirs(OUTPUT_VIDEOS_UNTARGETED, exist_ok=True)
os.makedirs(FINAL_VIDEOS_TARGETED, exist_ok=True)
os.makedirs(FINAL_VIDEOS_UNTARGETED, exist_ok=True)

# Read video_list.txt
with open(args.video_list, "r") as f:
    video_entries = [line.strip().split() for line in f.readlines()]

def run_attack(video_name, label, attack_type):
    """Runs the attack process (targeted or untargeted) for a given video."""
    npy_path = os.path.join(VIDEOS_DIR, f"{os.path.splitext(video_name)[0]}.npy")
    
    # Choose correct output folder based on attack type
    if attack_type == "targeted":
        output_npy_path = os.path.join(OUTPUT_VIDEOS_TARGETED, f"output_{label}.npy")
    else:
        output_npy_path = os.path.join(OUTPUT_VIDEOS_UNTARGETED, f"output_{label}.npy")

    if not os.path.exists(output_npy_path):
        print(f"⚡ Running **{attack_type} attack** on {npy_path} → {output_npy_path}")

        attack_command = [
            "python", "main.py",
            "--gpus", "0",
            "--video", npy_path,
            "--label", label,
            "--adv-save-path", output_npy_path,
            "--sigma", str(args.sigma)
        ]
        if attack_type == "untargeted":
            attack_command.append("--untargeted")  # Add the --untargeted flag

        subprocess.run(attack_command, check=True)
        torch.cuda.empty_cache()  # Free GPU memory after each attack
    else:
        print(f"✅ {output_npy_path} already exists, skipping {attack_type} attack.")

# Step 2: Run VBAD attack in Parallel (LIMITED CONCURRENCY)
processes = []

with multiprocessing.Pool(processes=1) as pool:  # Limits to 2 attacks at a time
    for video_name, label in video_entries:
        processes.append(pool.apply_async(run_attack, (video_name, label, "targeted")))
        processes.append(pool.apply_async(run_attack, (video_name, label, "untargeted")))

    # Wait for all processes to complete
    for p in processes:
        p.get()

# Step 3: Convert adversarial .npy videos to .mp4 (and store them separately)
for video_name, label in video_entries:
    output_targeted_npy = os.path.join(OUTPUT_VIDEOS_TARGETED, f"output_{label}.npy")
    output_untargeted_npy = os.path.join(OUTPUT_VIDEOS_UNTARGETED, f"output_{label}.npy")
    final_targeted_video = os.path.join(FINAL_VIDEOS_TARGETED, f"adv_{label}.mp4")
    final_untargeted_video = os.path.join(FINAL_VIDEOS_UNTARGETED, f"adv_{label}.mp4")

    # Convert **Targeted Attack Output**
    if os.path.exists(output_targeted_npy) and not os.path.exists(final_targeted_video):
        print(f"🎬 Converting {output_targeted_npy} → {final_targeted_video}")
        subprocess.run(["python", "convert_npy_to_mp4.py", "--npy", output_targeted_npy, "--output", final_targeted_video, "--fps", "30"])

    # Convert **Untargeted Attack Output**
    if os.path.exists(output_untargeted_npy) and not os.path.exists(final_untargeted_video):
        print(f"🎬 Converting {output_untargeted_npy} → {final_untargeted_video}")
        subprocess.run(["python", "convert_npy_to_mp4.py", "--npy", output_untargeted_npy, "--output", final_untargeted_video, "--fps", "30"])

print("🎉 Batch processing completed!")
