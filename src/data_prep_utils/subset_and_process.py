import os
import json
import random
from collections import Counter

import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from sklearn.model_selection import train_test_split

# Argument parser
parser = argparse.ArgumentParser(description='Preprocess BVIArtefact dataset')
parser.add_argument('--input_dir', type=str, default="/Volumes/SSD/BVIArtefact",
                    help='Input directory containing BVIArtefact dataset')
parser.add_argument('--output_dir', type=str, default="/Volumes/SSD/BVIArtefact_8_crops_all_videos",
                    help='Output directory for preprocessed data')
parser.add_argument('--num_samples', type=int, default=None, help='Number of videos to sample (None for all)')
parser.add_argument('--crop_size', type=int, default=224, help='Size of spatial crop')
parser.add_argument('--num_frames', type=int, default=8, help='Number of frames to extract')
parser.add_argument('--crops_per_video', type=int, default=4, help='Number of crops to extract per video')
parser.add_argument('--train_ratio', type=float, default=0.7, help='Ratio of videos for training set')
parser.add_argument('--val_ratio', type=float, default=0.15, help='Ratio of videos for validation set')
args = parser.parse_args()

# Configuration
INPUT_DIR = args.input_dir
OUTPUT_DIR = args.output_dir
LABELS_FILE = os.path.join(INPUT_DIR, "labels.json")
CROP_SIZE = (args.crop_size, args.crop_size)
NUM_FRAMES = args.num_frames
NUM_CROPS_PER_VIDEO = args.crops_per_video

random.seed(42)

# Create output directories
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

# Load labels
with open(LABELS_FILE, 'r') as f:
    labels = json.load(f)


def parse_size(size_str):
    """Convert size string to bytes"""
    size = float(size_str[:-1])
    unit = size_str[-1]
    if unit == 'G':
        return int(size * 1e9)
    elif unit == 'M':
        return int(size * 1e6)
    else:
        return int(size)


def read_file_sizes(filename):
    """Read file sizes from text file"""
    sizes = {}
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                sizes[parts[0]] = parse_size(parts[1])
    return sizes


def extract_random_crop(frames, num_frames, crop_size):
    """Extract a random spatio-temporal crop from the frames."""
    t, h, w, _ = frames.shape

    if t < num_frames:
        raise ValueError(f"Video has fewer frames ({t}) than required ({num_frames})")

    start_frame = random.randint(0, t - num_frames)
    top = random.randint(0, h - crop_size[0])
    left = random.randint(0, w - crop_size[1])

    crop = frames[start_frame:start_frame + num_frames,
           top:top + crop_size[0],
           left:left + crop_size[1]]

    return crop


def normalize(video, mean, std):
    """Normalize the video tensor"""
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    return (video - mean) / std


def process_videos(video_list, split):
    """Process videos and save crops for a specific split"""
    preprocessed_labels = {}
    label_counts = Counter()
    total_crops = 0

    for video_file, video_name in tqdm(video_list, desc=f"Processing {split} set"):
        video_path = os.path.join(INPUT_DIR, video_file)

        # Skip if video is not in labels
        if video_name not in labels:
            print(f"Skipping {video_file}: No labels found")
            continue

        video_labels = labels[video_name]

        try:
            # Read video
            cap = cv2.VideoCapture(video_path)
            frames = []
            while len(frames) < NUM_FRAMES * 2:  # Read more frames than needed
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

            if len(frames) < NUM_FRAMES:
                print(f"Warning: {video_file} has fewer than {NUM_FRAMES} frames. Skipping.")
                continue

            frames = np.array(frames)

            for i in range(NUM_CROPS_PER_VIDEO):
                # Extract random crop
                crop = extract_random_crop(frames, NUM_FRAMES, CROP_SIZE)

                # Convert to torch tensor and normalize
                crop = torch.from_numpy(crop).permute(0, 3, 1, 2).float() / 255.0

                # Normalize using ImageNet stats
                crop = normalize(crop, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

                # Generate unique filename for the crop
                crop_filename = f"{Path(video_name).stem}_crop_{i}.pt"
                crop_path = os.path.join(OUTPUT_DIR, split, crop_filename)

                # Save crop as .pt file
                torch.save(crop, crop_path)

                # Store labels for the crop
                preprocessed_labels[crop_filename] = video_labels

                total_crops += 1

            # Update label counts
            for artifact, present in video_labels.items():
                if present == 1:
                    label_counts[f"{artifact}_Positive"] += NUM_CROPS_PER_VIDEO
                else:
                    label_counts[f"{artifact}_Negative"] += NUM_CROPS_PER_VIDEO

        except Exception as e:
            print(f"Error processing {video_file}: {str(e)}")

    # Save preprocessed labels
    labels_path = os.path.join(OUTPUT_DIR, split, "labels.json")
    with open(labels_path, 'w') as f:
        json.dump(preprocessed_labels, f, indent=4)

    print(f"\n{split} set statistics:")
    print(f"Total crops generated: {total_crops}")
    print(f"Number of entries in labels JSON: {len(preprocessed_labels)}")

    # Check if numbers match
    if total_crops == len(preprocessed_labels):
        print("✅ Numbers match!")
    else:
        print("❌ Numbers don't match. There might be an issue.")

    return label_counts, total_crops


def check_split_overlap(output_dir):
    splits = ['train', 'val', 'test']
    parent_videos = {split: set() for split in splits}

    for split in splits:
        labels_path = Path(output_dir) / split / "labels.json"
        with open(labels_path, 'r') as f:
            labels = json.load(f)

        for crop_filename in labels.keys():
            # Extract parent video name by removing the "_crop_{i}.pt" suffix
            parent_video = crop_filename.rsplit('_crop_', 1)[0]
            parent_videos[split].add(parent_video)

    # Check for overlap between splits
    for i, split1 in enumerate(splits):
        for split2 in splits[i + 1:]:
            overlap = parent_videos[split1].intersection(parent_videos[split2])
            if overlap:
                print(f"❌ Overlap found between {split1} and {split2} splits:")
                print(f"   Common parent videos: {overlap}")
            else:
                print(f"✅ No overlap found between {split1} and {split2} splits")

    # Print summary
    print("\nSummary:")
    for split in splits:
        print(f"{split} split: {len(parent_videos[split])} unique parent videos")


def print_label_balance(label_counts, split_name):
    print(f"\n{split_name} set balance:")
    artifacts = ['black_screen', 'frame_drop', 'spatial_blur', 'transmission_error', 'aliasing', 'banding',
                 'dark_scenes', 'graininess', 'motion_blur']
    for artifact in artifacts:
        positive = label_counts[f"{artifact}_Positive"]
        negative = label_counts[f"{artifact}_Negative"]
        print(f"    {artifact}: Positive: {positive}, Negative: {negative}")


# Read file sizes
part1_sizes = read_file_sizes(os.path.join(INPUT_DIR, "part1_files_sizes.txt"))
part2_sizes = read_file_sizes(os.path.join(INPUT_DIR, "part2_files_sizes.txt"))

all_sizes = {**part1_sizes, **part2_sizes}

# Sort videos by size
sorted_videos = sorted(all_sizes.items(), key=lambda x: x[1])

# Sample videos if num_samples is specified
if args.num_samples is not None:
    sampled_videos = sorted_videos[:args.num_samples]
else:
    sampled_videos = sorted_videos

# Extract video files and their corresponding folders
video_files = [(os.path.join('part1' if f in part1_sizes else 'part2', f), f) for f, _ in sampled_videos]

# Split videos into train, validation, and test sets
train_videos, temp_videos = train_test_split(video_files, train_size=args.train_ratio, random_state=42)
val_ratio = args.val_ratio / (1 - args.train_ratio)
val_videos, test_videos = train_test_split(temp_videos, train_size=val_ratio, random_state=42)

# Modify the main part of the script to use the updated function
train_label_counts, train_crops = process_videos(train_videos, 'train')
val_label_counts, val_crops = process_videos(val_videos, 'val')
test_label_counts, test_crops = process_videos(test_videos, 'test')

# Add a final summary
print("\nFinal Summary:")
print(f"Total crops - Train: {train_crops}, Val: {val_crops}, Test: {test_crops}")
total_crops = train_crops + val_crops + test_crops
print(f"Total crops across all splits: {total_crops}")

# Check total number of label entries
train_labels = json.load(open(os.path.join(OUTPUT_DIR, 'train', 'labels.json')))
val_labels = json.load(open(os.path.join(OUTPUT_DIR, 'val', 'labels.json')))
test_labels = json.load(open(os.path.join(OUTPUT_DIR, 'test', 'labels.json')))

total_label_entries = len(train_labels) + len(val_labels) + len(test_labels)
print(f"Total label entries across all splits: {total_label_entries}")

if total_crops == total_label_entries:
    print("✅ Total crops match total label entries!")
else:
    print("❌ Total crops and total label entries don't match. There might be an issue.")

print_label_balance(train_label_counts, "Train")
print_label_balance(val_label_counts, "Val")
print_label_balance(test_label_counts, "Test")

check_split_overlap(OUTPUT_DIR)

print("Preprocessing completed.")

# sample usage of this script:
# python src/subset_and_process.py --input_dir /Volumes/SSD/BVIArtefact --output_dir /Volumes/SSD/BVIArtefact_crops --num_samples 100 --crop_size 224 --num_frames 8 --crops_per_video 2 --train_ratio 0.7 --val_ratio 0.15
