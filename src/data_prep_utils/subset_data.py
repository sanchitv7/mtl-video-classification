import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from src.data_prep_utils.split_dataset import split_dataset

# Configuration
local_labels_path = 'data/bviArtefactMetaInfo/processed_labels.json'
artefacts_to_choose = ['graininess', 'aliasing', 'banding', 'motion_blur']  # Add more labels as needed
size_limit_gb = 4  # Size limit in GB

part1_sizes_path = 'data/bviArtefactMetaInfo/part1_files_sizes.txt'
part2_sizes_path = 'data/bviArtefactMetaInfo/part2_files_sizes.txt'


def convert_to_bytes(size_str):
    size_unit = size_str[-1]
    size_value = float(size_str[:-1])
    if size_unit == 'G':
        return int(size_value * 1e9)
    elif size_unit == 'M':
        return int(size_value * 1e6)
    elif size_unit == 'K':
        return int(size_value * 1e3)
    else:
        return int(size_value)


def load_file_sizes(file_path):
    file_sizes = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            file_name = parts[0]
            file_size = convert_to_bytes(parts[1])
            file_sizes[file_name] = file_size
    return file_sizes


def get_balanced_videos(labels, artefacts, size_limit):
    video_labels = defaultdict(dict)
    for video, details in labels.items():
        for artefact in artefacts:
            video_labels[video][artefact] = details.get(artefact, 0)

    # Separate positive and negative videos
    positive_videos = [v for v, l in video_labels.items() if all(l[a] == 1 for a in artefacts)]
    negative_videos = [v for v, l in video_labels.items() if all(l[a] == 0 for a in artefacts)]

    # Sort videos by size (smallest to largest)
    positive_videos.sort(key=lambda x: file_sizes.get(x, 0))
    negative_videos.sort(key=lambda x: file_sizes.get(x, 0))

    balanced_videos = []
    total_size = 0

    print(f"Size limit: {size_limit / 1e9:.2f} GB")
    print(f"Total positive videos available: {len(positive_videos)}")
    print(f"Total negative videos available: {len(negative_videos)}")

    # Select videos while maintaining balance and respecting size limit
    for pos, neg in zip(positive_videos, negative_videos):
        pos_size = file_sizes.get(pos, 0)
        neg_size = file_sizes.get(neg, 0)

        if total_size + pos_size + neg_size <= size_limit:
            balanced_videos.extend([pos, neg])
            total_size += pos_size + neg_size
        else:
            break

    final_subset = {video: video_labels[video] for video in balanced_videos}

    final_size = sum(file_sizes.get(video, 0) for video in final_subset)
    print(f"\nFinal balanced dataset:")
    print(f"Size: {final_size / 1e9:.2f} GB")
    print(f"Total videos: {len(final_subset)}")
    print(f"Positive videos: {len(final_subset) // 2}")
    print(f"Negative videos: {len(final_subset) // 2}")

    return final_subset


def copy_videos_local(subset_videos, source_base_path, destination_base_path):
    progress_bar = tqdm(total=len(subset_videos), desc="Copying videos", unit="file", dynamic_ncols=True)

    for video in subset_videos:
        found = False
        for part in ['part1', 'part2']:
            source_path = os.path.join(source_base_path, part, video)
            destination_path = os.path.join(destination_base_path, video)
            if os.path.exists(source_path):
                progress_bar.set_postfix(file=video)
                shutil.copy2(source_path, destination_path)
                found = True
                break
        if not found:
            print(f"Video {video} not found in either part1 or part2.")
        progress_bar.update(1)

    progress_bar.close()


def main():
    parser = argparse.ArgumentParser(description="Create a balanced subset of videos for multi-label classification.")
    parser.add_argument("--local", help="Path to local bviDataset folder", type=str, required=True)
    parser.add_argument("--size_limit", help="Size limit in GB", type=float, default=2.0)
    args = parser.parse_args()

    global size_limit_gb
    size_limit_gb = args.size_limit

    # Load file sizes
    part1_file_sizes = load_file_sizes(part1_sizes_path)
    part2_file_sizes = load_file_sizes(part2_sizes_path)
    global file_sizes
    file_sizes = {**part1_file_sizes, **part2_file_sizes}

    # Load labels
    with open(local_labels_path, 'r') as f:
        labels = json.load(f)

    size_limit_bytes = size_limit_gb * 1e9
    balanced_subset = get_balanced_videos(labels, artefacts_to_choose, size_limit_bytes)

    # Create the local download directory
    local_download_dir = f'/Volumes/SSD/subsets/{"_".join([art for art in artefacts_to_choose])}_subset_{int(size_limit_gb)}_GB'
    os.makedirs(local_download_dir, exist_ok=True)

    # Save the subset list locally
    subset_file_path = f'{local_download_dir}/labels.json'
    with open(subset_file_path, 'w') as f:
        json.dump(balanced_subset, f, indent=4)

    print(f"Balanced subset saved to {subset_file_path}")

    # Verify the balance of the subset labels
    for artefact in artefacts_to_choose:
        presence_count = sum(1 for labels in balanced_subset.values() if labels[artefact] == 1)
        absence_count = sum(1 for labels in balanced_subset.values() if labels[artefact] == 0)
        print(f"{artefact}:")
        print(f"  Presence count: {presence_count}")
        print(f"  Absence count: {absence_count}")

    # Use local dataset
    print(f"Using local dataset at: {args.local}")
    copy_videos_local(balanced_subset.keys(), args.local, local_download_dir)

    print(f"All raw videos copied to {local_download_dir}")

    split_dataset(local_download_dir)


if __name__ == "__main__":
    main()
