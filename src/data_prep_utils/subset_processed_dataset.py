import argparse
import json
import os
import random
import shutil
from collections import defaultdict

from tqdm import tqdm


def load_labels(labels_path):
    with open(labels_path, 'r') as f:
        return json.load(f)


def get_balanced_subset(labels, artefacts, count_per_label):
    video_labels = defaultdict(dict)
    for video, details in labels.items():
        for artefact in artefacts:
            video_labels[video][artefact] = details.get(artefact, 0)

    final_subset = {}
    artefact_counts = {artefact: {'positive': 0, 'negative': 0} for artefact in artefacts}

    # Shuffle videos to ensure random selection
    shuffled_videos = list(video_labels.keys())
    random.shuffle(shuffled_videos)

    for video in shuffled_videos:
        include_video = True
        for artefact in artefacts:
            label = video_labels[video][artefact]
            if label == 1 and artefact_counts[artefact]['positive'] >= count_per_label:
                include_video = False
                break
            elif label == 0 and artefact_counts[artefact]['negative'] >= count_per_label:
                include_video = False
                break

        if include_video:
            final_subset[video] = video_labels[video]
            for artefact in artefacts:
                if video_labels[video][artefact] == 1:
                    artefact_counts[artefact]['positive'] += 1
                else:
                    artefact_counts[artefact]['negative'] += 1

        # Check if we have reached the target count for all artefacts
        if all(counts['positive'] >= count_per_label and counts['negative'] >= count_per_label
               for counts in artefact_counts.values()):
            break

    return final_subset


def copy_videos(videos, src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for video in tqdm(videos, desc=f"Copying to {os.path.basename(dst_dir)}"):
        src_path_part1 = os.path.join(src_dir, 'part1', video)
        src_path_part2 = os.path.join(src_dir, 'part2', video)
        dst_path = os.path.join(dst_dir, video)

        if os.path.exists(src_path_part1):
            shutil.copy2(src_path_part1, dst_path)
        elif os.path.exists(src_path_part2):
            shutil.copy2(src_path_part2, dst_path)
        else:
            print(f"Warning: Video {video} not found in either part1 or part2.")


def main():
    parser = argparse.ArgumentParser(description="Create a balanced subset of videos and relocate them.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to processed_BVIArtefact folder")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--count_per_label", type=int, default=500,
                        help="Number of videos per label (positive/negative)")
    args = parser.parse_args()

    # Load labels
    labels_path = os.path.join(args.input_dir, 'processed_labels.json')
    labels = load_labels(labels_path)

    # Define artefacts
    artefacts = ['']  # Add more labels as needed

    # Get balanced subset
    balanced_subset = get_balanced_subset(labels, artefacts, args.count_per_label)

    # Copy videos to output directory
    copy_videos(balanced_subset.keys(), args.input_dir, args.output_dir)

    # Save the subset labels
    subset_labels_path = os.path.join(args.output_dir, 'labels.json')
    with open(subset_labels_path, 'w') as f:
        json.dump(balanced_subset, f, indent=4)

    print(f"Balanced subset created in {args.output_dir}")
    print(f"Total videos in subset: {len(balanced_subset)}")

    # Verify the balance of the subset labels
    for artefact in artefacts:
        presence_count = sum(1 for labels in balanced_subset.values() if labels[artefact] == 1)
        absence_count = sum(1 for labels in balanced_subset.values() if labels[artefact] == 0)
        print(f"{artefact}:")
        print(f"  Presence count: {presence_count}")
        print(f"  Absence count: {absence_count}")


if __name__ == "__main__":
    main()

    # sample usage of the script
    # python subset_processed_dataset.py --input_dir /Volumes/SSD/preprocessed_BVIArtefact --output_dir /Volumes/SSD/balanced_subset --count_per_label 500
