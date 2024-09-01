import argparse
import json
import os
import random
import shutil

from tqdm import tqdm


def load_labels(labels_path):
    with open(labels_path, 'r') as f:
        return json.load(f)


def get_video_paths(input_dir):
    video_paths = {}
    for part in ['part1', 'part2']:
        part_dir = os.path.join(input_dir, part)
        for video in os.listdir(part_dir):
            video_paths[video] = os.path.join(part_dir, video)
    return video_paths


def get_maximum_balanced_subset(labels, video_paths):
    artefacts = set()
    for video_labels in labels.values():
        artefacts.update(video_labels.keys())

    balanced_subset = {}

    for artefact in artefacts:
        positive_videos = [video for video, video_labels in labels.items()
                           if video in video_paths and video_labels.get(artefact, 0) == 1]
        negative_videos = [video for video, video_labels in labels.items()
                           if video in video_paths and video_labels.get(artefact, 0) == 0]

        count_per_label = min(len(positive_videos), len(negative_videos))

        selected_positive = set(random.sample(positive_videos, count_per_label))
        selected_negative = set(random.sample(negative_videos, count_per_label))

        for video in selected_positive.union(selected_negative):
            if video not in balanced_subset:
                balanced_subset[video] = labels[video]
            balanced_subset[video][artefact] = 1 if video in selected_positive else 0

    return balanced_subset


def copy_videos(videos, video_paths, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for video in tqdm(videos, desc=f"Copying to {os.path.basename(dst_dir)}"):
        src_path = video_paths[video]
        dst_path = os.path.join(dst_dir, video)
        shutil.copy2(src_path, dst_path)


def create_subset_labels(balanced_subset):
    return balanced_subset


def main():
    parser = argparse.ArgumentParser(
        description="Create a maximum balanced subset of videos for all artefacts and relocate them.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to processed_BVIArtefact folder")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    args = parser.parse_args()

    labels_path = os.path.join(args.input_dir, 'processed_labels.json')
    labels = load_labels(labels_path)

    video_paths = get_video_paths(args.input_dir)

    balanced_subset = get_maximum_balanced_subset(labels, video_paths)

    copy_videos(balanced_subset.keys(), video_paths, args.output_dir)

    # Create and save the subset labels.json
    subset_labels = create_subset_labels(balanced_subset)
    labels_json_path = os.path.join(args.output_dir, 'labels.json')
    with open(labels_json_path, 'w') as f:
        json.dump(subset_labels, f, indent=4)

    print(f"Maximum balanced subset created in {args.output_dir}")
    print(f"Total videos in subset: {len(balanced_subset)}")
    print(f"Labels.json created at {labels_json_path}")

    artefacts = set()
    for video_labels in balanced_subset.values():
        artefacts.update(video_labels.keys())

    for artefact in sorted(artefacts):
        presence_count = sum(1 for labels in balanced_subset.values() if labels.get(artefact, 0) == 1)
        absence_count = sum(1 for labels in balanced_subset.values() if labels.get(artefact, 0) == 0)
        print(f"{artefact}:")
        print(f"  Presence count: {presence_count}")
        print(f"  Absence count: {absence_count}")


if __name__ == "__main__":
    main()
