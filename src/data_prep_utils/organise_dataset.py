import json
import os
import shutil


def organize_dataset(root_dir, artifact_type='graininess'):
    # Define paths
    splits = ['train', 'val', 'test']

    for split in splits:
        split_dir = os.path.join(root_dir, split)
        labels_file = os.path.join(split_dir, 'labels.json')

        # Check if the labels file exists
        if not os.path.exists(labels_file):
            print(f"Labels file not found for {split} split.")
            continue

        # Read labels
        with open(labels_file, 'r') as f:
            labels = json.load(f)

        # Create directories and move files
        artifact_dir = os.path.join(split_dir, artifact_type)
        positive_dir = os.path.join(artifact_dir, 'positive')
        negative_dir = os.path.join(artifact_dir, 'negative')

        os.makedirs(positive_dir, exist_ok=True)
        os.makedirs(negative_dir, exist_ok=True)

        for video, label_dict in labels.items():
            if artifact_type not in label_dict:
                print(f"Warning: {artifact_type} label not found for {video}")
                continue

            label = label_dict[artifact_type]
            dst_dir = positive_dir if label == 1 else negative_dir

            # Move video file
            src = os.path.join(split_dir, video)
            dst = os.path.join(dst_dir, video)

            if os.path.exists(src):
                shutil.move(src, dst)
            else:
                print(f"Warning: {video} not found in {split} split.")

    print(f"Dataset organization for {artifact_type} complete.")


# Usage
if __name__ == "__main__":
    dataset_root = "/Volumes/SSD/graininess_balanced_subset_split"
    artifact_type = "graininess"  # Change this to organize for a different artifact type
    organize_dataset(dataset_root, artifact_type)
