import random
import os
import json
import shutil
from collections import defaultdict
from pathlib import Path


def split_dataset(preprocessed_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Load labels
    with open(os.path.join(preprocessed_dir, 'preprocessed_labels.json'), 'r') as f:
        labels = json.load(f)

    # Group crops by artifacts
    artifact_crops = defaultdict(lambda: {'positive': set(), 'negative': set()})
    for crop, artifacts in labels.items():
        for artifact, value in artifacts.items():
            if value == 1:
                artifact_crops[artifact]['positive'].add(crop)
            else:
                artifact_crops[artifact]['negative'].add(crop)

    # Find the minimum number of crops for any artifact
    min_pos = min(len(crops['positive']) for crops in artifact_crops.values())
    min_neg = min(len(crops['negative']) for crops in artifact_crops.values())
    min_crops = min(min_pos, min_neg) * 2  # Ensure balance between positive and negative

    # Calculate the number of crops for each split
    train_size = int(min_crops * train_ratio)
    val_size = int(min_crops * val_ratio)
    test_size = min_crops - train_size - val_size

    splits = {'train': set(), 'val': set(), 'test': set()}
    split_artifacts = {split: defaultdict(lambda: {'positive': set(), 'negative': set()}) for split in splits}

    # Distribute crops ensuring balance for each artifact in each split
    for split, size in [('train', train_size), ('val', val_size), ('test', test_size)]:
        pos_count = size // 2
        neg_count = size - pos_count

        for artifact, crops in artifact_crops.items():
            pos_crops = list(crops['positive'])
            neg_crops = list(crops['negative'])
            random.shuffle(pos_crops)
            random.shuffle(neg_crops)

            for _ in range(pos_count):
                if pos_crops:
                    crop = pos_crops.pop()
                    if crop not in splits['train'] and crop not in splits['val'] and crop not in splits['test']:
                        splits[split].add(crop)
                        split_artifacts[split][artifact]['positive'].add(crop)

            for _ in range(neg_count):
                if neg_crops:
                    crop = neg_crops.pop()
                    if crop not in splits['train'] and crop not in splits['val'] and crop not in splits['test']:
                        splits[split].add(crop)
                        split_artifacts[split][artifact]['negative'].add(crop)

    # Create directories and move crops
    preprocessed_dir_path = Path(preprocessed_dir)
    data_split_path = preprocessed_dir_path.parent / str(preprocessed_dir_path.name + "_split")

    for split, crops in splits.items():
        os.makedirs(data_split_path / split, exist_ok=True)
        split_labels = {}
        for crop in crops:
            src = os.path.join(preprocessed_dir, crop)
            dst = os.path.join(data_split_path, split, crop)
            shutil.copy(src, dst)  # Use copy instead of move to preserve original data
            split_labels[crop] = labels[crop]
        with open(os.path.join(data_split_path, split, 'labels.json'), 'w') as f:
            json.dump(split_labels, f, indent=2)

    print("Dataset split complete")
    print(f"Train set: {len(splits['train'])} crops")
    print(f"Validation set: {len(splits['val'])} crops")
    print(f"Test set: {len(splits['test'])} crops")

    # Print balance information for each artifact in each split
    for split in splits:
        print(f"\n{split.capitalize()} set balance:")
        for artifact in artifact_crops:
            pos = len(split_artifacts[split][artifact]['positive'])
            neg = len(split_artifacts[split][artifact]['negative'])
            print(f"  {artifact}: Positive: {pos}, Negative: {neg}")


if __name__ == "__main__":
    preprocessed_dir = "/Volumes/SSD/BVIArtefact_crops"  # Update this to your preprocessed dataset path
    split_dataset(preprocessed_dir)
