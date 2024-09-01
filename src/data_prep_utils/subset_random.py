import os
import json
import random


def get_file_sizes(file_path):
    sizes = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                filename, size = parts
                sizes[filename] = int(size[:-1])  # Remove 'M' and convert to int
    return sizes


def create_dataset(labels_file, part1_sizes, part2_sizes, target_size_gb):
    # Load labels
    with open(labels_file, 'r') as f:
        labels = json.load(f)

    # Combine file sizes
    all_sizes = {**part1_sizes, **part2_sizes}

    # Create a list of (filename, size) tuples, sorted by size
    sorted_files = sorted(all_sizes.items(), key=lambda x: x[1])

    target_size_mb = target_size_gb * 1024
    selected_files = []
    current_size = 0

    # Randomly select files, prioritizing smaller ones
    while current_size < target_size_mb and sorted_files:
        # Randomly choose from the smallest 10% of remaining files
        chunk_size = max(1, len(sorted_files) // 10)
        chosen_file, file_size = random.choice(sorted_files[:chunk_size])

        if chosen_file in labels and (current_size + file_size) <= target_size_mb:
            selected_files.append(chosen_file)
            current_size += file_size

        sorted_files.remove((chosen_file, file_size))

    # Create a new labels dictionary with only the selected files
    selected_labels = {file: labels[file] for file in selected_files if file in labels}

    return selected_files, selected_labels, current_size / 1024  # Convert back to GB


# File paths
labels_file = '/Volumes/SSD/BVIArtefact/processed_labels.json'
part1_sizes_file = '/Volumes/SSD/BVIArtefact/part1_files_sizes.txt'
part2_sizes_file = '/Volumes/SSD/BVIArtefact/part1_files_sizes.txt'

# Target dataset size in GB
target_size_gb = 2  # Change this to your desired size

# Get file sizes
part1_sizes = get_file_sizes(part1_sizes_file)
part2_sizes = get_file_sizes(part2_sizes_file)

# Create the dataset
selected_files, selected_labels, actual_size_gb = create_dataset(
    labels_file, part1_sizes, part2_sizes, target_size_gb
)

# Print results
print(f"Selected {len(selected_files)} files")
print(f"Total size: {actual_size_gb:.2f} GB")

# Save the new labels to a file
output_dir = '/Volumes/SSD/BVIArtefact'
with open(os.path.join(output_dir, 'selected_labels.json'), 'w') as f:
    json.dump(selected_labels, f, indent=2)

# Save the list of selected files
with open(os.path.join(output_dir, 'selected_files.txt'), 'w') as f:
    for file in selected_files:
        f.write(f"{file}\n")

print("Selected labels saved to 'selected_labels.json'")
print("Selected files list saved to 'selected_files.txt'")
