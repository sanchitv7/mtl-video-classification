import os
import torch
import random
import json
from torchvision.io import read_video
from transformers import VideoMAEImageProcessor
from pathlib import Path

# Load the VideoMAE image processor
model_ckpt = "MCG-NJU/videomae-base"
image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt,
                                                         do_rescale=False)


def random_spatio_temporal_crop(video, num_frames=16, height=224, width=224):
    T, H, W, C = video.shape

    # Random temporal crop
    start_frame = random.randint(0, T - num_frames)
    video = video[start_frame:start_frame + num_frames]

    # Random spatial crop
    if H > height and W > width:
        top = random.randint(0, H - height)
        left = random.randint(0, W - width)
        video = video[:, top:top + height, left:left + width, :]
    else:
        video = torch.nn.functional.interpolate(video.permute(0, 3, 1, 2), size=(height, width)).permute(0, 2, 3, 1)

    return video


def preprocess_video(video_path, num_crops=6, num_frames=16, height=224, width=224):
    video, _, _ = read_video(video_path, pts_unit="sec")
    video = video.float() / 255.0  # Normalize to [0, 1]

    crops = []
    for _ in range(num_crops):
        crop = random_spatio_temporal_crop(video, num_frames, height, width)
        # Apply VideoMAE preprocessing
        crop = image_processor(list(crop.permute(0, 3, 1, 2)), return_tensors="pt")["pixel_values"]
        crops.append(crop.squeeze(0))  # Remove batch dimension

    return torch.stack(crops)  # Stack all crops


def main():
    dataset_root_path = "/Volumes/SSD/BVIArtefact"
    output_root_path = "/Volumes/SSD/BVIArtefact_preprocessed"
    os.makedirs(output_root_path, exist_ok=True)

    # Load original labels
    with open(os.path.join(dataset_root_path, "processed_labels.json"), "r") as f:
        original_labels = json.load(f)

    # New labels dictionary
    new_labels = {}

    # Process videos
    for part in ["part1", "part2"]:
        part_dir = os.path.join(dataset_root_path, part)
        for video_name in os.listdir(part_dir):
            if video_name.endswith('.avi'):
                video_path = os.path.join(part_dir, video_name)

                if video_name in original_labels:
                    try:
                        preprocessed_crops = preprocess_video(video_path)

                        # Save preprocessed video crops
                        output_name = f"{Path(video_name).stem}_crops.pt"
                        output_path = os.path.join(output_root_path, output_name)
                        torch.save(preprocessed_crops, output_path)

                        # Add to new labels dictionary
                        new_labels[output_name] = original_labels[video_name]

                        print(f"Processed {video_name}")
                    except Exception as e:
                        print(f"Error processing {video_name}: {str(e)}")
                else:
                    print(f"Skipping {video_name} - not found in labels")

    # Save the new labels
    with open(os.path.join(output_root_path, "preprocessed_labels.json"), "w") as f:
        json.dump(new_labels, f)

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
