# resize_bvi_artefact.py

import multiprocessing
import os
import re
import shutil

import ffmpeg
from tqdm import tqdm


def resize_video(input_path, output_path, width=224, height=224):
    try:
        (
            ffmpeg
            .input(input_path)
            .filter('scale', width, height)
            .output(output_path)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        return None  # Success
    except ffmpeg.Error as e:
        return f"Error processing {input_path}: {e.stderr.decode()}"


def get_new_filename(old_filename, width, height):
    pattern = r'(.+)_(\d+x\d+)_(\d+fps)_(.+)\.avi'
    match = re.match(pattern, old_filename)

    if match:
        video_name, old_resolution, fps, rest = match.groups()
        return f"{video_name}_{old_resolution}_to_{width}x{height}_{fps}_{rest}.avi"
    else:
        name, ext = os.path.splitext(old_filename)
        return f"{name}_to_{width}x{height}{ext}"


def process_video(args):
    input_path, output_dir, relative_path, width, height = args
    file = os.path.basename(input_path)
    new_filename = get_new_filename(file, width, height)
    output_path = os.path.join(output_dir, relative_path, new_filename)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return resize_video(input_path, output_path, width, height)


def preprocess_dataset(input_dir, output_dir, width=560, height=560, num_processes=None):
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    video_files = []
    for part in ['part1', 'part2']:
        part_dir = os.path.join(input_dir, part)
        print(f"Searching for videos in: {part_dir}")
        if not os.path.exists(part_dir):
            print(f"Directory not found: {part_dir}")
            continue
        for root, _, files in os.walk(part_dir):
            for file in files:
                if file.endswith('.avi'):
                    relative_path = os.path.relpath(root, input_dir)
                    input_path = os.path.join(root, file)
                    video_files.append((input_path, output_dir, relative_path, width, height))

    print(f"Found {len(video_files)} video files to process.")

    if not video_files:
        print("No video files found. Please check the input directory.")
        return

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_video, video_files), total=len(video_files), desc="Processing videos"))

    # Print any errors that occurred
    errors = [error for error in results if error is not None]
    for error in errors:
        print(error)

    # Copy json files to the output directory
    json_files = ['labels.json', 'processed_labels.json', 'subsets.json']
    for json_file in json_files:
        src = os.path.join(input_dir, json_file)
        dst = os.path.join(output_dir, json_file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"Warning: {json_file} not found in {input_dir}")

    print(f"Preprocessing completed! Processed {len(video_files)} videos with {len(errors)} errors.")


if __name__ == "__main__":
    input_dir = "/Volumes/SSD/BVIArtefact"
    output_dir = "/Volumes/SSD/preprocessed_BVIArtefact"

    # Get the full path of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct full paths for input and output directories
    input_dir = os.path.join(script_dir, input_dir)
    output_dir = os.path.join(script_dir, output_dir)

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    preprocess_dataset(input_dir, output_dir)
