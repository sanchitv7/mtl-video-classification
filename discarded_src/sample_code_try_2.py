'''
# dataset structure:
data/graininess_100_balanced_subset_split
├── test
│   ├── BirdsInCage_1920x1080_30fps_8bit_420_Pristine_QP32_FBT_1.avi
│   ├── Chimera1_4096x2160_60fps_10bit_420_graininess_QP32_FB_1.avi
│   ├── Chimera3_4096x2160_24fps_10bit_420_graininess_QP32_FT_1.avi
│   ├── ...
│   └── labels.json
├── train
│   ├── labels.json
│   ├── lamppost_1920x1080_120fps_8bit_420_Pristine_QP32_BT_3.avi
│   ├── lamppost_1920x1080_120fps_8bit_420_Pristine_QP47_SF_3.avi
│   ├── leaveswall_1920x1080_120fps_8bit_420_Motion_QP32_SB_1.avi
│   ├── leaveswall_1920x1080_120fps_8bit_420_Motion_QP32_SFB_4.avi
│   ├── library_1920x1080_120fps_8bit_420_aliasing_QP47_FT_1.avi
│   ├── ...
└── val
    ├── Chimera2_4096x2160_60fps_10bit_420_Dark_QP32_BT_1.avi
    ├── ...
    ├── labels.json
    ├── shields_1280x720_50fps_8bit_420_graininess_QP47_SFB_1.avi
    ├── station_1920x1080_30fps_8bit_420_graininess_QP32_SB_1.avi
    ├── svtmidnightsun_3840x2160_50fps_10bit_420_banding_QP47_SBT_3.avi
    ├── svtmidnightsun_3840x2160_50fps_10bit_420_banding_QP47_SFT_1.avi
    ├── svtsmokesauna_3840x2160_50fps_10bit_420_banding_QP32_F_4.avi
    ├── svtwaterflyover_3840x2160_50fps_10bit_420_banding_QP32_T_3.avi
    └── typing_1920x1080_120fps_8bit_420_aliasing_QP47_BT_4.avi

4 directories, 103 files
'''

'''
labels.json in each split is like:
{
  "Chimera1_4096x2160_60fps_10bit_420_graininess_QP47_FT_1.avi": {
    "graininess": 1
  },
  "riverbed_1920x1080_25fps_8bit_420_banding_QP47_SBT_1.avi": {
    "graininess": 0
  },
  "Meridian1_3840x2160_60fps_10bit_420_banding_QP47_SFT_1.avi": {
    "graininess": 0
  },
  '''

import os
import json
import torch
import numpy as np
from transformers import VivitImageProcessor, VivitForVideoClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from torchvision.io import read_video
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from functools import partial


def get_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        ToTensorV2(),
    ])


def apply_augmentation(frames, augmentation):
    aug_frames = []
    for frame in frames:
        augmented = augmentation(image=frame)
        aug_frames.append(augmented['image'])
    return torch.stack(aug_frames)


def uniform_frame_sample(video, num_frames):
    total_frames = len(video)
    if total_frames <= num_frames:
        return video

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    return video[indices]


def load_video(video_path, num_frames=32, augmentation=None):
    video, _, info = read_video(video_path, pts_unit='sec')

    # Uniform sampling
    sampled_frames = uniform_frame_sample(video, num_frames)

    if augmentation:
        sampled_frames = apply_augmentation(sampled_frames, augmentation)

    return sampled_frames.permute(0, 3, 1, 2).float() / 255.0


def create_dataset(data_dir, split):
    video_dir = os.path.join(data_dir, split)
    json_path = os.path.join(video_dir, 'labels.json')
    with open(json_path, 'r') as f:
        labels = json.load(f)

    video_files = [f for f in os.listdir(video_dir) if f.endswith('.avi')]

    dataset = Dataset.from_dict({
        'video_path': [os.path.join(video_dir, f) for f in video_files],
        'label': [labels[f]['graininess'] for f in video_files]
    })

    return dataset


# Load the image processor
image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")


def preprocess_video(example, image_processor, augmentation=None):
    video = load_video(example['video_path'], augmentation=augmentation)
    inputs = image_processor(list(video), return_tensors="pt")
    for k, v in inputs.items():
        example[k] = v.squeeze()
    return example


def preprocess_dataset(dataset, augmentation=None):
    return dataset.map(
        partial(preprocess_video, image_processor=image_processor, augmentation=augmentation),
        remove_columns=['video_path'],
        num_proc=4
    )


# Load and preprocess the datasets
data_dir = 'graininess_100_balanced_subset_split'
dataset = DatasetDict({
    'train': create_dataset(data_dir, 'train'),
    'validation': create_dataset(data_dir, 'val'),
    'test': create_dataset(data_dir, 'test')
})

augmentation = get_augmentation()

preprocessed_path = './preprocessed_dataset_augmented'
if os.path.exists(preprocessed_path):
    print("Loading preprocessed dataset...")
    preprocessed_dataset = DatasetDict.load_from_disk(preprocessed_path)
else:
    print("Preprocessing dataset with augmentation...")
    preprocessed_dataset = DatasetDict({
        'train': preprocess_dataset(dataset['train'], augmentation),
        'validation': preprocess_dataset(dataset['validation']),
        'test': preprocess_dataset(dataset['test'])
    })
    preprocessed_dataset.save_to_disk(preprocessed_path)
    print("Preprocessed dataset saved to disk.")

# Load the model
model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")
model.classifier = torch.nn.Linear(model.config.hidden_size, 2)
model.num_labels = 2

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=1000,
    load_best_model_at_end=True,
    fp16=True,  # Enable mixed precision training
    gradient_accumulation_steps=2,  # Accumulate gradients over 2 steps
)


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    accuracy = accuracy_score(labels, predictions)
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=preprocessed_dataset['train'],
    eval_dataset=preprocessed_dataset['validation'],
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model
evaluation_results = trainer.evaluate(preprocessed_dataset['test'])
print(evaluation_results)

# Save the model
trainer.save_model("./vivit_binary_classifier_augmented")


def predict_video(video_path):
    video = load_video(video_path)
    inputs = image_processor(list(video), return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    return predicted_class, probabilities[0][predicted_class].item()

# Example usage of prediction function
# video_path = "path/to/your/video.avi"
# predicted_class, confidence = predict_video(video_path)
# print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}")
