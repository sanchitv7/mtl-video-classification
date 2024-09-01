# dataset structure:
'''
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


# Import necessary libraries
import os
import json
import torch
import numpy as np
from transformers import VivitImageProcessor, VivitForVideoClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from torchvision.io import read_video
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from multiprocessing import Pool
import functools


def load_video(video_path):
    # Read the video file
    video, _, info = read_video(video_path, pts_unit='sec')

    # Set the number of frames we want to sample
    num_frames_to_sample = 32

    # Get the total number of frames in the video
    total_frames = video.shape[0]

    # Calculate the sampling rate to evenly distribute frames
    sampling_rate = max(total_frames // num_frames_to_sample, 1)

    # Sample frames at the calculated rate
    sampled_frames = video[::sampling_rate][:num_frames_to_sample]

    # If we don't have enough frames, pad with zeros
    if sampled_frames.shape[0] < num_frames_to_sample:
        padding = torch.zeros(
            (num_frames_to_sample - sampled_frames.shape[0], *sampled_frames.shape[1:]), dtype=sampled_frames.dtype)
        sampled_frames = torch.cat([sampled_frames, padding], dim=0)

    # Ensure we have exactly the number of frames we want
    sampled_frames = sampled_frames[:num_frames_to_sample]

    # Convert to numpy array and change to channel-first format (C, H, W)
    return sampled_frames.permute(0, 3, 1, 2).numpy()


def create_dataset(data_dir, split):
    # Construct the path to the video directory and labels file
    video_dir = os.path.join(data_dir, split)
    json_path = os.path.join(video_dir, 'labels.json')

    # Load the labels from the JSON file
    with open(json_path, 'r') as f:
        labels = json.load(f)

    # Get all video files in the directory
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.avi')]

    # Create a dataset with video paths and their corresponding labels
    dataset = Dataset.from_dict({
        'video_path': [os.path.join(video_dir, f) for f in video_files],
        'label': [labels[f]['graininess'] for f in video_files]
    })

    return dataset


# Load the ViViT image processor
image_processor = VivitImageProcessor.from_pretrained(
    "google/vivit-b-16x2-kinetics400")


def preprocess_video(example, image_processor):
    # Load the video
    video = load_video(example['video_path'])

    # Process the video frames using the ViViT image processor
    inputs = image_processor(list(video), return_tensors="np")

    # Add the processed inputs to the example dictionary
    for k, v in inputs.items():
        example[k] = v.squeeze()  # Remove batch dimension

    return example


def preprocess_dataset(dataset, num_proc=4):
    # Use multiprocessing to preprocess the dataset in parallel
    return dataset.map(
        functools.partial(preprocess_video, image_processor=image_processor),
        remove_columns=['video_path'],
        num_proc=num_proc
    )


# Define the path to the dataset
data_dir = 'graininess_100_balanced_subset_split'

# Load the datasets for each split
dataset = DatasetDict({
    'train': create_dataset(data_dir, 'train'),
    'validation': create_dataset(data_dir, 'val'),
    'test': create_dataset(data_dir, 'test')
})

# Define the path where the preprocessed dataset will be saved
preprocessed_path = './preprocessed_dataset'

# Check if preprocessed dataset already exists
if os.path.exists(preprocessed_path):
    print("Loading preprocessed dataset...")
    # Load the preprocessed dataset from disk
    preprocessed_dataset = DatasetDict.load_from_disk(preprocessed_path)
else:
    print("Preprocessing dataset...")
    # Preprocess each split of the dataset
    preprocessed_dataset = DatasetDict({
        split: preprocess_dataset(dataset[split])
        for split in dataset.keys()
    })
    # Save the preprocessed dataset to disk
    preprocessed_dataset.save_to_disk(preprocessed_path)
    print("Preprocessed dataset saved to disk.")

# Load the ViViT model
model = VivitForVideoClassification.from_pretrained(
    "google/vivit-b-16x2-kinetics400")

# Modify the model for binary classification
model.classifier = torch.nn.Linear(model.config.hidden_size, 2)
model.num_labels = 2

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",  # Directory to save the model checkpoints
    num_train_epochs=3,  # Number of training epochs
    per_device_train_batch_size=2,  # Batch size for training
    per_device_eval_batch_size=2,  # Batch size for evaluation
    warmup_steps=500,  # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # Strength of weight decay
    logging_dir='./logs',  # Directory for storing logs
    logging_steps=10,  # Log every X updates steps
    evaluation_strategy="steps",  # Evaluate during training
    eval_steps=100,  # Evaluate every X steps
    save_steps=1000,  # Save checkpoint every X steps
    # Load the best model when finished training (default metric is loss)
    load_best_model_at_end=True,
)

# Define function to compute evaluation metrics


def compute_metrics(eval_pred):
    # Get the predictions and true labels
    predictions = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids

    # Compute precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary')

    # Compute accuracy
    accuracy = accuracy_score(labels, predictions)

    # Return all metrics
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# Initialize the Trainer
trainer = Trainer(
    model=model,  # The instantiated model to be trained
    args=training_args,  # Training arguments, defined above
    train_dataset=preprocessed_dataset['train'],  # Training dataset
    eval_dataset=preprocessed_dataset['validation'],  # Evaluation dataset
    compute_metrics=compute_metrics,  # The function that computes metrics
)

# Train the model
trainer.train()

# Evaluate the model on the test set
evaluation_results = trainer.evaluate(preprocessed_dataset['test'])
print(evaluation_results)

# Save the final model
trainer.save_model("./vivit_binary_classifier")

# Function to predict on new videos


def predict_video(video_path):
    # Load and preprocess the video
    video = load_video(video_path)
    inputs = image_processor(list(video), return_tensors="pt")

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)

    # Get probabilities and predicted class
    probabilities = torch.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

    return predicted_class, probabilities[0][predicted_class].item()




# Example usage of prediction function
# video_path = "path/to/your/video.avi"
# predicted_class, confidence = predict_video(video_path)
# print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}")
