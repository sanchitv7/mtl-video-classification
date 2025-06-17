# Video Artifact Classification

This project implements a deep learning solution for detecting video artifacts, specifically focusing on graininess detection in videos. It uses the ViViT (Vision Transformer for Video) architecture and implements multi-task learning capabilities.

## Project Structure

```
mtl-video-classification/
├── data/                      # Dataset directory
│   └── graininess_100_balanced_subset_split/
│       ├── train/            # Training videos and labels
│       ├── val/              # Validation videos and labels
│       └── test/             # Test videos and labels
├── src/                      # Source code
│   ├── mtl_video_classification/  # Main package
│   └── data_prep_utils/      # Data preprocessing utilities
├── notebooks/                # Jupyter notebooks for experimentation
├── logs/                     # Training logs
└── requirements.txt          # Project dependencies
```

## Features

- Video artifact detection (graininess)
- Multi-task learning support
- Efficient video processing pipeline
- Data augmentation
- Mixed precision training
- Model checkpointing and evaluation

## Requirements

- Python 3.8+
- PyTorch
- CUDA-compatible GPU (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mtl-video-classification.git
cd mtl-video-classification
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

The dataset consists of video files in `.avi` format, organized into train/val/test splits. Each split contains:
- Video files
- `labels.json` file with binary labels for graininess (0 or 1)

Dataset structure:
```
data/graininess_100_balanced_subset_split/
├── train/
│   ├── video1.avi
│   ├── video2.avi
│   └── labels.json
├── val/
│   ├── video3.avi
│   └── labels.json
└── test/
    ├── video4.avi
    └── labels.json
```

## Usage

### Training

1. Prepare your dataset following the structure above
2. Run the training script:
```bash
python src/mtl_video_classification/train.py
```

### Inference

To classify a new video:
```python
from mtl_video_classification.predict import predict_video

# The function returns both the predicted class and confidence score
predicted_class, confidence = predict_video("path/to/video.avi")
print(f"Prediction: {'Grainy' if predicted_class == 1 else 'Not Grainy'}")
print(f"Confidence: {confidence:.2f}")
```

## Model Architecture

The project uses the ViViT (Vision Transformer for Video) model, specifically the `google/vivit-b-16x2-kinetics400` pre-trained model. The model is modified for binary classification and includes:

- Video frame sampling
- Spatial and temporal cropping
- Multi-task learning adapters
- Focal loss for handling class imbalance

## Training Process

1. **Data Preprocessing**:
   - Frame sampling (32 frames per video)
   - Spatial cropping (224x224)
   - Temporal cropping
   - Normalization

2. **Training Configuration**:
   - Mixed precision training (FP16)
   - Gradient accumulation
   - Data augmentation
   - Model checkpointing

3. **Evaluation**:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
