import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.io import read_video
from torchvision.models import ResNet50_Weights

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Define paths
data_path = "data/graininess_100_balanced_subset_split"
train_path = os.path.join(data_path, "train")
val_path = os.path.join(data_path, "val")
test_path = os.path.join(data_path, "test")

# Define artifact (can be extended for multi-task later)
artifact = "graininess"


# Helper function to load labels
def load_labels(split_path):
    with open(os.path.join(split_path, "labels.json"), "r") as f:
        return json.load(f)


# Custom dataset class
class VideoDataset(Dataset):
    def __init__(self, root_dir, labels, artifact):
        self.root_dir = root_dir
        self.labels = labels
        self.artifact = artifact
        self.video_files = [f for f in os.listdir(root_dir) if f.endswith('.avi')]
        self.transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        ])

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_name = self.video_files[idx]
        video_path = os.path.join(self.root_dir, video_name)
        label = self.labels[video_name][self.artifact]

        # Load video using torchvision
        video, _, _ = read_video(video_path, pts_unit='sec')

        # Subsample frames (adjust as needed)
        video = video[::video.shape[0] // 16][:16]

        # Apply normalization
        video = self.transform(video)

        # Rearrange dimensions to [C, T, H, W]
        video = video.permute(3, 0, 1, 2)

        return video, torch.tensor(label, dtype=torch.float32)


# Create datasets
train_labels = load_labels(train_path)
val_labels = load_labels(val_path)
test_labels = load_labels(test_path)

train_dataset = VideoDataset(train_path, train_labels, artifact)
val_dataset = VideoDataset(val_path, val_labels, artifact)
test_dataset = VideoDataset(test_path, test_labels, artifact)

# Create data loaders
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


# Define model
class VideoClassifier(nn.Module):
    def __init__(self, num_classes=1):
        super(VideoClassifier, self).__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.conv1 = nn.Conv2d(16, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        b, c, t, h, w = x.shape
        x = x.transpose(1, 2).reshape(b * t, c, h, w)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = x.reshape(b, t, -1).mean(1)
        x = self.fc(x)
        return torch.sigmoid(x)


model = VideoClassifier().to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for videos, labels in train_loader:
        videos, labels = videos.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = (outputs.squeeze() > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for videos, labels in val_loader:
            videos, labels = videos.to(device), labels.to(device)

            outputs = model(videos)
            loss = criterion(outputs.squeeze(), labels)

            running_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    print()


# Test function
def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for videos, labels in test_loader:
            videos, labels = videos.to(device), labels.to(device)

            outputs = model(videos)
            loss = criterion(outputs.squeeze(), labels)

            running_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = running_loss / len(test_loader)
    test_acc = correct / total
    return test_loss, test_acc


# Evaluate on test set
test_loss, test_acc = test(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

# Save the model
torch.save(model.state_dict(), f"video_classifier_{artifact}.pth")
print(f"Model saved as video_classifier_{artifact}.pth")
