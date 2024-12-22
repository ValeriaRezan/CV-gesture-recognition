from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.hub
import numpy as np
import os
import torch
import torch.nn as nn
from torchvision.models import resnet18
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class JesterDataset(Dataset):
    def __init__(self, frames_dir, train_csv, labels_txt, transform=None, num_frames=8):
        """
        Args:
            frames_dir (str): Path to folder containing all video subfolders.
            train_csv (str): Path to the train CSV with lines like "video_id;gesture_name".
            labels_txt (str): Path to the file listing all gestures in order (27 lines).
            transform (callable, optional): A function/transform to apply to each frame.
            num_frames (int): Number of frames to load for each video.
        """
        self.frames_dir = frames_dir
        self.transform = transform
        self.num_frames = num_frames

        # 1) Build a dictionary mapping gesture_name -> label index
        self.label_to_index = {}
        with open(labels_txt, "r") as f:
            for idx, line in enumerate(f):
                gesture_name = line.strip()
                self.label_to_index[gesture_name] = idx

        # 2) Read the train CSV (video_id;gesture_name) lines
        self.labels = []
        with open(train_csv, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                video_id, gesture_name = line.split(';')
                gesture_name = gesture_name.strip()
                label_idx = self.label_to_index[gesture_name]
                self.labels.append((video_id, label_idx))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        video_id, label = self.labels[idx]
        frames_path = os.path.join(self.frames_dir, video_id)

        # Get frame filenames, sorted
        frames_list = sorted(os.listdir(frames_path))

        total_frames = len(frames_list)

        # Handle edge cases where total_frames is less than the number of segments
        if total_frames < self.num_frames:
            raise ValueError(
                f"Video {video_id} has fewer frames ({total_frames}) than the number of segments ({self.num_frames}).")

        # Calculate segment length
        segment_length = total_frames // self.num_frames

        # Sample one frame from each segment
        frame_indices = [
            np.random.randint(i * segment_length, (i + 1) * segment_length)
            for i in range(self.num_frames)
        ]

        # Extract the frames at the selected indices
        frames_list = [frames_list[i] for i in frame_indices]

        # Load frames and apply transforms
        frames = []
        for frame_name in frames_list:
            frame_full_path = os.path.join(frames_path, frame_name)
            img = Image.open(frame_full_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        # Stack frames => shape: (n_segments, 3, H, W)
        frames_tensor = torch.stack(frames, dim=0)
        return frames_tensor, label


class Backbone(nn.Module):
    def __init__(self, pretrained=True, n_div=8, n_frames=8):
        super(Backbone, self).__init__()
        self.n_div = n_div
        self.n_frames = n_frames

        # Load the pretrained ResNet-18 model
        self.backbone = resnet18(pretrained=pretrained)

        # Remove the fully connected layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # Define the TemporalShift module
        self.temporal_shift = TemporalShift(n_div=n_div, n_frames=n_frames)

    def forward(self, x):
        batch_size, n_frames, channels, height, width = x.size()

        # Merge batch and temporal dimensions for 2D CNN processing
        x = x.view(-1, channels, height, width)

        # Pass through each block of the ResNet backbone
        out = x

        for layer in self.backbone:
            out = layer(out)  # Pass through the residual block
            # Apply temporal shift after each residual block
            out = self.temporal_shift(out)

        return out

class TemporalShift(nn.Module):paper,
    def __init__(self, n_frames=8, n_div=8):
        """
        Temporal Shift Module.

        Args:
            n_frames (int): Number of frames in the input tensor.
            n_div (int): Division factor to determine the shift size.
        """
        super(TemporalShift, self).__init__()
        self.n_frames = n_frames
        self.fold_div = n_div

    def shift(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_frames
        x = x.view(n_batch, self.n_frames, c, h, w)
        fold = c // self.fold_div

        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # no shift for the rest

        return out.view(nt, c, h, w)


    def forward(self, x):
        return self.shift(x)


class TSMResNet(nn.Module):
    def __init__(self, num_classes=27, pretrained=True, n_frames=8, n_div=8):
        super(TSMResNet, self).__init__()
        self.n_frames = n_frames
        self.n_div = n_div

        # Backbone with 2D convolution and TemporalShift after each block
        self.backbone = Backbone(pretrained, n_frames, n_div)

        # Global Pooling + Classification Head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

        # Freeze the backbone layers (set requires_grad=False)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the layers for training (classification head)
        for param in self.global_pool.parameters():
            param.requires_grad = True

        for param in self.fc.parameters():
            param.requires_grad = True

    def forward(self, x):

        batch_size, n_frames, c, h, w = x.shape
        # Process through the backbone
        x = self.backbone(x)  # Process with the modified ResNet backbone

        # Apply global average pooling
        x = self.global_pool(x)

        # Flatten and classify
        x = x.view(batch_size, n_frames, -1).mean(dim=1)  # Average over time
        x = self.fc(x)  # Fully Connected Layer
        return x


def evaluate(model, test_loader, criterion, device):
    """
    Evaluate the model on the test set.

    Args:
        model (nn.Module): Model to evaluate.
        test_loader (torch.utils.data.DataLoader): Data loader for the test set.
        criterion (callable): Loss function to use for evaluation.
        device (torch.device): Device to use for evaluation.

    Returns:
        float: Average loss on the test set.
        float: Accuracy on the test set.
    """
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        total_loss = 0.0
        num_correct = 0
        num_samples = 0

        for inputs, labels in test_loader:
            # Move inputs and labels to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Compute the logits (model output)
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Compute the accuracy (assuming logits are the class scores)
            _, predictions = torch.max(logits, dim=1)
            num_correct += (predictions == labels).sum().item()
            num_samples += len(inputs)

    avg_loss = total_loss / len(test_loader)
    accuracy = num_correct / num_samples

    return avg_loss, accuracy





def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs):
    """
    Train the model on the training set and evaluate it on the validation set every epoch.

    Args:
        model (nn.Module): Model to train.
        train_loader (torch.utils.data.DataLoader): Data loader for the training set.
        val_loader (torch.utils.data.DataLoader): Data loader for the validation set.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (callable): Loss function for training.
        device (torch.device): Device to use for training.
        num_epochs (int): Number of epochs to train the model.
    """
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model = model.to(device)  # Move model to device

    for epoch in range(num_epochs):
        metrics = []
        model.train()  # Set model to training mode

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', position=0, leave=True) as pbar:
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()  # Zero the gradients

                logits = model(inputs)  # Forward pass
                loss = criterion(logits, labels)  # Compute loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Optimize weights

                pbar.update(1)  # Update progress bar
                pbar.set_postfix(loss=loss.item())  # Display current loss

            # Evaluate on validation set after each epoch
            avg_loss, accuracy = evaluate(model, val_loader, criterion, device)
            metrics.append( avg_loss)
            print(f'Validation set: Average loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}')

        scheduler.step()  # Update learning rate
    return metrics


def get_train_val_loaders(frames_dir, train_csv, test_csv, labels_txt, transform, num_frames, val_split=0.1, batch_size=4):
    # Load the full dataset
    train_dataset = JesterDataset(frames_dir, train_csv, labels_txt, transform=transform, num_frames=num_frames)
    test_dataset = JesterDataset(frames_dir, test_csv, labels_txt, transform=transform, num_frames=num_frames)
    # Calculate split index
    total_samples = len(train_dataset)
    val_size = int(total_samples * val_split)
    train_size = total_samples - val_size

    # Randomly split dataset
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create DataLoader for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, pin_memory_device=device.type, prefetch_factor=2 )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, pin_memory_device=device.type, prefetch_factor=2 )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, pin_memory_device=device.type, prefetch_factor=2 )
    return train_loader, val_loader, test_loader


def plot_loss(loss_list):
    """
    Plots the loss curve over the epochs.

    Args:
        loss_list (list): List of loss values over epochs.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_list) + 1), loss_list, marker='o', linestyle='-', color='b', label='Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def main():
    frames = 8
    shifts = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model
    model = TSMResNet(n_frames=frames, n_div=shifts)

    # Define the loss function (CrossEntropyLoss for classification)
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer (Adam in this case)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    frames_dir = "data/20bn-jester-v1/"
    train_csv = "jester-v1-train.csv"  # your "video_id;gesture_name" file
    test_csv = "jester-v1-validation.csv"
    labels_txt = "jester-v1-labels.csv"  # the 27-line gestures file
    train_loader, val_loader, test_loader = get_train_val_loaders(frames_dir, train_csv, test_csv, labels_txt, transform, num_frames=frames)


    # Proceed with training
    num_epochs = 5  # Set the number of epochs you want to train the model
    scores = train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs)
    plot_loss(scores)
    avg_loss, accuracy = evaluate(model, test_loader, criterion, device)
    print(f'Test set: Average loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}')


if __name__ == '__main__':
    main()


